"""Offline unit & API tests for Milestone 2: Real Interactive Student Assessment Loop."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.assessment import (
    ASSESSMENT_ITEMS,
    SKILL_DIFFICULTY_MAP,
    build_socratic_intervention_prompt,
    compute_11_engineered_features,
)
from src.features import ENGINEERED_FEATURES
from src.storage import get_storage, reset_storage


def test_assessment_bank_has_18_calibrated_items_covering_all_5_skills():
    assert 15 <= len(ASSESSMENT_ITEMS) <= 20
    skills_seen = set()
    ids_seen = set()
    for item in ASSESSMENT_ITEMS:
        assert item["item_id"] not in ids_seen
        ids_seen.add(item["item_id"])
        skills_seen.add(item["skill_name"])
        assert len(item["options"]) == 4
        assert item["correct_answer"] in item["options"]
        assert len(item["hints"]) >= 2
        assert bool(item["bottom_hint_text"])
        assert 0.0 < item["problem_difficulty"] < 1.0
        assert 0.0 < item["skill_difficulty"] < 1.0
    assert skills_seen == set(SKILL_DIFFICULTY_MAP.keys())


def test_compute_11_engineered_features_exact_schema_and_rapid_guess():
    feats_rapid = compute_11_engineered_features(
        opportunity=1,
        attempt_count=1,
        hints_used=0,
        hint_total=3,
        bottom_hint_used=False,
        user_prev_acc=0.70,
        problem_difficulty=0.75,
        skill_difficulty=0.68,
        is_skill_unmapped=0,
        ms_first_response=1500.0,
    )
    assert list(feats_rapid.keys()) == list(ENGINEERED_FEATURES)
    assert feats_rapid["is_rapid_guess"] == 1
    assert feats_rapid["log_opportunity"] == round(math.log1p(1), 6)
    assert feats_rapid["log_response_time"] == round(math.log1p(1500.0), 6)

    feats_struggle = compute_11_engineered_features(
        opportunity=4,
        attempt_count=5,
        hints_used=3,
        hint_total=3,
        bottom_hint_used=True,
        user_prev_acc=0.25,
        problem_difficulty=0.48,
        skill_difficulty=0.64,
        is_skill_unmapped=0,
        ms_first_response=28000.0,
    )
    assert feats_struggle["is_rapid_guess"] == 0
    assert feats_struggle["bottom_hint"] == 1
    assert feats_struggle["hint_ratio"] == 1.0


def test_live_assessment_full_loop_feature_store_xai_and_no_answer_leak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from src.induce_drift import generate_synthetic_baseline

    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("LIVE_INTERACTIONS_PATH", str(tmp_path / "live.csv"))
    monkeypatch.setenv("DRIFT_REPORTS_DIR", str(tmp_path / "drift"))
    reset_storage()

    import app.main as main_mod

    if not Path(main_mod.DEFAULT_MODEL_PATH).exists():
        synth = generate_synthetic_baseline(n_rows=300, seed=7)
        X_s = synth[list(ENGINEERED_FEATURES)].astype(float)
        y_s = synth["correct"].astype(int)
        # Ensure strong directional response for struggle (high attempts/hints -> 0)
        y_s = ((X_s["attempt_count"] <= 1.5) & (X_s["bottom_hint"] < 0.5)).astype(int)
        rf = RandomForestClassifier(n_estimators=20, max_depth=6, random_state=42)
        rf.fit(X_s, y_s)
        tmp_pkl = tmp_path / "temp_baseline.pkl"
        joblib.dump(rf, tmp_pkl)
        monkeypatch.setattr(main_mod, "MODEL_PATH", str(tmp_pkl))
        main_mod.reset_model_cache()

    client = TestClient(main_mod.app)

    # 1. List items (auto-seeds 18 items, never leaks correct_answer)
    res_list = client.get("/assessment/items")
    assert res_list.status_code == 200
    body = res_list.json()
    assert body["count"] == len(ASSESSMENT_ITEMS)
    for it in body["items"]:
        assert "correct_answer" not in it
        assert "bottom_hint_text" not in it
        assert it["hint_total"] == 3

    # Filter by skill
    phish_res = client.get("/assessment/items", params={"skill_name": "phishing_cyber_hygiene"})
    assert phish_res.status_code == 200
    assert phish_res.json()["count"] == 4

    # 2. Start timed session on dl_phish_01
    start_res = client.post(
        "/assessment/start",
        json={"student_id": "stu_live_1", "item_id": "dl_phish_01"},
    )
    assert start_res.status_code == 200
    start_data = start_res.json()
    sess_id = start_data["attempt_session_id"]
    assert start_data["student_context"]["opportunity"] == 1
    assert start_data["student_context"]["user_prev_acc"] == 0.7

    # 3. Request progressive hints up to bottom-out hint
    h0 = client.post(
        "/assessment/hint",
        json={"item_id": "dl_phish_01", "hint_index": 0, "attempt_session_id": sess_id},
    )
    assert h0.status_code == 200
    assert h0.json()["is_bottom_hint"] is False
    assert h0.json()["hints_used"] == 1

    h2 = client.post(
        "/assessment/hint",
        json={"item_id": "dl_phish_01", "hint_index": 2, "attempt_session_id": sess_id},
    )
    assert h2.status_code == 200
    assert h2.json()["is_bottom_hint"] is True
    assert h2.json()["hint_ratio"] == 1.0

    # 4. Submit WRONG answer (simulating struggling attempt with hints)
    item_full = next(it for it in ASSESSMENT_ITEMS if it["item_id"] == "dl_phish_01")
    wrong_opt = next(o for o in item_full["options"] if o != item_full["correct_answer"])
    sub1 = client.post(
        "/assessment/submit",
        json={
            "student_id": "stu_live_1",
            "item_id": "dl_phish_01",
            "selected_answer": wrong_opt,
            "attempt_session_id": sess_id,
            "attempt_count": 4,
            "ms_first_response": 35000.0,
        },
    )
    assert sub1.status_code == 200
    s1 = sub1.json()
    assert s1["is_correct"] is False
    assert s1["correct"] == 0
    assert s1["needs_intervention"] is True
    assert s1["trigger_tutor"] is True
    assert list(s1["computed_features"].keys()) == list(ENGINEERED_FEATURES)
    assert s1["computed_features"]["bottom_hint"] == 1
    assert s1["computed_features"]["hint_count"] == 3.0
    assert len(s1["explanation"]["attributions"]) == 11
    # Socratic prompt must NEVER leak the correct answer
    assert item_full["correct_answer"] not in (s1["suggested_tutor_prompt"] or "")
    # Feature Store updated: total_attempts=1, user_prev_acc=0.0, next opportunity=2
    assert s1["student_profile_after"]["total_attempts"] == 1
    assert s1["student_profile_after"]["user_prev_acc"] == 0.0
    assert s1["student_profile_after"]["opportunity"] == 2

    # 5. Next item automatically pulls updated pre-attempt user_prev_acc=0.0 & opportunity=2!
    item2_full = next(it for it in ASSESSMENT_ITEMS if it["item_id"] == "dl_phish_02")
    sub2 = client.post(
        "/assessment/submit",
        json={
            "student_id": "stu_live_1",
            "item_id": "dl_phish_02",
            "selected_answer": item2_full["correct_answer"],
            "attempt_count": 1,
            "hints_used": 0,
            "bottom_hint_used": False,
            "ms_first_response": 11000.0,
        },
    )
    assert sub2.status_code == 200
    s2 = sub2.json()
    assert s2["is_correct"] is True
    assert s2["correct"] == 1
    assert s2["computed_features"]["user_prev_acc"] == 0.0
    assert s2["computed_features"]["log_opportunity"] == round(math.log1p(2), 6)
    assert s2["student_profile_after"]["total_attempts"] == 2
    assert s2["student_profile_after"]["user_prev_acc"] == 0.5
    assert s2["student_profile_after"]["opportunity"] == 3

    # Verify both labeled rows were logged to telemetry
    df = get_storage().load_interactions(labeled_only=True)
    assert len(df) == 2
    assert list(df["correct"]) == [0, 1]

    # 6. Multi-attempt retry on same item session escalates attempt_count -> needs_intervention
    s_retry = client.post(
        "/assessment/start",
        json={"student_id": "stu_retry", "item_id": "dl_phish_03"},
    ).json()
    retry_sess = s_retry["attempt_session_id"]
    item3 = next(it for it in ASSESSMENT_ITEMS if it["item_id"] == "dl_phish_03")
    wrongs = [o for o in item3["options"] if o != item3["correct_answer"]]

    try1 = client.post(
        "/assessment/submit",
        json={
            "student_id": "stu_retry",
            "item_id": "dl_phish_03",
            "selected_answer": wrongs[0],
            "attempt_session_id": retry_sess,
            "hints_used": 1,
            "bottom_hint_used": False,
            "ms_first_response": 9000.0,
        },
    ).json()
    assert try1["attempt_number"] == 1
    assert try1["is_correct"] is False
    # 1 light hint on try 1 -> does NOT force auto-interrupt
    assert try1["intervention_tier"] in {"light_support", "mastery"}
    assert try1["trigger_tutor"] is False

    # Retry #2 after hints escalates attempt_count=2 and flips needs_intervention=True
    try2 = client.post(
        "/assessment/submit",
        json={
            "student_id": "stu_retry",
            "item_id": "dl_phish_03",
            "selected_answer": wrongs[1],
            "attempt_session_id": retry_sess,
            "hints_used": 2,
            "bottom_hint_used": True,
            "ms_first_response": 9000.0,
        },
    ).json()
    assert try2["attempt_number"] == 2
    assert try2["needs_intervention"] is True
    assert try2["trigger_tutor"] is True
    # Retrying same item does NOT double-increment student total_attempts (stays 1 problem)
    assert try2["student_profile_after"]["total_attempts"] == 1

    reset_storage()


def test_socratic_prompt_builder_never_leaks_answer():
    item = ASSESSMENT_ITEMS[0]
    p_wrong = build_socratic_intervention_prompt(
        item=item, selected_answer="wrong", is_correct=False, needs_intervention=True
    )
    assert p_wrong is not None
    assert item["correct_answer"] not in p_wrong

    p_ok = build_socratic_intervention_prompt(
        item=item, selected_answer=item["correct_answer"], is_correct=True, needs_intervention=False
    )
    assert p_ok is None
