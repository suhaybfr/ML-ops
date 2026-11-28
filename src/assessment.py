"""Graded Digital Literacy assessment bank & automated 11-feature engine (Milestone 2).

Provides 18 calibrated Digital Literacy items across the 5 core curriculum skills
(source credibility, digital privacy, phishing & cyber hygiene, search strategies,
software navigation) and computes the exact 11 canonical ``ENGINEERED_FEATURES``
automatically from live student interactions + the Online Feature Store.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from src.features import ENGINEERED_FEATURES
from src.storage import get_storage

SKILL_DIFFICULTY_MAP: dict[str, float] = {
    "source_credibility": 0.66,
    "digital_privacy": 0.71,
    "phishing_cyber_hygiene": 0.64,
    "search_strategies": 0.73,
    "software_navigation": 0.76,
}

ASSESSMENT_ITEMS: list[dict[str, Any]] = [
    # ── Source Credibility (4 items) ────────────────────────────────────────
    {
        "item_id": "dl_cred_01",
        "skill_name": "source_credibility",
        "prompt": (
            "You see a viral social-media post claiming a new law takes effect tomorrow. "
            "According to the SIFT method, what is the best first step after pausing?"
        ),
        "options": [
            "Read the comments section to see what most people think",
            "Open a new tab to investigate the author or publisher (lateral reading)",
            "Share it with friends and ask if it looks true",
            "Check how many likes and reposts the post has",
        ],
        "correct_answer": "Open a new tab to investigate the author or publisher (lateral reading)",
        "hints": [
            "Think about what the 'I' in SIFT stands for: Investigate the source.",
            "Professional fact-checkers leave the page and search the publisher in a new tab.",
        ],
        "bottom_hint_text": (
            "Lateral reading means opening a new tab to check who published the claim and "
            "whether they are a credible, accountable organisation."
        ),
        "problem_difficulty": 0.78,
        "skill_difficulty": SKILL_DIFFICULTY_MAP["source_credibility"],
        "is_skill_unmapped": 0,
    },
    {
        "item_id": "dl_cred_02",
        "skill_name": "source_credibility",
        "prompt": (
            "An online article cites a dramatic statistic from a '2019 study' but provides "
            "no link, author name, or dataset. Which CRAAP criteria are most clearly violated?"
        ),
        "options": [
            "Authority and Accuracy",
            "Relevance and Formatting",
            "Font readability and Page speed",
            "Popularity and Share count",
        ],
        "correct_answer": "Authority and Accuracy",
        "hints": [
            "Missing an identifiable author speaks to who is behind the claim.",
            "Missing verifiable evidence or citations speaks to whether the claim can be checked.",
        ],
        "bottom_hint_text": (
            "In the CRAAP checklist, Authority checks author credentials and Accuracy checks "
            "whether evidence is cited and verifiable."
        ),
        "problem_difficulty": 0.68,
        "skill_difficulty": SKILL_DIFFICULTY_MAP["source_credibility"],
        "is_skill_unmapped": 0,
    },
    {
        "item_id": "dl_cred_03",
        "skill_name": "source_credibility",
        "prompt": (
            "A dramatic flood photo is circulating as 'breaking news today', but you suspect "
            "it may be from an older event. Which technique best verifies its origin?"
        ),
        "options": [
            "Zoom in on the image resolution",
            "Run a reverse image search to trace where the photo first appeared",
            "Check the file size in kilobytes",
            "Read the caption written by the account that reposted it",
        ],
        "correct_answer": "Run a reverse image search to trace where the photo first appeared",
        "hints": [
            "The 'T' in SIFT stands for Trace claims and media back to their original context.",
            "Search engines let you upload or link an image to find earlier appearances online.",
        ],
        "bottom_hint_text": (
            "A reverse image search traces a photo back to its original publication date and event."
        ),
        "problem_difficulty": 0.74,
        "skill_difficulty": SKILL_DIFFICULTY_MAP["source_credibility"],
        "is_skill_unmapped": 0,
    },
    {
        "item_id": "dl_cred_04",
        "skill_name": "source_credibility",
        "prompt": (
            "A polished website at 'health-journal-daily.info' publishes an article praising "
            "a single brand of supplement and links directly to its checkout page without "
            "disclosure. Which warning sign is most critical here?"
        ),
        "options": [
            "Undisclosed commercial purpose / conflict of interest",
            "The article uses bullet points",
            "The page loads over HTTPS",
            "The website has a dark-mode toggle",
        ],
        "correct_answer": "Undisclosed commercial purpose / conflict of interest",
        "hints": [
            "Consider the 'P' in CRAAP: Purpose — why was this page written?",
            "Ask whether the site is informing neutrally or selling a product without saying so.",
        ],
        "bottom_hint_text": (
            "Evaluating Purpose reveals when content is actually an undisclosed advertisement "
            "or has a financial conflict of interest."
        ),
        "problem_difficulty": 0.52,
        "skill_difficulty": SKILL_DIFFICULTY_MAP["source_credibility"],
        "is_skill_unmapped": 0,
    },
    # ── Digital Privacy (4 items) ───────────────────────────────────────────
    {
        "item_id": "dl_priv_01",
        "skill_name": "digital_privacy",
        "prompt": (
            "A simple flashlight app asks for permission to access your contacts, microphone, "
            "and precise background location. What is the safest action?"
        ),
        "options": [
            "Grant all permissions so the app installs faster",
            "Deny unnecessary permissions or uninstall the app and choose a safer alternative",
            "Grant permissions now and ignore them later",
            "Create a second email account and grant the permissions",
        ],
        "correct_answer": (
            "Deny unnecessary permissions or uninstall the app and choose a safer alternative"
        ),
        "hints": [
            "Think about the principle of least privilege for mobile apps.",
            "Does a flashlight actually need your contacts or microphone to turn on an LED?",
        ],
        "bottom_hint_text": (
            "Only grant apps permissions they genuinely need to function; excessive requests "
            "are a privacy red flag."
        ),
        "problem_difficulty": 0.84,
        "skill_difficulty": SKILL_DIFFICULTY_MAP["digital_privacy"],
        "is_skill_unmapped": 0,
    },
    {
        "item_id": "dl_priv_02",
        "skill_name": "digital_privacy",
        "prompt": (
            "Which combination of account habits gives the strongest protection against "
            "credential-stuffing attacks across multiple websites?"
        ),
        "options": [
            "Reusing one complex password with symbols on every site",
            "Using a long, unique passphrase per account plus multi-factor authentication (MFA)",
            "Writing one short password and changing the last digit each year",
            "Turning off browser updates so cookies are not cleared",
        ],
        "correct_answer": (
            "Using a long, unique passphrase per account plus multi-factor authentication (MFA)"
        ),
        "hints": [
            "Credential stuffing happens when a password leaked from one site is tried on others.",
            "How do you stop a leaked password from unlocking your other accounts?",
        ],
        "bottom_hint_text": (
            "A unique passphrase per account prevents reuse attacks, and MFA blocks sign-ins "
            "even if a password is stolen."
        ),
        "problem_difficulty": 0.79,
        "skill_difficulty": SKILL_DIFFICULTY_MAP["digital_privacy"],
        "is_skill_unmapped": 0,
    },
    {
        "item_id": "dl_priv_03",
        "skill_name": "digital_privacy",
        "prompt": (
            "Which of the following is an example of your PASSIVE digital footprint rather "
            "than your active digital footprint?"
        ),
        "options": [
            "A public comment you post on a forum",
            "Third-party tracking cookies and IP/location logs collected while you browse",
            "A photo you upload to your profile",
            "A review you submit for a product",
        ],
        "correct_answer": (
            "Third-party tracking cookies and IP/location logs collected while you browse"
        ),
        "hints": [
            "Active footprint = things you intentionally publish or post.",
            "Passive footprint = data collected automatically in the background as you use sites.",
        ],
        "bottom_hint_text": (
            "Tracking cookies, device telemetry, and IP/location logs are gathered in the "
            "background without you actively posting them."
        ),
        "problem_difficulty": 0.61,
        "skill_difficulty": SKILL_DIFFICULTY_MAP["digital_privacy"],
        "is_skill_unmapped": 0,
    },
    {
        "item_id": "dl_priv_04",
        "skill_name": "digital_privacy",
        "prompt": (
            "You are connected to open café Wi-Fi with no password and need to check your "
            "bank account. What is the safest practice?"
        ),
        "options": [
            "Log in normally because café networks are private",
            "Use mobile cellular data or a trusted VPN instead of untrusted public Wi-Fi",
            "Disable HTTPS so the page loads faster",
            "Save your banking password on the café's shared kiosk",
        ],
        "correct_answer": (
            "Use mobile cellular data or a trusted VPN instead of untrusted public Wi-Fi"
        ),
        "hints": [
            "Open public Wi-Fi networks can be monitored or spoofed by attackers nearby.",
            "Think about how to encrypt your traffic or avoid the open network altogether.",
        ],
        "bottom_hint_text": (
            "Avoid sensitive logins on open public Wi-Fi; switch to cellular data or use a "
            "trusted VPN."
        ),
        "problem_difficulty": 0.72,
        "skill_difficulty": SKILL_DIFFICULTY_MAP["digital_privacy"],
        "is_skill_unmapped": 0,
    },
    # ── Phishing & Cyber Hygiene (4 items) ──────────────────────────────────
    {
        "item_id": "dl_phish_01",
        "skill_name": "phishing_cyber_hygiene",
        "prompt": (
            "You receive an urgent email: 'Your university account will be deleted in 2 hours! "
            "Verify now.' The display name says 'IT Helpdesk', but the address is "
            "'support@univeristy-verify-login.xyz'. What is the strongest red flag?"
        ),
        "options": [
            "The email uses a plain font",
            "Mismatched look-alike sender domain combined with artificial urgency",
            "The email arrived in the morning",
            "The subject line is short",
        ],
        "correct_answer": "Mismatched look-alike sender domain combined with artificial urgency",
        "hints": [
            "Look carefully at the spelling of the domain after the '@' sign.",
            "Notice how the message tries to rush you into acting without thinking.",
        ],
        "bottom_hint_text": (
            "Swapped letters in a look-alike domain ('univeristy-verify-login.xyz') and "
            "urgent threats are classic phishing indicators."
        ),
        "problem_difficulty": 0.80,
        "skill_difficulty": SKILL_DIFFICULTY_MAP["phishing_cyber_hygiene"],
        "is_skill_unmapped": 0,
    },
    {
        "item_id": "dl_phish_02",
        "skill_name": "phishing_cyber_hygiene",
        "prompt": (
            "An unexpected text message claims a parcel is delayed and includes a shortened "
            "link asking for a small redelivery fee and card details. What should you do?"
        ),
        "options": [
            "Click the link to see if the logo looks real",
            "Reply 'STOP' with your home address so they can deliver it",
            "Do not click; verify independently via the courier's official website or app",
            "Forward the link to friends to see if it opens on their phone",
        ],
        "correct_answer": (
            "Do not click; verify independently via the courier's official website or app"
        ),
        "hints": [
            "This is a text-based phishing attack (smishing).",
            "Never use the link or phone number inside a suspicious message to verify it.",
        ],
        "bottom_hint_text": (
            "Do not click or reply; navigate to the courier's official site or app yourself "
            "to check any real deliveries."
        ),
        "problem_difficulty": 0.76,
        "skill_difficulty": SKILL_DIFFICULTY_MAP["phishing_cyber_hygiene"],
        "is_skill_unmapped": 0,
    },
    {
        "item_id": "dl_phish_03",
        "skill_name": "phishing_cyber_hygiene",
        "prompt": (
            "Before clicking a link inside an email that claims to go to your bank, what "
            "quick check reveals the real destination URL?"
        ),
        "options": [
            "Hover over (or long-press on mobile) the link to preview the true URL domain",
            "Check whether the link text is colored blue",
            "Count the number of words in the email",
            "Click it once and immediately press the Back button",
        ],
        "correct_answer": (
            "Hover over (or long-press on mobile) the link to preview the true URL domain"
        ),
        "hints": [
            "Visible link text in HTML emails can say one thing while pointing somewhere else.",
            "Your browser or email client can preview a link destination without opening it.",
        ],
        "bottom_hint_text": (
            "Hovering (or long-pressing on mobile) previews the actual URL so you can verify "
            "the domain before clicking."
        ),
        "problem_difficulty": 0.70,
        "skill_difficulty": SKILL_DIFFICULTY_MAP["phishing_cyber_hygiene"],
        "is_skill_unmapped": 0,
    },
    {
        "item_id": "dl_phish_04",
        "skill_name": "phishing_cyber_hygiene",
        "prompt": (
            "You accidentally entered your password on a look-alike phishing page. Which "
            "sequence of actions best limits the damage?"
        ),
        "options": [
            "Delete the email from your inbox and do nothing else",
            "Change the password immediately (and anywhere reused), enable MFA, and alert IT",
            "Wait a week to see if any suspicious emails appear",
            "Turn off Wi-Fi for five minutes and log back in with the same password",
        ],
        "correct_answer": (
            "Change the password immediately (and anywhere reused), enable MFA, and alert IT"
        ),
        "hints": [
            "Once submitted, the attacker has that password—deleting the email won't revoke it.",
            "Invalidate the stolen credential and add a second authentication factor.",
        ],
        "bottom_hint_text": (
            "Immediately change the compromised password everywhere it was used, enable MFA, "
            "and notify your IT/security contact."
        ),
        "problem_difficulty": 0.48,
        "skill_difficulty": SKILL_DIFFICULTY_MAP["phishing_cyber_hygiene"],
        "is_skill_unmapped": 0,
    },
    # ── Search Strategies (3 items) ─────────────────────────────────────────
    {
        "item_id": "dl_srch_01",
        "skill_name": "search_strategies",
        "prompt": (
            "You want to find official government PDF guides about ransomware prevention "
            "and exclude results about movies. Which search query is most effective?"
        ),
        "options": [
            "please show me official government pdf guides about ransomware not movies",
            'site:.gov filetype:pdf "ransomware prevention" -movie',
            "ransomware pdf gov maybe",
            "https://gov.com/ransomware",
        ],
        "correct_answer": 'site:.gov filetype:pdf "ransomware prevention" -movie',
        "hints": [
            "Use site:, filetype:, quotes for exact phrases, and minus (-) to exclude terms.",
            "Combining domain, format, exact phrase, and exclusion removes irrelevant hits.",
        ],
        "bottom_hint_text": (
            'Use site:.gov for government domains, filetype:pdf for PDFs, quotes for exact '
            "phrases, and -movie to exclude movie results."
        ),
        "problem_difficulty": 0.56,
        "skill_difficulty": SKILL_DIFFICULTY_MAP["search_strategies"],
        "is_skill_unmapped": 0,
    },
    {
        "item_id": "dl_srch_02",
        "skill_name": "search_strategies",
        "prompt": (
            "Your search returns almost zero results because your keyword is too specific. "
            "How should you refine your query?"
        ),
        "options": [
            "Add more AND terms to narrow it further",
            "Broaden with synonyms using OR and remove overly narrow phrases",
            "Repeat the same word three times in capital letters",
            "Switch to searching full conversational paragraphs",
        ],
        "correct_answer": "Broaden with synonyms using OR and remove overly narrow phrases",
        "hints": [
            "Remember the difference between Boolean AND (narrows) and OR (broadens).",
            "Think about alternative words authors might use for the same concept.",
        ],
        "bottom_hint_text": (
            "When you get too few results, broaden your query with synonyms joined by OR."
        ),
        "problem_difficulty": 0.77,
        "skill_difficulty": SKILL_DIFFICULTY_MAP["search_strategies"],
        "is_skill_unmapped": 0,
    },
    {
        "item_id": "dl_srch_03",
        "skill_name": "search_strategies",
        "prompt": (
            "The top two links on a search results page have a small 'Sponsored' label next "
            "to them. What does that indicate?"
        ),
        "options": [
            "They are peer-reviewed academic articles",
            "They are paid advertisements, not necessarily the most authoritative sources",
            "They are verified government websites",
            "They are the only safe links on the page",
        ],
        "correct_answer": (
            "They are paid advertisements, not necessarily the most authoritative sources"
        ),
        "hints": [
            "Search engines place paid placements above organic results.",
            "Always look past ad labels and compare multiple organic results.",
        ],
        "bottom_hint_text": (
            "'Sponsored' marks paid ads; look below them and compare multiple independent sources."
        ),
        "problem_difficulty": 0.85,
        "skill_difficulty": SKILL_DIFFICULTY_MAP["search_strategies"],
        "is_skill_unmapped": 0,
    },
    # ── Software Navigation (3 items) ───────────────────────────────────────
    {
        "item_id": "dl_nav_01",
        "skill_name": "software_navigation",
        "prompt": (
            "You are sharing a draft report in cloud storage with classmates so they can "
            "leave feedback without directly altering your paragraphs. Which permission "
            "level should you assign?"
        ),
        "options": [
            "Editor",
            "Commenter",
            "Owner",
            "Public unrestricted edit",
        ],
        "correct_answer": "Commenter",
        "hints": [
            "Cloud tools typically offer Viewer, Commenter, and Editor roles.",
            "Which role allows feedback notes without overwriting the document text?",
        ],
        "bottom_hint_text": (
            "Commenter lets reviewers add feedback notes while blocking direct edits to text."
        ),
        "problem_difficulty": 0.81,
        "skill_difficulty": SKILL_DIFFICULTY_MAP["software_navigation"],
        "is_skill_unmapped": 0,
    },
    {
        "item_id": "dl_nav_02",
        "skill_name": "software_navigation",
        "prompt": (
            "You accidentally deleted a whole paragraph in a document editor. What is the "
            "fastest universal keyboard shortcut to reverse that action?"
        ),
        "options": [
            "Ctrl+Z (Cmd+Z on macOS) — Undo",
            "Ctrl+F (Cmd+F on macOS) — Find",
            "Ctrl+P (Cmd+P on macOS) — Print",
            "Alt+F4 — Close window",
        ],
        "correct_answer": "Ctrl+Z (Cmd+Z on macOS) — Undo",
        "hints": [
            "Almost every desktop and web editor shares the same shortcut for Undo.",
            "It pairs with Ctrl+Y / Cmd+Shift+Z for Redo.",
        ],
        "bottom_hint_text": (
            "Ctrl+Z (or Cmd+Z on macOS) immediately undoes the last edit."
        ),
        "problem_difficulty": 0.88,
        "skill_difficulty": SKILL_DIFFICULTY_MAP["software_navigation"],
        "is_skill_unmapped": 0,
    },
    {
        "item_id": "dl_nav_03",
        "skill_name": "software_navigation",
        "prompt": (
            "You need to change a formatting setting in an unfamiliar app, and you don't know "
            "which menu tab hides it. What is the most transferable strategy?"
        ),
        "options": [
            "Reinstall the application",
            "Use the app's built-in Help / command search bar or hover icons for tooltips",
            "Click random buttons until something changes",
            "Delete your document and start in another program",
        ],
        "correct_answer": (
            "Use the app's built-in Help / command search bar or hover icons for tooltips"
        ),
        "hints": [
            "Modern apps include an in-app search box for commands and settings.",
            "Hovering over toolbar icons also reveals their names and tooltips safely.",
        ],
        "bottom_hint_text": (
            "In-app Help/command search and icon tooltips let you locate commands safely "
            "across almost any software."
        ),
        "problem_difficulty": 0.79,
        "skill_difficulty": SKILL_DIFFICULTY_MAP["software_navigation"],
        "is_skill_unmapped": 0,
    },
]


def seed_assessment_items(storage: Any = None) -> int:
    """Idempotently seed all 18 graded Digital Literacy items into the active storage."""
    repo = storage if storage is not None else get_storage()
    for item in ASSESSMENT_ITEMS:
        repo.upsert_assessment_item(item)
    return len(ASSESSMENT_ITEMS)


def ensure_assessment_items_seeded(storage: Any = None) -> list[dict[str, Any]]:
    """Return all assessment items, auto-seeding the 18-item bank if empty."""
    repo = storage if storage is not None else get_storage()
    existing = repo.list_assessment_items()
    if not existing:
        seed_assessment_items(repo)
        existing = repo.list_assessment_items()
    return existing


def sanitize_item_for_student(item: dict[str, Any]) -> dict[str, Any]:
    """Strip ``correct_answer`` and hidden hint bodies before sending an item to the student UI."""
    hints = list(item.get("hints") or [])
    has_bottom = bool(item.get("bottom_hint_text"))
    return {
        "item_id": item["item_id"],
        "skill_name": item["skill_name"],
        "prompt": item["prompt"],
        "options": list(item.get("options") or []),
        "hint_total": len(hints) + (1 if has_bottom else 0),
        "problem_difficulty": float(item.get("problem_difficulty", 0.70)),
        "skill_difficulty": float(item.get("skill_difficulty", 0.69)),
        "is_skill_unmapped": int(item.get("is_skill_unmapped", 0)),
    }


def get_hint_for_item(item: dict[str, Any], hint_index: int) -> dict[str, Any]:
    """Return the progressive hint (or bottom-out hint) at 0-based ``hint_index``."""
    hints = list(item.get("hints") or [])
    bottom_text = item.get("bottom_hint_text")
    all_hints: list[tuple[str, bool]] = [(h, False) for h in hints]
    if bottom_text:
        all_hints.append((str(bottom_text), True))
    if not all_hints or hint_index < 0 or hint_index >= len(all_hints):
        raise IndexError(f"Hint index {hint_index} out of range (total={len(all_hints)}).")
    text, is_bottom = all_hints[hint_index]
    hints_used = hint_index + 1
    hint_total = len(all_hints)
    return {
        "item_id": item["item_id"],
        "hint_index": hint_index,
        "hint_text": text,
        "is_bottom_hint": is_bottom,
        "hints_used": hints_used,
        "hint_total": hint_total,
        "hint_ratio": round(hints_used / max(hint_total, 1), 4),
    }


def compute_11_engineered_features(
    *,
    opportunity: int,
    attempt_count: int,
    hints_used: int,
    hint_total: int,
    bottom_hint_used: bool,
    user_prev_acc: float,
    problem_difficulty: float,
    skill_difficulty: float,
    is_skill_unmapped: int,
    ms_first_response: float,
) -> dict[str, Any]:
    """Automatically compute the exact 11 canonical ``ENGINEERED_FEATURES`` (leak-free)."""
    safe_opp = max(1, int(opportunity))
    safe_attempts = float(min(max(int(attempt_count), 1), 50))
    safe_hints = float(min(max(int(hints_used), 0), 50))
    safe_hint_total = max(int(hint_total), 1)
    safe_ratio = round(min(max(safe_hints / safe_hint_total, 0.0), 1.0), 4)
    safe_ms = max(0.0, float(ms_first_response))
    log_rt = round(min(math.log1p(safe_ms), 25.0), 6)

    features: dict[str, Any] = {
        "log_opportunity": round(min(math.log1p(safe_opp), 20.0), 6),
        "attempt_count": safe_attempts,
        "hint_count": safe_hints,
        "bottom_hint": 1 if bottom_hint_used else 0,
        "hint_ratio": safe_ratio,
        "user_prev_acc": round(min(max(float(user_prev_acc), 0.0), 1.0), 4),
        "problem_difficulty": round(min(max(float(problem_difficulty), 0.0), 1.0), 4),
        "skill_difficulty": round(min(max(float(skill_difficulty), 0.0), 1.0), 4),
        "is_skill_unmapped": 1 if int(is_skill_unmapped) else 0,
        "log_response_time": log_rt,
        "is_rapid_guess": 1 if safe_ms < 3000.0 else 0,
    }
    # Enforce exact canonical ordering
    return {col: features[col] for col in ENGINEERED_FEATURES}


def build_socratic_intervention_prompt(
    item: dict[str, Any],
    selected_answer: str,
    is_correct: bool,
    needs_intervention: bool,
) -> Optional[str]:
    """Build a safe Socratic tutor prompt referencing the student's item without leaking answers."""
    if is_correct and not needs_intervention:
        return None
    skill_label = str(item.get("skill_name", "digital_literacy")).replace("_", " ")
    prompt_excerpt = str(item.get("prompt", "")).strip()
    if is_correct:
        return (
            f"I just answered a {skill_label} practice question correctly, but the model noticed "
            f"I needed several hints or attempts. Question: '{prompt_excerpt}'. "
            "Can you help me check my reasoning Socratically so I truly master the concept?"
        )
    return (
        f"I'm working on a {skill_label} practice question and got it wrong. "
        f"Question: '{prompt_excerpt}'. I chose '{selected_answer}'. "
        "Please don't give me the direct answer — guide me Socratically to see what I missed."
    )
