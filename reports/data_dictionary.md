# Data Dictionary: Engineered Features

## Feature Engineering Hypotheses
1. **Skill Association (H1):** Student performance depends on the skill domain.
2. **Practice Volume & Learning Curve (H2):** Practice increases performance with diminishing returns (`log_opportunity`).
3. **Historical Performance Trajectory (H3):** Prior cumulative accuracy (`user_prev_acc`) predicts future success.
4. **Help-Seeking & Reliance on Hints (H4):** Higher hint usage (`hint_ratio`, `bottom_hint`) indicates lower immediate mastery.
5. **Unassisted Mastery Signal (H5):** First-try correctness without hints (`first_try_correct`) signals strong concept proficiency.
6. **Problem Inherent Difficulty (H6):** Historical problem pass rate (`problem_difficulty`) models item hardness.
7. **Domain Inherent Difficulty (H7):** Skill-level pass rate (`skill_difficulty`) captures concept domain difficulty.
8. **Unmapped Skill Penalty (H8):** Unmapped skill entries (`is_skill_unmapped`) exhibit distinct completion behavior.
9. **Response Latency & Engagement (H9):** Fast responses under 3s (`is_rapid_guess`) signal guessing, while log latency (`log_response_time`) captures thinking effort.

---

## Field Specifications
| Column Name | Data Type | Feature Category | Description |
| :--- | :--- | :--- | :--- |
| `user_id` | Integer | Entity Identifier | Unique student ID |
| `problem_id` | Integer | Entity Identifier | Unique problem ID |
| `skill_name` | String | Concept Identifier | Name of the skill |
| `opportunity` | Integer | Historical Practice | Cumulative practice count |
| `log_opportunity` | Float | Learning Curve | ln(1 + opportunity) |
| `attempt_count` | Integer | Interaction | Total attempts on current problem |
| `hint_count` | Integer | Help-Seeking | Number of hints requested |
| `bottom_hint` | Binary | Help-Seeking | 1 if bottom hint viewed, else 0 |
| `hint_ratio` | Float | Help-Seeking | hint_count / hint_total |
| `first_try_correct` | Binary | Mastery Signal | 1 if attempt_count=1, correct=1, hint_count=0 |
| `user_prev_acc` | Float | Student History | Cumulative past accuracy (expanding mean shifted by 1) |
| `problem_difficulty`| Float | Item Difficulty | Target-encoded problem pass rate |
| `skill_difficulty` | Float | Concept Difficulty| Target-encoded skill pass rate |
| `is_skill_unmapped` | Binary | Quality Indicator | 1 if skill_name == "Unknown", else 0 |
| `log_response_time` | Float | Speed Signal | ln(1 + max(0, ms_first_response)) |
| `is_rapid_guess` | Binary | Engagement | 1 if ms_first_response < 3000 ms, else 0 |
| `correct` | Binary | Target | 1 = Correct response, 0 = Incorrect |

