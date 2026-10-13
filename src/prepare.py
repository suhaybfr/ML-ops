import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)["prepare"]


def prepare_data():
    params = load_params()
    raw_data_path = os.path.join("data", "raw", "skill_builder_data.csv")
    output_dir = os.path.join("data", "processed")
    os.makedirs(output_dir, exist_ok=True)

    print(f"[Prepare] Loading raw data from: {raw_data_path}")
    # Load dataset
    df = pd.read_csv(raw_data_path, encoding="latin1", low_memory=False)
    print(f"[Prepare] Raw dataset shape: {df.shape}")

    # Relevant columns for student correctness prediction
    feature_cols = [
        "attempt_count",
        "ms_first_response",
        "hint_count",
        "hint_total",
        "overlap_time",
        "opportunity",
    ]
    target_col = "correct"

    # Filter to valid target rows (binary 0 or 1)
    df = df[df[target_col].isin([0, 1])].copy()
    df[target_col] = df[target_col].astype(int)

    # Impute missing values with median
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    # Optional sampling for rapid baseline prototyping
    sample_size = params.get("sample_size")
    if sample_size and sample_size < len(df):
        print(f"[Prepare] Subsetting to {sample_size} rows for fast reproducible training")
        df = df.sample(n=sample_size, random_state=params["random_state"])

    # Keep only selected features and target
    selected_cols = feature_cols + [target_col]
    df = df[selected_cols]

    # Stratified Train-Test split
    train_df, test_df = train_test_split(
        df,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=df[target_col],
    )

    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[Prepare] Saved train set ({len(train_df)} rows) to: {train_path}")
    print(f"[Prepare] Saved test set ({len(test_df)} rows) to: {test_path}")


if __name__ == "__main__":
    prepare_data()
