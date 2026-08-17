from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#Mappings to align with utils.py mappings
CLASS_MAPPING = {
    "Analysis": 0,
    "Backdoor": 1,
    "DoS": 2,
    "Exploits": 3,
    "Fuzzers": 4,
    "Generic": 5,
    "Normal": 6,
    "Reconnaissance": 7,
    "Shellcode": 8,
    "Worms": 9,
}

#Needed to encode categorical features as per the paper
CATEGORICAL_COLUMNS = ["proto", "service", "state"]
#Dropping ID and redundant label column
DROP_COLUMNS = ["id", "label"]

def prepare_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    
    train_x = train_df.drop(
        columns=["attack_cat", *DROP_COLUMNS],
        errors="ignore",
    ).copy()

    test_x = test_df.drop(
        columns=["attack_cat", *DROP_COLUMNS],
        errors="ignore",
    ).copy()

    test_x = test_x[train_x.columns]

    numerical_columns = [
        column
        for column in train_x.columns
        if column not in CATEGORICAL_COLUMNS
    ]

    # Coerce numerical columns and remove infinities.
    for column in numerical_columns:
        train_x[column] = pd.to_numeric(
            train_x[column],
            errors="coerce",
        )
        test_x[column] = pd.to_numeric(
            test_x[column],
            errors="coerce",
        )

    train_x[numerical_columns] = train_x[numerical_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    test_x[numerical_columns] = test_x[numerical_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numerical_pipeline,
                numerical_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_COLUMNS,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    train_processed = preprocessor.fit_transform(train_x)
    test_processed = preprocessor.transform(test_x)

    feature_names = list(preprocessor.get_feature_names_out())

    return train_processed, test_processed, feature_names


def encode_targets(df: pd.DataFrame) -> np.ndarray:
    """Convert attack category names into ReCDA class identifiers."""

    categories = df["attack_cat"].astype(str).str.strip()

    unknown_categories = sorted(
        set(categories.unique()) - set(CLASS_MAPPING)
    )

    if unknown_categories:
        raise ValueError(
            "Unexpected attack categories: "
            f"{unknown_categories}"
        )

    return categories.map(CLASS_MAPPING).to_numpy(dtype=np.int64)


def main() -> None:
    train_path = Path(r"CSV Files/Training and Testing Sets/UNSW_NB15_training-set.csv")
    test_path = Path(r"CSV Files/Training and Testing Sets/UNSW_NB15_testing-set.csv")
    output_path = Path(r"data/unsw")

    output_path.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_target = encode_targets(train_df)
    test_target = encode_targets(test_df)

    train_x, test_x, feature_names = prepare_features(
        train_df,
        test_df,
    )

    treated_train = pd.DataFrame(
        train_x,
        columns=feature_names,
    )
    treated_train["classes"] = train_target

    treated_test = pd.DataFrame(
        test_x,
        columns=feature_names,
    )
    treated_test["classes"] = test_target

    train_output = output_path / "treated_train.csv"
    test_output = output_path / "treated_test.csv"

    treated_train.to_csv(train_output, index=False)
    treated_test.to_csv(test_output, index=False)

    print(f"Training shape: {treated_train.shape}")
    print(f"Testing shape:  {treated_test.shape}")
    print(f"Feature count:  {len(feature_names)}")

    print("\nTraining classes:")
    print(treated_train["classes"].value_counts().sort_index())

    print("\nTesting classes:")
    print(treated_test["classes"].value_counts().sort_index())

    print(f"\nSaved: {train_output}")
    print(f"Saved: {test_output}")


if __name__ == "__main__":
    main()