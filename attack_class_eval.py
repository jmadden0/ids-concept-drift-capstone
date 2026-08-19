import pandas as pd
import numpy as np
from pathlib import Path

print("Loading test data...")
test_path = Path("data/unsw/treated_test.csv")
test_df = pd.read_csv(test_path)

detailed_targets = test_df["classes"].to_numpy(dtype=np.int64)
merge_mapping = {
    0: 1,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7, 
    8: 1,
    9: 1,
}
class_targets = np.array([merge_mapping[x] for x in detailed_targets], dtype=np.int64)
test_features = test_df.drop(columns=["classes"]).to_numpy(dtype=np.float64)

print(f"\n\nTest data shape: {test_features.shape}")
print(f"Number of classes: {len(np.unique(class_targets))}")
print(f"Shape of class targets: {class_targets.shape}")
