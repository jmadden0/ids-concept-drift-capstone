import pandas as pd
import numpy as np
from pathlib import Path
import torch
from src.model import ReCDA, MLP

CLASS_NAMES = {
    1: "Others",
    2: "DoS",
    3: "Exploits",
    4: "Fuzzers",
    5: "Generic",
    6: "Normal",
    7: "Reconnaissance",
}

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

print(f"Test data shape: {test_features.shape}")
print(f"Number of classes: {len(np.unique(class_targets))}")
print(f"Shape of class targets: {class_targets.shape}\n")

#Set device for Pytorch
device = torch.device("cpu")

print("Recreating and loading models...")
#Recreate the encoder and classifier models and load the saved weights
encoder = ReCDA(input_size=test_features.shape[1], embedding_size=16, encoder_depth=4, head_depth=2).to(device)
encoder.load_state_dict(
    torch.load("model/tuned_encoder.pth", map_location=device)
)
encoder.eval()

classifier = MLP(input_size=16, hidden_size=2, num_layers=1).to(device)
classifier.load_state_dict(
    torch.load("model/classifier.pth", map_location=device)
)
classifier.eval()
print("Models loaded successfully.\n")

#Convert test data to torch tensors
test_tensor = torch.tensor(
    test_features,
    dtype=torch.float32,
).to(device)

print("Making predictions on test data...")
#Make predictions without tracking gradients (learning)
with torch.no_grad():
    test_embeddings = encoder.get_embeddings(test_tensor)
    outputs = classifier(test_embeddings)
    predictions = outputs.max(1).indices.cpu().numpy()
    
print("Predictions Completed.")
print(f"Predictions shape: {predictions.shape}")
print(f"Number of classes in predictions: {len(np.unique(predictions))}\n")

#Evaluate the predictions by attack class
results = []

print("Evaluating predictions by attack class...")
for class_id, class_name in CLASS_NAMES.items():
    class_mask = class_targets == class_id
    class_predictions = predictions[class_mask]
    class_count = len(class_predictions)
    
    if class_name == "Normal":
        predicted_attack = np.sum(class_predictions == 1)
        false_positive_rate = predicted_attack / class_count if class_count > 0 else np.nan
        
        results.append({
            "class": class_name,
            "class_count": class_count,
            "detected": np.nan,
            "missed": np.nan,
            "detection_rate": np.nan,
            "false_negative_rate": np.nan,
            "false_positive_rate": false_positive_rate
        })
    
    else:
        detected = np.sum(class_predictions == 1)
        missed = np.sum(class_predictions == 0)
        
        detection_rate = detected / class_count if class_count > 0 else np.nan
        false_negative_rate = missed / class_count if class_count > 0 else np.nan
        
        results.append({
            "class": class_name,
            "class_count": class_count,
            "detected": detected,
            "missed": missed,
            "detection_rate": detection_rate,
            "false_negative_rate": false_negative_rate,
            "false_positive_rate": np.nan
        })
    print(f"Class: {class_name}, Count: {class_count}, Detected: {detected if class_name != 'Normal' else 'N/A'}, Missed: {missed if class_name != 'Normal' else 'N/A'}, Detection Rate: {detection_rate if class_name != 'Normal' else 'N/A'}, False Negative Rate: {false_negative_rate if class_name != 'Normal' else 'N/A'}, False Positive Rate: {false_positive_rate if class_name == 'Normal' else 'N/A'}")
        
print("\nEvaluation completed.\n")