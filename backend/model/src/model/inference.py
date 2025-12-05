import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from model import KeystrokeIDModel


def evaluate_model(model, test_loader, device, class_map):
    """
    Evaluate the model on the test set and display metrics.
    """
    model.eval()  # Set model to evaluation mode

    all_preds = []
    all_labels = []

    print("Running evaluation on Test Set...")

    with torch.no_grad():  # Disable gradient calculation for speed/memory
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(features)

            # Get predictions (max logit)
            _, preds = torch.max(outputs, 1)

            # Move to CPU and convert to numpy for sklearn
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- Metrics ---
    # Invert class_map to get names from IDs (e.g., {0: 'Hussein', 1: 'Michael'})
    idx_to_class = {v: k for k, v in class_map.items()}
    target_names = [idx_to_class[i] for i in range(len(class_map))]

    # 1. Accuracy
    acc = accuracy_score(all_labels, all_preds)
    print(f"\nTest Set Accuracy: {acc*100:.2f}%")

    # 2. Detailed Report (Precision, Recall, F1)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names))

    # 3. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plotting the Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Keystroke Identification Confusion Matrix')
    plt.show()

    return acc, all_preds, all_labels


def load_model(model_source, device=None):
    """
    Load a trained model from a checkpoint file.

    Args:
        model_source: Path or file to the model checkpoint file
        device: torch device (defaults to cuda if available, else cpu)

    Returns:
        Tuple containing:
            - model: Loaded KeystrokeIDModel
            - scaler: StandardScaler for feature normalization
            - class_map: Dictionary mapping user names to class IDs
            - device: torch device used
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(model_source, map_location=device, weights_only=False)

    # Extract model configuration
    input_size = checkpoint['input_size']
    hidden_size = checkpoint['hidden_size']
    num_layers = checkpoint['num_layers']
    num_classes = checkpoint['num_classes']

    # Initialize and load model
    model = KeystrokeIDModel(input_size, hidden_size, num_layers, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Load scaler and class map
    scaler = checkpoint['scaler']
    class_map = checkpoint['class_map']

    return model, scaler, class_map, device


def predict_single(model, keystroke_sequence, scaler, device, class_map):
    """
    Predict the user identity for a single keystroke sequence.

    Args:
        model: Trained KeystrokeIDModel
        keystroke_sequence: numpy array of shape (Seq_Len, 3) with raw features
        scaler: StandardScaler fitted on training data
        device: torch device
        class_map: Dictionary mapping user names to class IDs

    Returns:
        predicted_user: String name of predicted user
        confidence: Float probability of prediction
    """
    model.eval()

    # Normalize the sequence
    seq_normalized = scaler.transform(keystroke_sequence.reshape(-1, 3))
    seq_normalized = seq_normalized.reshape(1, -1, 3)  # Add batch dimension

    # Convert to tensor
    seq_tensor = torch.FloatTensor(seq_normalized).to(device)

    with torch.no_grad():
        output = model(seq_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    # Get user name from class ID
    idx_to_class = {v: k for k, v in class_map.items()}
    predicted_user = idx_to_class[predicted_class.item()]

    return predicted_user, confidence.item()
