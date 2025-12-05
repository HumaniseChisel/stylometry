import torch
from torch.utils.data import DataLoader

from model import (
    KeystrokeDataProcessor,
    KeystrokeIDModel,
    TensorDataset,
    evaluate_model,
    train_model,
)

# Configuration
DIR_PATH = "./model/src/model/data"  # Point this to your root directory containing user folders

# Hyperparameters
INPUT_SIZE = 3      # [Hold Time, Flight Time, Key Code]
HIDDEN_SIZE = 64    # Number of features in the hidden state
NUM_LAYERS = 2      # Number of stacked LSTM layers
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20


def main():
    # 1. Prepare Data
    print("Loading and processing data...")
    processor = KeystrokeDataProcessor()
    processor.load_from_directory(root_dir=DIR_PATH)

    (X_train, y_train), (X_val, y_val), (X_test, y_test), num_classes, scaler = processor.get_splits()

    print(f"Training Shape: {X_train.shape}")
    print(f"Validation Shape: {X_val.shape}")
    print(f"Test Shape: {X_test.shape}")
    print(f"Number of classes: {num_classes}")

    # 2. Create Data Loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Model Setup
    print("\nInitializing model...")
    model = KeystrokeIDModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, num_classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)

    # 4. Train the Model
    print("\nStarting training...")
    model = train_model(model, train_loader, val_loader, device, epochs=EPOCHS, learning_rate=LEARNING_RATE)

    # 5. Evaluate on Test Set
    print("\n" + "="*50)
    print("FINAL EVALUATION ON TEST SET")
    print("="*50)
    evaluate_model(model, test_loader, device, processor.class_map)

    # 6. Save the Model (Optional)
    print("\nSaving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_map': processor.class_map,
        'scaler': scaler,
        'input_size': INPUT_SIZE,
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS,
        'num_classes': num_classes
    }, 'keystroke_model.pth')
    print("Model saved to 'keystroke_model.pth'")


if __name__ == "__main__":
    main()
