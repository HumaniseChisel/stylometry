import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import os
import json
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configuration
# MAX_SEQ_LEN = 50   # How many keystrokes per sample?
# MIN_SEQ_LEN = 10   # Drop samples shorter than this
DIR_PATH = "./data"    # Point this to your root directory containing user folders


class KeystrokeDataProcessor:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []  # List of numpy arrays (Seq_Len, Features)
        self.labels = []   # List of integers
        self.class_map = {} # {'Hussein': 0, 'michael': 1}
        
    def load_data(self):
        # 1. Identify all users (subdirectories)
        user_dirs = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        self.class_map = {name: i for i, name in enumerate(sorted(user_dirs))}
        
        print(f"Found {len(self.class_map)} users: {self.class_map}")

        # 2. Iterate through files
        for user_name in user_dirs:
            label = self.class_map[user_name]
            user_path = os.path.join(self.root_dir, user_name)
            json_files = glob.glob(os.path.join(user_path, "*.json"))
            
            for j_file in json_files:
                with open(j_file, 'r') as f:
                    try:
                        content = json.load(f)
                        logs = content.get('keystrokeLogs', [])
                        
                        # if len(logs) < MIN_SEQ_LEN:
                        #     continue
                            
                        # Process one full session (file) into features
                        features = self._extract_features(logs)

                        # if features.shape != (41, 3):
                        #         print(f"WARNING: Skipping malformed chunk in {j_file} with shape {features.shape}")
                        #         continue

                        self.samples.append(features[:41])
                        self.labels.append(label)
                        
                        # # CHUNKING: If a file is long (e.g., 200 keys), split it into multiple samples of MAX_SEQ_LEN
                        # # This creates more training data.
                        # num_chunks = len(features) // MAX_SEQ_LEN
                        # for i in range(num_chunks):
                        #     start = i * MAX_SEQ_LEN
                        #     end = start + MAX_SEQ_LEN
                        #     self.samples.append(features[start:end])
                        #     self.labels.append(label)
                            
                    except Exception as e:
                        print(f"Error parsing {j_file}: {e}")

        print(f"Total samples extracted: {len(self.samples)}")

    def _extract_features(self, logs):
        """
        Converts raw JSON logs into a numpy array of shape (N, 3).
        Features: [Hold Time, Flight Time, KeyCode]
        """
        processed_seq = []
        
        for i in range(len(logs)):
            curr = logs[i]
            
            # 1. Hold Time (Normalize: usually in ms)
            hold = curr.get('holdTime', 0)
            
            flight = curr.get('flightTime', 0)

            # # 2. Flight Time (Calculate manually for accuracy)
            # # Flight = Current_Down - Previous_Up
            # # Previous_Up = Previous_Down + Previous_Hold
            # if i == 0:
            #     flight = 0.0 # First key has no flight time
            # else:
            #     prev = logs[i-1]
            #     prev_up = prev['keyDownTime'] + prev['holdTime']
            #     curr_down = curr['keyDownTime']
            #     flight = curr_down - prev_up
            #     # Note: Flight time can be negative (rollover typing), which is a GOOD feature.
            
            # 3. Key Code (ASCII value)
            # We convert the character to its ASCII int value. 
            key_char = curr.get('key', '')
            if len(key_char) == 1:
                key_code = ord(key_char)
            else:
                # Handle special keys (Shift, Enter, etc.) if they appear as strings
                key_code = 0 
            
            processed_seq.append([hold, flight, key_code])
            
        return np.array(processed_seq, dtype=np.float32)

    def get_splits(self):
        # Convert lists to numpy arrays
        X = np.array(self.samples)
        y = np.array(self.labels)
        
        # Stratified Split: Ensures every user is represented in Train/Val/Test
        # Train: 70%, Temp: 30%
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
        
        # Split Temp into Val (15%) and Test (15%)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
        
        # Normalization (StandardScaler)
        # IMPORTANT: Fit scalar ONLY on training data, then transform Val/Test
        # We reshape to (N*Seq, Features) to fit, then reshape back
        scaler = StandardScaler()
        
        N, L, F = X_train.shape
        X_train_reshaped = X_train.reshape(-1, F)
        scaler.fit(X_train_reshaped) # Compute Mean/Std
        
        # Apply Transform
        X_train = scaler.transform(X_train_reshaped).reshape(N, L, F)
        X_val   = scaler.transform(X_val.reshape(-1, F)).reshape(X_val.shape[0], L, F)
        X_test  = scaler.transform(X_test.reshape(-1, F)).reshape(X_test.shape[0], L, F)
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test), len(self.class_map)

# ==========================================
# 2. PyTorch Dataset Wrapper
# ==========================================
class TensorDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = torch.FloatTensor(x_data)
        self.y = torch.LongTensor(y_data)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# ==========================================
# 2. The Bi-LSTM Architecture
# ==========================================
class KeystrokeIDModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(KeystrokeIDModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer
        # batch_first=True ensures input is (Batch, Seq, Feature) rather than (Seq, Batch, Feature)
        # bidirectional=True lets the model see future context (useful for typing rhythm)
        self.lstm = nn.LSTM(input_size, 
                            hidden_size, 
                            num_layers, 
                            batch_first=True, 
                            bidirectional=True)
        
        # Fully Connected Layer for Classification
        # We multiply hidden_size * 2 because it is Bi-Directional
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x shape: (Batch_Size, Seq_Length, Input_Size)
        
        # Initialize hidden state and cell state (optional, PyTorch defaults to zeros)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        # out shape: (Batch_Size, Seq_Length, Hidden_Size * 2)
        out, _ = self.lstm(x, (h0, c0))
        
        # We generally only care about the output of the *last* time step for classification
        # out[:, -1, :] selects the last vector in the sequence
        out = out[:, -1, :]
        
        out = self.dropout(out)
        out = self.fc(out)
        return out

# ==========================================
# 3. The Setup & Training Loop
# ==========================================

# Hyperparameters
INPUT_SIZE = 3      # e.g., [Hold Time, Flight Time, Key Code]
HIDDEN_SIZE = 64    # Number of features in the hidden state
NUM_LAYERS = 2      # Number of stacked LSTM layers
NUM_CLASSES = 5     # Number of users to identify
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 5

# Assuming the Bi-LSTM Model class (KeystrokeIDModel) is already defined as in the previous response

# 1. Prepare Data
processor = KeystrokeDataProcessor(root_dir=DIR_PATH) # UPDATE THIS PATH
processor.load_data()

(X_train, y_train), (X_val, y_val), (X_test, y_test), num_classes = processor.get_splits()

print(f"Training Shape: {X_train.shape}")
print(f"Validation Shape: {X_val.shape}")

# 2. Create Loaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 3. Model Setup
INPUT_SIZE = 3 # Hold, Flight, KeyCode
HIDDEN_SIZE = 64
NUM_LAYERS = 2
model = KeystrokeIDModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. Train with Validation
EPOCHS = 20

for epoch in range(EPOCHS):
    # --- Training Phase ---
    model.train()
    train_loss = 0
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    # --- Validation Phase ---
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {100 * correct / total:.2f}%")


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, test_loader, device, class_map):
    model.eval()  # Set model to evaluation mode
    
    all_preds = []
    all_labels = []
    
    print("Running evaluation on Test Set...")
    
    with torch.no_grad(): # Disable gradient calculation for speed/memory
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

# --- RUN IT ---
# Ensure 'device', 'model', and 'test_loader' are available from your training script
evaluate_model(model, test_loader, device, processor.class_map)