import torch
from torch.utils.data import Dataset
import os
import json
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Union, Tuple, Optional


class KeystrokeDataProcessor:
    def __init__(self):
        self.samples = []  # List of numpy arrays (Seq_Len, Features)
        self.labels = []   # List of integers
        self.class_map = {}  # {'Hussein': 0, 'michael': 1}

    def load_from_directory(self, root_dir: str):
        """
        Load data from a directory structure where each subdirectory represents a user.

        Args:
            root_dir: Path to root directory containing user subdirectories
        """
        # 1. Identify all users (subdirectories)
        user_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.class_map = {name: i for i, name in enumerate(sorted(user_dirs))}

        print(f"Found {len(self.class_map)} users: {self.class_map}")

        # 2. Iterate through files
        for user_name in user_dirs:
            label = self.class_map[user_name]
            user_path = os.path.join(root_dir, user_name)
            json_files = glob.glob(os.path.join(user_path, "*.json"))

            for j_file in json_files:
                self._load_single_file(j_file, label)

        print(f"Total samples extracted: {len(self.samples)}")

    # def load_from_file(self, file_path: str, label: Union[int, str], class_map: Optional[Dict[str, int]] = None):
    #     """
    #     Load data from a single JSON file.

    #     Args:
    #         file_path: Path to JSON file
    #         label: Integer label or string class name
    #         class_map: Optional existing class mapping. If label is string and class_map not provided,
    #                   will create/update internal class_map
    #     """
    #     if isinstance(label, str):
    #         if class_map is not None:
    #             self.class_map = class_map
    #             label = class_map[label]
    #         else:
    #             if label not in self.class_map:
    #                 self.class_map[label] = len(self.class_map)
    #             label = self.class_map[label]

    #     self._load_single_file(file_path, label)

    def load_from_raw_data(self, keystroke_logs: List[Dict], label: Union[int, str],
                          class_map: Optional[Dict[str, int]] = None):
        """
        Load data directly from raw keystroke log data.

        Args:
            keystroke_logs: List of keystroke log dictionaries
            label: Integer label or string class name
            class_map: Optional existing class mapping
        """
        if isinstance(label, str):
            if class_map is not None:
                self.class_map = class_map
                label = class_map[label]
            else:
                if label not in self.class_map:
                    self.class_map[label] = len(self.class_map)
                label = self.class_map[label]

        features = self._extract_features(keystroke_logs)
        self.samples.append(features[:41]) # TODO: make it not 41
        self.labels.append(label)

    def _load_single_file(self, file_path: str, label: int):
        """
        Internal method to load a single JSON file.

        Args:
            file_path: Path to JSON file
            label: Integer label for this sample
        """
        with open(file_path, 'r') as f:
            try:
                content = json.load(f)
                logs = content.get('keystrokeLogs', [])

                # Process one full session (file) into features
                features = self._extract_features(logs)

                self.samples.append(features[:41])
                self.labels.append(label)

            except Exception as e:
                print(f"Error parsing {file_path}: {e}")

    def _extract_features(self, logs: List[Dict]) -> np.ndarray:
        """
        Converts raw JSON logs into a numpy array of shape (N, 3).
        Features: [Hold Time, Flight Time, KeyCode]

        Args:
            logs: List of keystroke log dictionaries

        Returns:
            Numpy array of shape (N, 3)
        """
        processed_seq = []

        for i in range(len(logs)):
            curr = logs[i]

            # 1. Hold Time (Normalize: usually in ms)
            hold = curr.get('holdTime', 0)

            flight = curr.get('flightTime', 0)

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

    def get_splits(self, test_size: float = 0.3, val_split: float = 0.5,
                   random_state: int = 42) -> Tuple[Tuple[np.ndarray, np.ndarray],
                                                      Tuple[np.ndarray, np.ndarray],
                                                      Tuple[np.ndarray, np.ndarray],
                                                      int, StandardScaler]:
        """
        Split data into train/val/test sets with normalization.

        Args:
            test_size: Proportion of data for temp split (val + test)
            val_split: Proportion of temp data for validation (0.5 means equal val/test)
            random_state: Random seed for reproducibility

        Returns:
            Tuple containing:
                - (X_train, y_train)
                - (X_val, y_val)
                - (X_test, y_test)
                - num_classes
                - fitted scaler
        """
        # Convert lists to numpy arrays
        X = np.array(self.samples)
        y = np.array(self.labels)

        # Stratified Split: Ensures every user is represented in Train/Val/Test
        # Train: 70%, Temp: 30%
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

        # Split Temp into Val (15%) and Test (15%)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=val_split, stratify=y_temp, random_state=random_state
        )

        # Normalization (StandardScaler)
        # IMPORTANT: Fit scalar ONLY on training data, then transform Val/Test
        # We reshape to (N*Seq, Features) to fit, then reshape back
        scaler = StandardScaler()

        N, L, F = X_train.shape
        X_train_reshaped = X_train.reshape(-1, F)
        scaler.fit(X_train_reshaped)  # Compute Mean/Std

        # Apply Transform
        X_train = scaler.transform(X_train_reshaped).reshape(N, L, F)
        X_val = scaler.transform(X_val.reshape(-1, F)).reshape(X_val.shape[0], L, F)
        X_test = scaler.transform(X_test.reshape(-1, F)).reshape(X_test.shape[0], L, F)

        return (X_train, y_train), (X_val, y_val), (X_test, y_test), len(self.class_map), scaler


class TensorDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = torch.FloatTensor(x_data)
        self.y = torch.LongTensor(y_data)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
