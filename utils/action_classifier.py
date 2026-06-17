import torch
import torch.nn as nn
import numpy as np
import os

class ActionLSTM(nn.Module):
    """
    LSTM Action Classifier.
    Input shape: (batch_size, sequence_length, input_dim) where input_dim is 26 (13 keypoints * 2).
    Output shape: (batch_size, num_classes) where classes are:
      0: batting
      1: bowling
      2: fielding
      3: invalid
    """
    def __init__(self, input_dim=26, hidden_dim=64, num_layers=2, num_classes=4):
        super(ActionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Take the output of the last time step
        last_step_out = lstm_out[:, -1, :]
        logits = self.fc(last_step_out)
        return logits

# Indices of the 13 required landmarks
KEYPOINT_IDS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

def format_pose_sequence(all_landmarks, max_seq_len=20) -> np.ndarray:
    """
    Converts list of landmarks dicts to a standardized numpy array of shape (1, max_seq_len, 26).
    Fills missing landmarks or missing frames with zero padding.
    """
    seq = []
    for lm in all_landmarks:
        if lm is None:
            # Pad a missing frame with zeros
            seq.append([0.0] * (len(KEYPOINT_IDS) * 2))
            continue
            
        frame_coords = []
        for kid in KEYPOINT_IDS:
            pt = lm.get(kid)
            if pt:
                frame_coords.extend([pt[0], pt[1]])
            else:
                frame_coords.extend([0.0, 0.0])
        seq.append(frame_coords)
        
    # Trim or pad sequence length to max_seq_len
    if len(seq) < max_seq_len:
        padding = [[0.0] * (len(KEYPOINT_IDS) * 2) for _ in range(max_seq_len - len(seq))]
        seq.extend(padding)
    else:
        seq = seq[:max_seq_len]
        
    return np.array([seq], dtype=np.float32)

class ActionClassifier:
    def __init__(self, model_path="action_classifier.pth"):
        self.model = ActionLSTM()
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        abs_model_path = os.path.join(base_dir, model_path)
        if os.path.exists(abs_model_path):
            try:
                self.model.load_state_dict(torch.load(abs_model_path, map_location=torch.device('cpu')))
                self.model.eval()
                print(f"[Info] Action Classifier model loaded from {abs_model_path}")
            except Exception as e:
                print(f"[Warn] Failed to load Action Classifier: {e}")
        else:
            print(f"[Info] Action Classifier weights not found at {abs_model_path}. Using default initialization.")
            
    def predict(self, all_landmarks) -> str:
        """
        Predicts the action class from a pose sequence.
        Returns one of: 'batting', 'bowling', 'fielding', 'invalid'
        """
        seq = format_pose_sequence(all_landmarks)
        x = torch.tensor(seq)
        
        with torch.no_grad():
            outputs = self.model(x)
            pred_idx = torch.argmax(outputs, dim=1).item()
            
        classes = ['batting', 'bowling', 'fielding', 'invalid']
        return classes[pred_idx]
