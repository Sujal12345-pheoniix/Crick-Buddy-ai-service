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
        print("[Info] Action Classifier initialized in deterministic trajectory-based mode.")
            
    def predict(self, all_landmarks) -> str:
        """
        Predicts the action class from a pose sequence using deterministic joint trajectory rules.
        Returns one of: 'batting', 'bowling', 'fielding', 'invalid'
        """
        # Filter valid frames
        valid_lms = [lm for lm in all_landmarks if lm is not None]
        if len(valid_lms) < 3:
            return "invalid"

        # Landmark IDs:
        # RIGHT_WRIST = 16, LEFT_WRIST = 15
        # RIGHT_SHOULDER = 12, LEFT_SHOULDER = 11
        rw_ys = [lm[16][1] for lm in valid_lms if 16 in lm]
        lw_ys = [lm[15][1] for lm in valid_lms if 15 in lm]
        rw_xs = [lm[16][0] for lm in valid_lms if 16 in lm]
        lw_xs = [lm[15][0] for lm in valid_lms if 15 in lm]
        
        rs_ys = [lm[12][1] for lm in valid_lms if 12 in lm]
        ls_ys = [lm[11][1] for lm in valid_lms if 11 in lm]

        # Calculate coordinates displacement ranges
        rw_range_y = max(rw_ys) - min(rw_ys) if rw_ys else 0
        lw_range_y = max(lw_ys) - min(lw_ys) if lw_ys else 0
        rw_range_x = max(rw_xs) - min(rw_xs) if rw_xs else 0
        lw_range_x = max(lw_xs) - min(lw_xs) if lw_xs else 0
        
        max_range_y = max(rw_range_y, lw_range_y)
        max_range_x = max(rw_range_x, lw_range_x)

        # Check if wrist reaches high above shoulder (release point check for bowling)
        # Note: Y goes from 0 (top of image) to 1 (bottom of image)
        rw_above_shoulder = any(rw_y < rs_y for rw_y, rs_y in zip(rw_ys, rs_ys)) if (rw_ys and rs_ys) else False
        lw_above_shoulder = any(lw_y < ls_y for lw_y, ls_y in zip(lw_ys, ls_ys)) if (lw_ys and ls_ys) else False
        reaches_high = rw_above_shoulder or lw_above_shoulder

        # Deterministic classification rules:
        # 1. Bowling: Large vertical range of motion, wrist goes above shoulder height
        if max_range_y > 0.32 and reaches_high:
            return "bowling"
        # 2. Batting: Significant horizontal motion, wrist remains mostly below shoulder height
        elif max_range_x > 0.15:
            return "batting"
        # 3. Static/Unmoving: Extremely low motion
        elif max_range_x < 0.05 and max_range_y < 0.05:
            return "invalid"
        # 4. Fallback: Moderate motion / default to fielding
        else:
            return "fielding"

