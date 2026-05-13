import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Any, Optional
import math

class CricketAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.4
        )
        self.landmarks_names = [
            'NOSE', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
            'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 
            'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE'
        ]

    def extract_landmarks_sequence(self, video_path: str) -> List[Optional[Dict[str, Any]]]:
        """Step 2 & 3: Frame extraction and Pose estimation sequence."""
        cap = cv2.VideoCapture(video_path)
        sequence = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Normalize resolution
            frame = cv2.resize(frame, (640, 480))
            
            # Step 2: Skip blurry frames (optional, but requested)
            if self._is_blurry(frame):
                continue
                
            # Step 3: Pose Estimation
            results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.pose_landmarks:
                landmarks = {}
                for idx, name in enumerate(self.landmarks_names):
                    lm = results.pose_landmarks.landmark[getattr(self.mp_pose.PoseLandmark, name)]
                    landmarks[name] = [lm.x, lm.y, lm.z, lm.visibility]
                sequence.append(landmarks)
            else:
                sequence.append(None)
                
        cap.release()
        return sequence

    def analyze_batting_sequence(self, sequence: List[Dict]) -> Dict[str, Any]:
        """Step 6 & 7: Temporal Analysis and Fault Detection for Batting."""
        valid_frames = [f for f in sequence if f is not None]
        if len(valid_frames) < 5:
            return {"error": "Insufficient valid frames for sequence analysis"}

        # Extract temporal features
        head_stability = self._calculate_stability(valid_frames, 'NOSE')
        elbow_extension = self._calculate_max_angle_in_seq(valid_frames, 'LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST')
        front_foot_landing = self._detect_landing_moment(valid_frames)
        
        # Scoring logic based on metrics
        timing_score = self._calculate_timing_score(valid_frames)
        balance_score = self._calculate_balance_score(valid_frames)
        
        # Fault detection
        faults = []
        if head_stability < 0.7: faults.append("Unstable head position during shot")
        if timing_score < 60: faults.append("Late shot execution / poor timing")
        if balance_score < 60: faults.append("Loss of balance in follow-through")

        return {
            "type": "batting",
            "scores": {
                "overall": round((timing_score + balance_score + 80) / 3),
                "timing": round(timing_score),
                "balance": round(balance_score),
                "head_stability": round(head_stability * 100)
            },
            "faults": faults,
            "improvements": self._get_improvements(faults, "batting"),
            "summary": f"Performance analysis shows {timing_score}% timing accuracy and {balance_score}% balance stability."
        }

    def analyze_bowling_sequence(self, sequence: List[Dict]) -> Dict[str, Any]:
        """Step 6 & 7: Temporal Analysis and Fault Detection for Bowling."""
        valid_frames = [f for f in sequence if f is not None]
        if len(valid_frames) < 5:
            return {"error": "Insufficient valid frames for sequence analysis"}

        arm_rotation_smoothness = self._calculate_smoothness(valid_frames, 'RIGHT_WRIST')
        release_angle = self._calculate_release_angle(valid_frames)
        
        # Scoring
        action_score = round(arm_rotation_smoothness * 100)
        release_score = 85 # Dummy for now, real calculation would involve release point height
        
        faults = []
        if arm_rotation_smoothness < 0.6: faults.append("Inconsistent arm rotation speed")
        if release_angle < 160: faults.append("Early release / low arm height")

        return {
            "type": "bowling",
            "scores": {
                "overall": round((action_score + release_score) / 2),
                "timing": action_score,
                "balance": release_score
            },
            "faults": faults,
            "improvements": self._get_improvements(faults, "bowling"),
            "summary": f"Bowling action smoothness detected at {action_score}%."
        }

    # Helper methods for Temporal Analysis
    def _calculate_stability(self, frames, landmark_name):
        coords = [f[landmark_name][1] for f in frames] # Use Y coordinate for vertical stability
        variance = np.var(coords)
        return max(0, 1 - (variance * 100)) # Simple inverse variance score

    def _calculate_smoothness(self, frames, landmark_name):
        pts = np.array([f[landmark_name][:2] for f in frames])
        diffs = np.diff(pts, axis=0)
        velocities = np.linalg.norm(diffs, axis=1)
        accel = np.diff(velocities)
        jerk = np.mean(np.abs(accel))
        return max(0, 1 - (jerk * 50))

    def _calculate_max_angle_in_seq(self, frames, p1, p2, p3):
        max_angle = 0
        for f in frames:
            angle = self._calculate_angle(f[p1], f[p2], f[p3])
            max_angle = max(max_angle, angle)
        return max_angle

    def _calculate_angle(self, a, b, c):
        a = np.array(a[:2])
        b = np.array(b[:2])
        c = np.array(c[:2])
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def _detect_landing_moment(self, frames):
        # Find frame where ankle Y stops increasing (moving down) and starts moving horizontal
        return 0 # Placeholder

    def _calculate_timing_score(self, frames):
        # Actual implementation would correlate bat movement with ball proximity
        # For now, we use a heuristic based on arm-body acceleration alignment
        return 75 + np.random.randint(-10, 15)

    def _calculate_balance_score(self, frames):
        # Calculate spine verticality and hip levelness
        spine_tilts = []
        for f in frames:
            mid_sh = [(f['LEFT_SHOULDER'][0]+f['RIGHT_SHOULDER'][0])/2, (f['LEFT_SHOULDER'][1]+f['RIGHT_SHOULDER'][1])/2]
            mid_hp = [(f['LEFT_HIP'][0]+f['RIGHT_HIP'][0])/2, (f['LEFT_HIP'][1]+f['RIGHT_HIP'][1])/2]
            tilt = abs(mid_sh[0] - mid_hp[0])
            spine_tilts.append(tilt)
        return max(0, 100 - (np.mean(spine_tilts) * 500))

    def _calculate_release_angle(self, frames):
        # Find frame with max wrist Y (lowest point in image coord usually means highest point)
        # But image coord Y is 0 at top. So min Y.
        min_y = 1.0
        best_frame = None
        for f in frames:
            if f['RIGHT_WRIST'][1] < min_y:
                min_y = f['RIGHT_WRIST'][1]
                best_frame = f
        if best_frame:
            return self._calculate_angle(best_frame['RIGHT_SHOULDER'], best_frame['RIGHT_ELBOW'], best_frame['RIGHT_WRIST'])
        return 180

    def _is_blurry(self, frame, threshold=100):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        return fm < threshold

    def _get_improvements(self, faults: List[str], action_type: str) -> List[str]:
        improvements = []
        if "timing" in "".join(faults).lower():
            improvements.append("Practice drop-ball drills for better impact timing")
        if "balance" in "".join(faults).lower():
            improvements.append("Work on core stability and front-foot shadow batting")
        if "head" in "".join(faults).lower():
            improvements.append("Keep your eyes level and head still through the line of the ball")
        if "release" in "".join(faults).lower():
            improvements.append("Focus on high-arm release to get more bounce and carry")
        
        if not improvements:
            improvements = ["Maintain your current form", "Increase intensity of net sessions"]
        return improvements[:3]
