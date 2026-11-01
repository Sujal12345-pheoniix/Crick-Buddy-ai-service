"""
YOLOv8 inference on ONNX Runtime — no PyTorch at runtime.

PyTorch + Ultralytics cost ~300 MB of RAM just to run two small detectors, which
pushes the service past a 512 MB hosting limit. The models in models/ are the
Ultralytics yolov8n / yolov8n-pose weights exported to ONNX (dynamic input size);
this module reproduces Ultralytics' pre- and post-processing for them.

Re-export (needs ultralytics + onnx, in a separate env because onnx needs protobuf>=5):
    YOLO("yolov8n.pt").export(format="onnx", dynamic=True, simplify=True, opset=17)
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

import cv2
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


@dataclass
class Detection:
    box: List[float]                     # [x1, y1, x2, y2] in source-image pixels
    conf: float
    cls: int
    keypoints: Optional[np.ndarray] = None   # (17, 3) x, y, confidence (pose model only)


class YoloOnnx:
    def __init__(self, path: str, threads: Optional[int] = None):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # No growing memory pool: inputs vary in size and RAM is the scarce resource here.
        opts.enable_cpu_mem_arena = False
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.is_pose = self.session.get_outputs()[0].shape[1] == 56

    @staticmethod
    def _letterbox(img: np.ndarray, size: int):
        """Resize keeping aspect into a stride-32 canvas, as Ultralytics does (grey 114 padding)."""
        h, w = img.shape[:2]
        r = min(size / h, size / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        # minimal rectangle padded to a multiple of 32 (Ultralytics "auto" letterbox)
        cw, ch = int(np.ceil(nw / 32) * 32), int(np.ceil(nh / 32) * 32)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else img
        dw, dh = (cw - nw) / 2, (ch - nh) / 2
        top, left = int(round(dh - 0.1)), int(round(dw - 0.1))   # same split as Ultralytics' LetterBox
        canvas = np.full((ch, cw, 3), 114, dtype=np.uint8)
        canvas[top:top + nh, left:left + nw] = resized
        return canvas, r, left, top

    def __call__(self, img_bgr: np.ndarray, imgsz: int = 640, conf: float = 0.25, iou: float = 0.7,
                 classes: Optional[Sequence[int]] = None, max_det: int = 50) -> List[Detection]:
        canvas, r, pad_x, pad_y = self._letterbox(img_bgr, imgsz)
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.session.run(None, {self.input_name: blob})[0][0].T          # (anchors, 84 | 56)

        if self.is_pose:
            scores, cls_ids = out[:, 4], np.zeros(len(out), dtype=int)
        else:
            cls_scores = out[:, 4:]
            cls_ids = cls_scores.argmax(1)
            scores = cls_scores[np.arange(len(out)), cls_ids]
        keep = scores >= conf
        if classes is not None:
            keep &= np.isin(cls_ids, list(classes))
        if not np.any(keep):
            return []
        out, scores, cls_ids = out[keep], scores[keep], cls_ids[keep]

        cx, cy, bw, bh = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        # class-aware NMS: offset boxes per class so different classes never suppress each other
        shifted = boxes + cls_ids[:, None] * 4096.0
        rects = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in shifted]   # x, y, w, h
        idx = cv2.dnn.NMSBoxes(rects, scores.tolist(), conf, iou)
        idx = np.array(idx).reshape(-1)[:max_det]

        h, w = img_bgr.shape[:2]
        dets = []
        for i in idx:
            b = boxes[i].copy()
            b[[0, 2]] = np.clip((b[[0, 2]] - pad_x) / r, 0, w)
            b[[1, 3]] = np.clip((b[[1, 3]] - pad_y) / r, 0, h)
            kp = None
            if self.is_pose:
                kp = out[i, 5:].reshape(17, 3).copy()
                kp[:, 0] = (kp[:, 0] - pad_x) / r
                kp[:, 1] = (kp[:, 1] - pad_y) / r
            dets.append(Detection([float(v) for v in b], float(scores[i]), int(cls_ids[i]), kp))
        return dets


def load_model(env_var: str, default_name: str) -> YoloOnnx:
    path = os.getenv(env_var) or os.path.join(MODELS_DIR, default_name)
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(MODELS_DIR), path)
    return YoloOnnx(path)
