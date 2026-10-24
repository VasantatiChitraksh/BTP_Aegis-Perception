"""YOLOv8 COCO decoding with explicit RGB, letterbox and class-aware NMS."""

import cv2
import numpy as np

from .common import ROAD_CLASSES


def letterbox(rgb, size=640):
    height, width = rgb.shape[:2]
    gain = min(size / height, size / width)
    new_w, new_h = round(width * gain), round(height * gain)
    dw, dh = (size - new_w) / 2, (size - new_h) / 2
    left, top = round(dw - 0.1), round(dh - 0.1)
    right, bottom = round(dw + 0.1), round(dh + 0.1)
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    x = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32) / 255
    return x, {"gain": gain, "pad": (left, top), "original_wh": (width, height)}


def nms(boxes, scores, threshold):
    """Greedy IoU NMS. Returns original indices; tie order is stable."""
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        others = order[1:]
        lo = np.maximum(boxes[i, :2], boxes[others, :2])
        hi = np.minimum(boxes[i, 2:], boxes[others, 2:])
        inter = np.maximum(hi - lo, 0).prod(axis=1)
        area = np.maximum(boxes[others, 2:] - boxes[others, :2], 0).prod(axis=1)
        own = np.maximum(boxes[i, 2:] - boxes[i, :2], 0).prod()
        iou = inter / np.maximum(own + area - inter, 1e-9)
        order = others[iou <= threshold]
    return np.array(keep, dtype=int)


def decode(output, geometry, conf=0.25, iou=0.7, max_det=300, multi_label=False):
    """Decode [1,84,8400] YOLOv8n output (no objectness, no embedded NMS)."""
    output = np.asarray(output)
    if output.shape != (1, 84, 8400) or not np.isfinite(output).all():
        raise ValueError(f"Expected finite YOLOv8n [1,84,8400], received {output.shape}")
    rows = output[0].T
    if multi_label:
        anchors, classes = np.where(rows[:, 4:] > conf)
        rows = rows[anchors]
        scores = rows[np.arange(len(rows)), classes + 4]
    else:
        classes = rows[:, 4:].argmax(axis=1)
        scores = rows[np.arange(len(rows)), classes + 4]
    valid = (scores > conf) & np.isin(classes, ROAD_CLASSES)
    rows, scores, classes = rows[valid], scores[valid], classes[valid]
    boxes = np.concatenate((rows[:, :2] - rows[:, 2:4] / 2, rows[:, :2] + rows[:, 2:4] / 2), axis=1)
    keep = []
    for class_id in ROAD_CLASSES:
        indices = np.where(classes == class_id)[0]
        # Match Ultralytics' pre-NMS candidate ceiling while bounding worst-case work.
        indices = indices[np.argsort(-scores[indices], kind="stable")[:30000]]
        keep.extend(indices[nms(boxes[indices], scores[indices], iou)])
    keep = np.asarray(keep, dtype=int)
    keep = keep[np.argsort(-scores[keep], kind="stable")[:max_det]]
    boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
    left, top = geometry["pad"]
    boxes -= np.array([left, top, left, top])
    boxes /= geometry["gain"]
    width, height = geometry["original_wh"]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    return np.column_stack((boxes, scores, classes)).astype(np.float32)
