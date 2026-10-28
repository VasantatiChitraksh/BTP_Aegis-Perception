# YOLOv8 Detector Loader

This folder (`edge_pipeline/yolov8_loader/`) handles loading, format conversion, and road object detection for the edge pipeline.

## Road Objects Filtered
The detector is specifically pre-configured to detect critical ADAS and robotic car road targets:
* `Class 0`: Person / Pedestrian
* `Class 1`: Bicycle
* `Class 2`: Car
* `Class 3`: Motorcycle
* `Class 5`: Bus
* `Class 7`: Truck

## Features
* Automatic weight management: If local weights are absent, it safely initializes official `yolov8n.pt` and caches it in `edge_pipeline/checkpoints/yolov8/`.
* High-speed native letterboxing ($640\times640$), tensor normalization, and vectorized NMS.
* Dual runtime support: Direct PyTorch / Ultralytics execution or high-throughput ONNX Runtime / TensorRT backend.
* Output formatting: Provides structured coordinates `[x1, y1, x2, y2]`, confidence scores, class names, and visual bounding box overlays.
