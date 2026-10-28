# Final End-to-End Perception Pipeline

This folder (`edge_pipeline/final_pipeline/`) houses the complete, deployable edge perception loop chaining:
1. **Extreme-Weather Image Restoration** (Restormer quantized via your devised algorithm)
2. **Real-Time Road Object Detection** (YOLOv8 filtering road-critical classes: Pedestrian, Bicycle, Car, Motorcycle, Bus, Truck)
3. **Telemetry & Dual Visualization** (Split-view comparison of Raw Degraded Input vs Restored Perception)

## Pipeline Architecture

```
[Camera / Video / Image]
           │
           ▼ (Aspect-preserving resize to 256x448, [-1, 1] normalization)
[Restormer Quantized Engine]  <--- Devised Edge Quantization Algorithm
           │
           ▼ (Denormalization to [0, 255] RGB)
[Restored Road Scene]
           │
           ▼ (Letterbox 640x640)
[YOLOv8 Detection Engine]     <--- COCO Road Classes: [0, 1, 2, 3, 5, 7]
           │
           ▼ (Vectorized NMS & Coordinate Rescaling)
[Final Detection Output]      <--- Bounding Boxes, Class Confidences, Latency Telemetry
```

## How to Run

### 1. Single Image Test
```bash
python edge_pipeline/final_pipeline/run_inference.py \
    --checkpoint edge_pipeline/checkpoints/restormer/restormer_imported.pt \
    --image data/test_rain_sample.jpg \
    --output-dir artifacts/edge_v1/pipeline_outputs
```

### 2. Directory Batch Benchmark
```bash
python edge_pipeline/final_pipeline/run_inference.py \
    --checkpoint edge_pipeline/checkpoints/restormer/restormer_imported.pt \
    --input-dir data/raw/smoke/hazy \
    --output-dir artifacts/edge_v1/pipeline_outputs
```

### 3. Live Camera Stream on JetAuto Pro / Jetson
```bash
python edge_pipeline/final_pipeline/run_inference.py \
    --checkpoint edge_pipeline/checkpoints/restormer/restormer_imported.pt \
    --camera 0
```
