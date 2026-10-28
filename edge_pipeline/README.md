# Edge Deployment Pipeline — BTP Aegis Perception

Welcome to the unified edge deployment pipeline. This package is structured to seamlessly import trained checkpoints (`best.pt` of Restormer), apply your **devised edge quantization algorithm**, load **YOLOv8** road-object detection, and execute the complete **end-to-end perception pipeline** on target edge hardware:
* **JetAuto Pro AI Robot Car** (NVIDIA Jetson Nano 4GB Maxwell)
* **NVIDIA Jetson Orin Nano** (Ampere with 16 Tensor Cores)
* **Raspberry Pi 5** (Quad-Core Cortex-A76 CPU)

---

## Directory Architecture

```
edge_pipeline/
├── README.md                      <-- This guide
│
├── import_restormer/              <-- Checkpoint importer & validator
│   ├── README.md
│   └── import_restormer.py        <-- Validates architecture, tests forward pass, strips optimizer
│
├── yolov8_loader/                 <-- YOLOv8 road object detector loader
│   ├── README.md
│   └── load_yolov8.py             <-- Letterbox, NMS, road class filtering [0,1,2,3,5,7]
│
├── quantization_scripts/          <-- Devised Edge Quantization Algorithm suite
│   ├── README.md
│   ├── custom_quantizer.py        <-- Extensible framework for your custom quantization math
│   ├── edge_quantize.py           <-- Quantization CLI & profiling (compression, latency, PSNR)
│   └── export_trt_edge.py         <-- TensorRT engine compilation for Jetson Orin / JetAuto
│
├── final_pipeline/                <-- The Final End-to-End Perception Pipeline
│   ├── README.md
│   ├── end_to_end_pipeline.py     <-- Chained Restormer -> YOLOv8 class
│   └── run_inference.py           <-- CLI runner for images, directories, video, and live camera
│
└── checkpoints/                   <-- Central model repository
    ├── restormer/                 <-- Place trained best.pt here
    └── yolov8/                    <-- Place or auto-download yolov8n.pt here
```

---

## Quickstart: Step-by-Step Execution

### Step 1: Import Restormer Checkpoint
Drop your trained `best.pt` into `edge_pipeline/checkpoints/restormer/` and run:
```bash
python edge_pipeline/import_restormer/import_restormer.py \
    --checkpoint edge_pipeline/checkpoints/restormer/best.pt \
    --output-dir edge_pipeline/checkpoints/restormer
```
*(Tip: If DGX training is still running, create an initialized template for instant local dry-runs:)*
```bash
python edge_pipeline/import_restormer/import_restormer.py --create-template
```

### Step 2: Apply Your Devised Quantization Algorithm
Quantize the Restormer model using your custom quantization formulation:
```bash
python edge_pipeline/quantization_scripts/edge_quantize.py \
    --checkpoint edge_pipeline/checkpoints/restormer/restormer_imported.pt \
    --image data/test_rain_sample.jpg \
    --output-dir edge_pipeline/checkpoints/restormer/quantized \
    --bits 8
```
This evaluates latency, model compression ratio ($3\times\text{--}4\times$), and PSNR retention, outputting `edge_quantization_report.json`.

### Step 3: Run the Final Perception Pipeline
Run the chained restoration + YOLOv8 detection pipeline on a degraded test image or live camera stream:
```bash
python edge_pipeline/final_pipeline/run_inference.py \
    --checkpoint edge_pipeline/checkpoints/restormer/restormer_imported.pt \
    --image data/test_rain_sample.jpg \
    --output-dir artifacts/edge_v1/pipeline_outputs
```
This produces an annotated split-view image comparing raw degraded detection against restored perception.
