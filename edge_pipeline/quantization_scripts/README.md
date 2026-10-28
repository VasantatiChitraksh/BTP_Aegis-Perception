# Edge Quantization Scripts

This folder (`edge_pipeline/quantization_scripts/`) contains the quantization algorithms and benchmarking tools for the edge deployment pipeline.

## Structure & Modules

1. **`custom_quantizer.py` — Devised Quantization Algorithm Framework:**
   * Contains `DevisedEdgeQuantizer` tailored for all-weather restoration architectures.
   * Includes adaptive outlier clipping (percentile-variance curves) to protect attention softmax ranges.
   * Features channel-wise scale computation for Conv2d layers.
   * **Custom Algorithm Hook:** An extensible interface where your new quantization mathematics can be plugged in directly.

2. **`edge_quantize.py` — Benchmark & Profiling CLI:**
   * Runs baseline FP32 vs your devised quantization algorithm.
   * Measures inference latency (ms), weight memory reduction (MB), compression ratio ($X\times$), and relative PSNR/SSIM retention.
   * Saves `edge_quantization_report.json` and quantized model weights `restormer_quantized_sim.pt`.

3. **`export_trt_edge.py` — TensorRT Compilation:**
   * Exports static ONNX graph and compiles TensorRT engines for the Jetson Orin Nano / JetAuto Pro.
