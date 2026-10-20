# Jetson & Rover (JetAuto Pro) Edge Quantization Guide

> **Hardware Targets:**
> 1. **JetAuto Pro AI Robot Car:** Jetson Nano (Maxwell 128 CUDA cores, 4 GB LPDDR4, STM32, ROS, 3D Depth Camera, EAI G4 LiDAR)
> 2. **Edge Accelerator:** NVIDIA Jetson Orin Nano (Ampere 512 CUDA cores, 16 Tensor Cores, 6-core Cortex-A78AE)
> 3. **CPU Edge Node:** Raspberry Pi 5 (Quad-core Cortex-A76)

---

## 1. Executive Summary: What Works Best Where?

| Hardware | Target Device | Best Quantization Mode | Runtime Engine | Expected Speedup | Expected PSNR Impact |
|---|---|---|---|---|---|
| **JetAuto Pro Rover** | Jetson Nano (Maxwell 4GB) | **TensorRT FP16** | NVIDIA TensorRT C++/PyCUDA | **2.0× – 2.5×** | **< 0.05 dB** (Near Zero) |
| **Standalone Edge GPU** | Jetson Orin Nano (Ampere) | **TensorRT FP16 (Primary) / INT8 (Max Speed)** | NVIDIA TensorRT Tensor Cores | **2.2× (FP16) / 3.4× (INT8)** | **< 0.05 dB (FP16) / < 0.25 dB (INT8)** |
| **Edge CPU** | Raspberry Pi 5 (3 units) | **PyTorch / ONNX INT8 Dynamic / Static** | ONNX Runtime (QNNPACK) | **1.5× – 2.0×** | **0.2 – 0.4 dB** |

---

## 2. Crucial Findings: Why FP16 is King on the JetAuto Pro Car

### ⚠️ The Jetson Nano Maxwell Trap (JetAuto Pro Controller)
The original Jetson Nano powering the JetAuto Pro car uses NVIDIA's **Maxwell (sm_53)** architecture.
1. **Maxwell has NO native INT8 hardware instructions (no DP4A support)** and **NO Tensor Cores**.
2. If you try to force INT8 on a Jetson Nano, TensorRT or PyTorch must emulate INT8 using 32-bit math, which is **slower than FP16** and increases memory overhead!
3. **Maxwell HAS native FP16 execution units**: FP16 runs at **2× the rate of FP32** on Maxwell.
4. **Conclusion for JetAuto Pro**: **TensorRT FP16 is the strict winner.** It gives maximum FPS, minimal battery drain, and zero image degradation.

### ⚡ The Jetson Orin Nano Advantage
The Jetson Orin Nano features **Ampere architecture with 16 3rd-Gen Tensor Cores**.
- **FP16** runs directly on Tensor Cores: instant 2× speedup without any calibration data.
- **INT8** runs directly on INT8 Tensor Cores: achieves up to **3.5× speedup**, which is critical if you want to run high-resolution de-weathering at $\ge 30$ FPS while simultaneously running YOLOv8 detection and ROS navigation!

---

## 3. Post-Training Quantization (PTQ) vs Quantization-Aware Training (QAT)

### Question: *Should we quantize post or during training?*

### ✅ Recommended Strategy: **PTQ First, QAT Only If Necessary**

1. **Phase 1: Post-Training Quantization (PTQ) via TensorRT FP16 (Immediate)**
   - Train the models normally in PyTorch FP32/BF16 on your DGX A100 cluster.
   - Export best checkpoints to ONNX:
     ```bash
     python scripts/deployment/export_onnx.py --checkpoint artifacts/restoration/liteweatherformer/best.pt --output artifacts/onnx/liteweatherformer.onnx
     ```
   - Build TensorRT FP16 engine on Jetson:
     ```bash
     /usr/src/tensorrt/bin/trtexec --onnx=artifacts/onnx/liteweatherformer.onnx --saveEngine=artifacts/tensorrt/liteweatherformer_fp16.engine --fp16
     ```
   - **Result:** You get instant acceleration with **< 0.05 dB PSNR degradation**. No fine-tuning needed!

2. **Phase 2: PTQ INT8 with Calibration Cache (For Jetson Orin Nano)**
   - Use ~100 representative rain/fog/smoke images from `data/raw/` to calibrate the scale factors using `IInt8EntropyCalibrator2`.
   - If the PSNR drop is **under 0.3 dB**, PTQ is 100% sufficient and you are ready to deploy!

3. **Phase 3: Quantization-Aware Training (QAT) (Fallback Only)**
   - Only activate QAT if:
     - PTQ INT8 causes artifacts or drops PSNR by $> 0.5 \text{ dB}$.
     - Downstream YOLOv8 object detection accuracy (mAP) drops significantly on degraded images.
   - In QAT, you insert fake-quantization operators into the PyTorch network and fine-tune for 5–10 epochs on DGX with learning rate $1\times 10^{-5}$.

---

## 4. Model Selection for the Rover & Jetson

| Rank | Model | Parameters | Edge Deployability | Why it fits the Rover |
|---|---|---|---|---|
| 🥇 | **LiteWeatherFormer** | ~3.8 M | ⭐⭐⭐⭐⭐ (Best) | Hybrid architecture: Conv stem preserves road lane/sign edges; efficient low-resolution transformer captures haze/smoke; weather prompts allow 1 model for all conditions. |
| 🥈 | **RAMiT** | ~4.1 M | ⭐⭐⭐⭐⭐ | Extremely lightweight; pure conv + channel/spatial attention; minimal memory footprint on Jetson Nano. |
| 🥉 | **Attention-UNet** | ~34 M | ⭐⭐⭐⭐ | Strong baseline; 100% standard Conv2d operations; exports cleanly to TensorRT. |
| 4 | **TransWeather** | ~21 M | ⭐⭐⭐ | Good all-in-one baseline, but transformer attention requires more memory on 4GB Jetson. |
| 5 | **Restormer** | ~26 M | ⭐⭐ | Highest restoration quality, but heavy memory consumption; risk of latency bottleneck on JetAuto Pro. |

---

## 5. Network Pruning: Removing Unnecessary Layers

To fit models into the **4 GB RAM** and **10W power budget** of the JetAuto Pro car without throttling:
1. Run magnitude-based unstructured pruning using:
   ```bash
   python quantization/scripts/prune_model.py \
       --checkpoint artifacts/restoration/liteweatherformer/best.pt \
       --sparsity 0.3 \
       --method global \
       --image data/raw/rain/realrain1k/rain/0001.jpg \
       --output-dir artifacts/quantization/pruning
   ```
2. **What layers are safe to prune?**
   - **Deep bottleneck channels:** In U-Net and LiteWeatherFormer, layers at the lowest resolution (bottleneck) have high channel redundancy ($512$ or $1024$ channels). Pruning up to 50% of weights here causes almost zero perceptual degradation.
   - **Attention linear projections:** Minor attention heads that specialize in high-frequency noise can be pruned without damaging semantic object shapes.
   - **Layers to NEVER prune heavily:** The initial input convolution stem (`down1` / `stem`) and final output reconstruction layer (`final` / `head`). Pruning these directly destroys edge details required by YOLOv8!

---

## 6. End-to-End Rover Perception Pipeline (ROS + Camera + TensorRT)

On the **JetAuto Pro Robot Car**:
```
[3D Depth Camera / USB]
        │  (640x480 RGB @ 30 FPS via ROS /camera/rgb/image_raw)
        ▼
[Preprocessing Node] (Resize to 256x256, normalize to [-1, 1])
        │
        ▼
[Deweathering TensorRT FP16 Engine] (LiteWeatherFormer ~11ms on Orin Nano / ~28ms on Jetson Nano)
        │
        ▼
[Restored Image Topic] (/camera/deweathered/image_raw)
        │
        ▼
[YOLOv8n TensorRT Engine] (~8ms on Orin Nano / ~22ms on Jetson Nano)
        │
        ▼
[Bounding Boxes & Detections] -> [ROS Nav2 / Obstacle Avoidance / 6DOF Robotic Arm]
```

To test real-time inference on the JetAuto Pro camera directly:
```bash
python quantization/scripts/rover_infer_trt.py \
    --engine artifacts/tensorrt/liteweatherformer_fp16.engine \
    --camera 0
```
