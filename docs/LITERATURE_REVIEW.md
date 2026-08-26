# Focused literature review

This review is scoped to the revised execution plan: dataset-based multi-weather
restoration, task-driven object detection, and efficient edge inference. Links
point to primary papers or official proceedings.

## Papers referred to

### Core restoration and architecture papers

1. [Image-to-Image Translation with Conditional Adversarial Networks (Pix2Pix), CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.html)
2. [Attention U-Net: Learning Where to Look for the Pancreas, 2018](https://arxiv.org/abs/1804.03999)
3. [All in One Bad Weather Removal Using Architectural Search, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_All_in_One_Bad_Weather_Removal_Using_Architectural_Search_CVPR_2020_paper.html)
4. [TransWeather: Transformer-Based Restoration of Images Degraded by Adverse Weather Conditions, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Valanarasu_TransWeather_Transformer-Based_Restoration_of_Images_Degraded_by_Adverse_Weather_Conditions_CVPR_2022_paper.html)
5. [PromptIR: Prompting for All-in-One Blind Image Restoration, NeurIPS 2023](https://openreview.net/pdf/cbe1dce66d50ff3df554eb2f0f78eaab057b2d80.pdf)
6. [Learning Weather-General and Weather-Specific Features for Image Restoration Under Multiple Adverse Weather Conditions, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_Learning_Weather-General_and_Weather-Specific_Features_for_Image_Restoration_Under_Multiple_CVPR_2023_paper.html)
7. [Adverse Weather Removal with Codebook Priors, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Ye_Adverse_Weather_Removal_with_Codebook_Priors_ICCV_2023_paper.html)

### Task-driven detection and efficient-processing papers

8. [RestoreX-AI: A Contrastive Approach Towards Guiding Image Restoration via Explainable AI Systems, CVPR Workshops 2022](https://openaccess.thecvf.com/content/CVPR2022W/V4AS/html/Marathe_RestoreX-AI_A_Contrastive_Approach_Towards_Guiding_Image_Restoration_via_Explainable_CVPRW_2022_paper.html)
9. [ERUP-YOLO: Enhancing Object Detection Robustness for Adverse Weather by Unified Image-Adaptive Processing, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/papers/Ogino_ERUP-YOLO_Enhancing_Object_Detection_Robustness_for_Adverse_Weather_Condition_by_WACV_2025_paper.pdf)
10. [Efficient Scene Recovery Using Luminous Flux Prior, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Efficient_Scene_Recovery_Using_Luminous_Flux_Prior_CVPR_2024_paper.html)

### Dataset and adverse-driving papers

11. [DAWN: Vehicle Detection in Adverse Weather Nature Dataset, 2020](https://arxiv.org/abs/2008.05402)
12. [BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Yu_BDD100K_A_Diverse_Driving_Dataset_for_Heterogeneous_Multitask_Learning_CVPR_2020_paper.html)
13. [Depth-Attentional Features for Single-Image Rain Removal / RainCityscapes, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Hu_Depth-Attentional_Features_for_Single-Image_Rain_Removal_CVPR_2019_paper.html)
14. [ACDC: The Adverse Conditions Dataset with Correspondences for Semantic Driving Scene Understanding, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Sakaridis_ACDC_The_Adverse_Conditions_Dataset_With_Correspondences_for_Semantic_Driving_ICCV_2021_paper.html)
15. [SynFog: A Photo-realistic Synthetic Fog Dataset for Advancing Real-World Defogging in Autonomous Driving, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Xie_SynFog_A_Photo-realistic_Synthetic_Fog_Dataset_based_on_End-to-end_Imaging_CVPR_2024_paper.html)

### Diffusion and recent methods reviewed only for future work

16. [Restoring Vision in Adverse Weather Conditions with Patch-Based Denoising Diffusion Models (WeatherDiffusion), 2022](https://arxiv.org/abs/2207.14626)
17. [DiffIR: Efficient Diffusion Model for Image Restoration, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Xia_DiffIR_Efficient_Diffusion_Model_for_Image_Restoration_ICCV_2023_paper.html)
18. [Learning Diffusion Texture Priors for Image Restoration, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Ye_Learning_Diffusion_Texture_Priors_for_Image_Restoration_CVPR_2024_paper.html)
19. [UniRestore: Unified Perceptual and Task-Oriented Image Restoration Model Using Diffusion Prior, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_UniRestore_Unified_Perceptual_and_Task-Oriented_Image_Restoration_Model_Using_Diffusion_CVPR_2025_paper.html)
20. [MOERL: When Mixture-of-Experts Meet Reinforcement Learning for Adverse Weather Image Restoration, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_MOERL_When_Mixture-of-Experts_Meet_Reinforcement_Learning_for_Adverse_Weather_Image_ICCV_2025_paper.html)

## Main findings

### 1. Better-looking restoration does not guarantee better detection

RestoreX-AI shows that greater visual denoising can fail to improve object
detection. UniRestore similarly distinguishes perceptual restoration from
task-oriented restoration. Therefore:

- downstream mAP must be the primary metric;
- PSNR and SSIM are supporting metrics for paired datasets;
- erased, shifted, or hallucinated road objects must be treated as failures even
  when an output looks visually cleaner.

### 2. The semester-5 attention result is a baseline, not sufficient novelty

Pix2Pix and Attention U-Net establish the two main components already used in
the semester-5 work. Attention gates are worth reproducing and ablating, but
adding them to U-Net is not by itself a strong new 2027 contribution.

The stronger contribution would be a rigorous demonstration of when restoration
improves adverse-weather detection and whether that improvement survives edge
optimization.

### 3. Separate weather models are the safest first deliverable

All-in-One, TransWeather, PromptIR, and weather-general/weather-specific methods
show that one model can handle several degradations. However, unified models add
training and architecture risk.

The current plan should first reproduce separate smoke, rain, fog, and snow
models. A unified conditional model should remain a stretch experiment after the
main result table exists.

### 4. Dataset choice determines what can be claimed

- Paired datasets allow clean/degraded/restored fidelity comparison.
- RainCityscapes and Foggy Cityscapes are valuable because they retain driving
  scenes and usable annotations, but their weather is synthetic.
- DAWN provides real rain, fog, snow, and sandstorm scenes with detection boxes,
  making it important for real-weather validation.
- DAWN does not provide a matched clean image for each adverse scene, so it
  supports raw-adverse versus restored comparison—not a same-scene clean test.
- ACDC provides adverse/normal correspondences but its native task is semantic
  segmentation, so it is not a direct DAWN-style box-detection replacement.

### 5. Real-versus-synthetic generalization is a central research risk

Strong results on paired synthetic datasets may not transfer to naturally
occurring rain, fog, or snow. The evaluation therefore needs both:

1. controlled paired-dataset measurements; and
2. held-out real-weather evaluation on DAWN.

No locally generated weather images are required for this plan.

### 6. A weather-trained detector is a mandatory competing baseline

ERUP-YOLO and related task-adaptive approaches show that detector-oriented
preprocessing may outperform a visually oriented restoration front-end. The
project must compare:

1. a detector trained on clean data;
2. a detector fine-tuned on available adverse-weather dataset images; and
3. the detector receiving restored images.

Without this comparison, it would be unclear whether the added restoration
network provides value beyond ordinary adverse-weather training.

### 7. Edge efficiency must be measured end to end

A lightweight detector alone does not establish a real-time pipeline. The paper
must measure restoration, detection, transfers, preprocessing, and postprocessing
together. FP32/FP16/INT8 results should report mAP, latency, FPS, memory, power,
and energy per frame.

### 8. Diffusion should remain future work for this cycle

WeatherDiffusion demonstrates strong adverse-weather restoration but relies on
iterative denoising. DiffIR and diffusion-prior methods reduce that cost, while
UniRestore is especially relevant to the longer-term task-oriented goal.

These papers should inform future work, but implementing generic diffusion image
generation or diffusion restoration now would add complexity and conflict with
the scoped v2 delivery plan.

## Defensible paper direction

The strongest claim available inside the current scope is:

> A controlled attention-gated, multi-weather restoration front-end improves
> held-out road-object detection under adverse weather, and its accuracy,
> latency, and power trade-off remains useful after edge quantization.

This claim must be accepted or rejected using held-out datasets and downstream
detection evidence. If restoration does not improve mAP, the result should guide
the project toward task-oriented preprocessing or a detector-only solution.
