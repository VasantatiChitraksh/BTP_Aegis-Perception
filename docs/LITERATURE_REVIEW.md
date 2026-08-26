# Focused literature review

This review is scoped to the revised execution plan: efficient restoration,
task-driven object detection, multi-weather generalisation, and edge inference.
Links point to primary papers, official proceedings, or official dataset pages.

## Findings that shape the experiments

1. **A visually better image is not necessarily a better detector input.**
   RestoreX-AI reports cases where more denoising does not improve detection,
   while [UniRestore](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_UniRestore_Unified_Perceptual_and_Task-Oriented_Image_Restoration_Model_Using_Diffusion_CVPR_2025_paper.html)
   explicitly separates perceptual and task-oriented restoration. Therefore mAP
   is primary; PSNR/SSIM are supporting metrics.
2. **All-in-one restoration is credible but not automatically edge-efficient.**
   [All-in-One](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_All_in_One_Bad_Weather_Removal_Using_Architectural_Search_CVPR_2020_paper.html)
   uses multiple task-specific encoders, while
   [TransWeather](https://openaccess.thecvf.com/content/CVPR2022/html/Valanarasu_TransWeather_Transformer-Based_Restoration_of_Images_Degraded_by_Adverse_Weather_Conditions_CVPR_2022_paper.html)
   uses one encoder/decoder with weather queries. This motivates separate models
   first and one conditional model only as a stretch comparison.
3. **Real-vs-synthetic generalisation is a central risk.** Synthetic paired data
   enables fidelity metrics, but DAWN supplies real adverse images. The paper
   needs both controlled paired experiments and a held-out real-weather test.
4. **Task-adaptive lightweight preprocessing is a serious competing direction.**
   [ERUP-YOLO](https://openaccess.thecvf.com/content/WACV2025/papers/Ogino_ERUP-YOLO_Enhancing_Object_Detection_Robustness_for_Adverse_Weather_Condition_by_WACV_2025_paper.pdf)
   learns differentiable preprocessing filters for adverse-weather detection.
   It is a useful non-generative baseline if time remains.
5. **Diffusion quality comes with deployment and fidelity risks.**
   [WeatherDiffusion](https://arxiv.org/abs/2207.14626) uses patch-based iterative
   restoration, while
   [DiffIR](https://openaccess.thecvf.com/content/ICCV2023/html/Xia_DiffIR_Efficient_Diffusion_Model_for_Image_Restoration_ICCV_2023_paper.html)
   reduces work by diffusing a compact prior. These are future-work baselines,
   not the core 13-week implementation.

## Must-read set before modelling

| Paper | Why it is required | Decision/use here |
|---|---|---|
| [Pix2Pix, CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.html) | Defines the cGAN + L1 baseline and paired translation protocol | Reproduction baseline |
| [Attention U-Net, 2018](https://arxiv.org/abs/1804.03999) | Defines additive attention gates for skip features | Sem-5 contribution being ablated |
| [TransWeather, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Valanarasu_TransWeather_Transformer-Based_Restoration_of_Images_Degraded_by_Adverse_Weather_Conditions_CVPR_2022_paper.html) | Single-model rain/fog/snow restoration with weather embeddings | Unified-model stretch baseline |
| [PromptIR, NeurIPS 2023](https://openreview.net/pdf/cbe1dce66d50ff3df554eb2f0f78eaab057b2d80.pdf) | Blind all-in-one restoration via degradation prompts | Modern pretrained comparison if compatible weights are available |
| [Weather-general/specific features, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_Learning_Weather-General_and_Weather-Specific_Features_for_Image_Restoration_Under_Multiple_CVPR_2023_paper.html) | Separates shared and condition-specific parameters and introduces a real multi-weather benchmark | Design reference for conditional model |
| [Adverse Weather Removal with Codebook Priors, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Ye_Adverse_Weather_Removal_with_Codebook_Priors_ICCV_2023_paper.html) | Shows texture/structure recovery limits of unified models | Failure-analysis reference; not edge baseline |
| [RestoreX-AI, CVPRW 2022](https://openaccess.thecvf.com/content/CVPR2022W/V4AS/html/Marathe_RestoreX-AI_A_Contrastive_Approach_Towards_Guiding_Image_Restoration_via_Explainable_CVPRW_2022_paper.html) | Directly tests whether restoration helps object detection | Justifies task-driven matrix |
| [ERUP-YOLO, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/papers/Ogino_ERUP-YOLO_Enhancing_Object_Detection_Robustness_for_Adverse_Weather_Condition_by_WACV_2025_paper.pdf) | Lightweight, detector-oriented enhancement rather than human-oriented restoration | Optional efficient competitor |
| [DAWN paper](https://arxiv.org/abs/2008.05402) | Defines the real adverse-weather object-detection benchmark | Primary held-out real-weather evaluation |
| [BDD100K, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Yu_BDD100K_A_Diverse_Driving_Dataset_for_Heterogeneous_Multitask_Learning_CVPR_2020_paper.html) | Large driving data with environmental/weather diversity and detection labels | Detector fine-tuning pool |

## Dataset and degradation references

- [RainCityscapes, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Hu_Depth-Attentional_Features_for_Single-Image_Rain_Removal_CVPR_2019_paper.html)
  models rain streaks and fog as depth-dependent effects on Cityscapes. It is
  preferable to generic Rain100 for the task-driven driving experiment.
- [ACDC, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Sakaridis_ACDC_The_Adverse_Conditions_Dataset_With_Correspondences_for_Semantic_Driving_ICCV_2021_paper.html)
  contains 4,006 fog/night/rain/snow images with normal-condition
  correspondences and semantic labels. It is valuable for qualitative and
  segmentation follow-up, but is not a drop-in bounding-box benchmark.
- [SynFog, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Xie_SynFog_A_Photo-realistic_Synthetic_Fog_Dataset_based_on_End-to-end_Imaging_CVPR_2024_paper.html)
  highlights the synthetic-to-real gap in autonomous-driving defogging and is a
  stronger fog reference than simple alpha blending.
- [Efficient Scene Recovery using Luminous Flux Prior, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Efficient_Scene_Recovery_Using_Luminous_Flux_Prior_CVPR_2024_paper.html)
  is a non-learning, `O(N log N)` recovery approach worth timing as an edge
  reference for haze/sandstorm, if an implementation is available.

## Diffusion and recent advances: future-work queue

- [WeatherDiffusion](https://arxiv.org/abs/2207.14626): adverse-weather-specific,
  patch-based diffusion restoration; relevant quality baseline, likely too slow
  for the initial edge target.
- [DiffIR, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Xia_DiffIR_Efficient_Diffusion_Model_for_Image_Restoration_ICCV_2023_paper.html):
  compact prior diffusion with fewer iterations; the first diffusion architecture
  to profile in the next phase.
- [Learning Diffusion Texture Priors, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Ye_Learning_Diffusion_Texture_Priors_for_Image_Restoration_CVPR_2024_paper.html):
  diffuses texture priors rather than the whole image, relevant to fidelity.
- [UniRestore, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_UniRestore_Unified_Perceptual_and_Task-Oriented_Image_Restoration_Model_Using_Diffusion_CVPR_2025_paper.html):
  explicitly unifies perceptual and task-oriented restoration, closest to the
  long-term scientific goal.
- [MOERL, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_MOERL_When_Mixture-of-Experts_Meet_Reinforcement_Learning_for_Adverse_Weather_Image_ICCV_2025_paper.html):
  adaptive mixture-of-experts restoration; useful future all-in-one comparison,
  but too large a pivot for the current plan.

## Research gap and defensible paper claim

The strongest claim available inside the current scope is not “a new restoration
model has higher PSNR.” It is:

> A controlled attention-gated, multi-weather restoration front-end improves
> held-out road-object detection under adverse weather, and its accuracy/latency/
> power trade-off remains useful after edge quantisation.

That claim requires a clean protocol, real-weather validation, negative/failure
cases, and end-to-end hardware measurements. If attention does not improve mAP,
that is still an important result: the paper should then emphasize task-oriented
losses or bypassing restoration rather than optimize only for visual quality.
