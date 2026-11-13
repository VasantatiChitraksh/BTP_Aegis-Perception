Native-resolution controlled diagnostic

Open Native_Resolution_Restormer_Comparison.pdf first. The *_native_comparison.jpg files show every weather condition. truck_rain_native.jpg shows a gain example; do not omit the smoke regression in the complete report.

run.py uses frozen weights; adds weather at source resolution once; compares direct YOLO against native Restormer followed by identical YOLO preprocessing. Restormer input retains source pixels, padded only to multiples of eight. The 128 and 256 controls are supplementary and excluded from native_comparison.csv. Results are a small local diagnostic, not mAP or an edge performance benchmark. No Jetson accessed for new inference, no training or threshold changes.

Full per-condition predictions, matching counts, tensors and images are stored under each source/weather/mode. comparison.csv contains all 90 paths; native_comparison.csv contains 15 paired conditions. Camera is unlabelled; two labelled scenes contain five objects, repeated across conditions. Original dataset scenes are already hazy.
