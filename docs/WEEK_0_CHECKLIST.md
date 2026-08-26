# Kickoff checklist

The v2 schedule places 24 August–6 September 2026 in Weeks 1–2, so overdue Week 0
items should be closed before GPU-heavy work.

- [ ] Commit the supplied proposal/report files without modifying them.
- [ ] Recover the exact smoke archive and historical Colab/checkpoint artifacts.
- [ ] Confirm GPU allocation, storage budget, and experiment tracker.
- [ ] Confirm Jetson model and access date; record JetPack/TensorRT versions.
- [ ] Create dataset cards and archive checksums.
- [ ] Build and audit smoke split manifest.
- [ ] Verify DAWN release, labels, class mapping, and frozen split.
- [ ] Run environment/tests and a one-batch model smoke test on a PyTorch machine.
- [ ] Launch E1 vanilla/attention reproduction with seeds 42/43/44.
- [ ] Run zero-shot YOLOv8n motivating baseline before training new restorers.

Meeting decision needed: whether the publication minimum uses fog or snow as the
third condition. Both remain configured; run both only if the data and compute
are ready without threatening the core submission.
