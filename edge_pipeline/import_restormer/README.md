# Restormer Model Importer

This folder (`edge_pipeline/import_restormer/`) contains tools to import, validate, inspect, and prepare the trained `best.pt` of Restormer for the edge quantization and deployment pipeline.

## How to Import Your Trained Checkpoint

1. **Place your trained `best.pt` file** from DGX cluster or local training run into:
   `edge_pipeline/checkpoints/restormer/best.pt`
2. **Run the importer script:**
   ```bash
   python edge_pipeline/import_restormer/import_restormer.py \
       --checkpoint edge_pipeline/checkpoints/restormer/best.pt \
       --output-dir edge_pipeline/checkpoints/restormer
   ```
3. **What the script does automatically:**
   * Inspects checkpoint dictionary keys (`generator`, `config`, `epoch`).
   * Strips distributed training prefixes (`module.`) if present.
   * Reconstructs the exact Restormer generator (`dim=48`, `num_blocks=[4,6,6,8]`, `heads=[1,2,4,8]`, `ffn_expansion_factor=2.66`).
   * Verifies an end-to-end forward pass at edge deployment geometry ($256\times448$ or $256\times256$).
   * Saves a clean standalone package `restormer_imported.pt` and `restormer_metadata.json` ready for quantization.

## Dry-Run Without Trained Weights (Local Development)

If your DGX training job is still in progress, you can generate a verified initialized Restormer template instantly:
```bash
python edge_pipeline/import_restormer/import_restormer.py --create-template
```
This enables full pipeline testing and debugging on your laptop or Jetson without waiting for training.
