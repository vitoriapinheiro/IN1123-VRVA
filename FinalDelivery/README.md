# FinalDelivery — Cross-dataset comparison

| File | Description |
|------|-------------|
| `viddar_cross_dataset_comparison.ipynb` | **v1** — first synthetic COCO run |
| `viddar_cross_dataset_comparison2.ipynb` | **v2** — balanced data + better DINO queries + Option B fine-tune |
| `viddar_cross_dataset_comparison3.ipynb` | **v3 (recommended)** — SAM-ranked box selection on the **same v2 images** |
| `Results/` | Saved Colab outputs |
| `_build_comparison*.py` | Rebuild notebooks (optional) |

Paper: `../Entrega4/ThirdRound/Project_vrva.tex`

## Which notebook to run

| Goal | Notebook |
|------|----------|
| Need the synthetic dataset on Drive | Run **v2** Section [1] once |
| Best current method | Run **v3** (requires v2 dataset already generated) |

### What each version fixed

1. **v1 → v2:** class balance; DINO synonyms / specific COCO names; (fine-tune B — did **not** help).
2. **v2 → v3:** wrong-box errors (especially vehicles). Among DINO candidates, pick the box whose SAM multimask has the highest **predicted** IoU. No decoder fine-tuning.

### How to run v3

1. Upload `viddar_cross_dataset_comparison3.ipynb` to Colab (GPU).
2. Ensure v2 already created `MyDrive/viddar_samed/cross_dataset_v2/synthetic_coco_balanced/`.
3. Run all cells (~20–35 min).
4. Outputs: `MyDrive/viddar_samed/cross_dataset_v3/`.
