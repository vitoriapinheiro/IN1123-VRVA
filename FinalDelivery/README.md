# FinalDelivery

The COCO follow-up to the ViDDAR replication, the saved Colab outputs, and the demos.

| Path | Role |
|---|---|
| [`viddar_cross_dataset_comparison.ipynb`](viddar_cross_dataset_comparison.ipynb) | v1. First synthetic COCO run |
| [`viddar_cross_dataset_comparison2.ipynb`](viddar_cross_dataset_comparison2.ipynb) | v2. Balanced classes, better DINO queries, Option B fine-tune |
| [`viddar_cross_dataset_comparison3.ipynb`](viddar_cross_dataset_comparison3.ipynb) | v3. SAM-ranked boxes on the v2 images. This is the segmentation method to use |
| [`Demo/`](Demo/README.md) | One-scene demos |
| [`Results/`](Results/README.md) | Executed notebooks and the figures saved from them |
| `_build_comparison_nb.py`, `_build_comparison2_nb.py`, `_build_comparison3_nb.py` | Rebuild the three notebooks from source. Optional |

Paper: [`../Entrega4/ThirdRound/Project_vrva.tex`](../Entrega4/ThirdRound/Project_vrva.tex).

## Why a second dataset

ViDDAR’s AR-content mask is estimated by pixel difference, so a wrong obstruction call can come from the key-object mask or from that estimate. COCO val2017 has instance masks. The notebooks paste a semi-transparent rectangle either over the object or away from it and save the exact rectangle mask. About 48% of samples are obstructed, matching ViDDAR’s 145/306. Only classes that exist in both sets are used: stop sign, fire hydrant, traffic light, knife, scissors, toaster, TV, and vehicle (`car`, `truck`, `bus`, `motorcycle`).

## What each version changed

### v1 → v2

v1 drew up to 400 instances with no per-class cap, so 259 of 400 were vehicles. DINO was prompted with the ViDDAR label (`vehicle`) at thresholds 0.35 / 0.25. The test IoU was computed only on images where a box came back: 36 of 67.

v2 makes five changes:

1. At most 50 images per class. The set is 321 images (150 obstructed), split 223 / 45 / 53. Scissors and toaster have fewer COCO instances than 50.
2. The stored query is the COCO name (`car`, `motorcycle`, `tv`), with a synonym list, thresholds 0.25 / 0.20, and the best score across prompts. The ViDDAR label is only a fallback.
3. Option B’s mask decoder is fine-tuned on the synthetic train split (Dice + BCE, AdamW at 1e-5, 25 epochs, encoders frozen). Epoch 0, the untouched ViDDAR decoder, is allowed to win.
4. DINO recall is reported, and zero-shot SAM, transferred B, adapted B, and the oracle are scored on the same images.
5. The ViDDAR column is filled with the Entrega 2 numbers. In v1 the Drive CSV used different method names, so that column was empty.

### v2 → v3

The fine-tune did not help, and vehicles stayed at DINO IoU 0.35 against an oracle of 0.86. v3 keeps up to eight DINO boxes, runs SAM with `multimask_output=True` on each, and keeps the mask whose predicted IoU is highest. No ground truth is used in that choice, and the decoder is not trained. The images are the v2 set, so a difference against v2 is the ranking, not a new sample.

## How to run

Colab, **Runtime → T4 GPU**, then run all cells from the top.

| Goal | Notebook | Time on a T4 | Writes |
|---|---|---|---|
| Build the synthetic set | v2, section [1]. v1’s COCO download is reused if it is already on Drive | 5–10 min after COCO is cached | `MyDrive/viddar_samed/cross_dataset_v2/` |
| Full v2 comparison, including the fine-tune | v2, all cells | about 45–60 min | the same folder, plus `checkpoints/mask_decoder_synth_adapted.pth` |
| Best current method | v3, after v2 section [1] | 20–35 min | `MyDrive/viddar_samed/cross_dataset_v3/` |

v3 stops immediately if `cross_dataset_v2/synthetic_coco_balanced/labels.csv` is missing.

Option A and Option B load only when these files already exist from the Entrega 4 run:

- `MyDrive/viddar_samed/checkpoints/stat_v2_rep1/epoch_79.pth`
- `MyDrive/viddar_samed/checkpoints/option_b_maskdecoder_v2/mask_decoder_best.pth`

SAM ViT-B is downloaded into `checkpoints/` on the first run. To regenerate a notebook after editing a builder, run `python3 _build_comparison2_nb.py` (or the v1 / v3 builder) from this folder.

## Results

Saved outputs: [`Results/README.md`](Results/README.md). The short version:

| Method | ViDDAR test (n=61) | Synth v1 (n=36) | Synth v2 (n=53) |
|---|---|---|---|
| SAM zero-shot + DINO | 0.869 | 0.566 | 0.657 |
| Option B, transferred | 0.886 | 0.548 | 0.626 |
| Option B, adapted on COCO | — | — | 0.624 |
| SAM + oracle box | 0.895 | 0.858 | 0.835 |
| Option A, transferred | 0.686 | 0.081 | 0.154 |

v2 DINO recall is 319/321 (99.4%). Adapted B minus zero-shot DINO is −0.033, 95% CI [−0.071, −0.003]. Adapting the decoder on synthetic boxes is worse than leaving SAM as it is. The oracle stays near 0.84–0.86, so SAM segments the object when the box is the ground-truth box.

v1 and v2 are different image draws. The v1 mean is only over images DINO already found. The v2 mean includes the images v1 dropped. Both facts are part of the result.
