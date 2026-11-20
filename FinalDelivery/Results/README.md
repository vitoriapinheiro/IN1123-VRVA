# Saved results

These notebooks are the Colab sessions with their outputs kept. Run the copies in the parent folder when you need a new session. Those copies have empty outputs.

| File | What was executed |
|---|---|
| [`viddar_cross_dataset_comparison.ipynb`](viddar_cross_dataset_comparison.ipynb) | v1, full run, including obstruction |
| [`viddar_cross_dataset_comparison2.ipynb`](viddar_cross_dataset_comparison2.ipynb) | v2 dataset, DINO recall, test segmentation, and the v1/v2/ViDDAR table. The fine-tune log (section [3]) and the obstruction table (section [5]) were not saved in this file |
| [`viddar_live_demo.ipynb`](viddar_live_demo.ipynb) | One demo pass. See the note in [`../Demo/README.md`](../Demo/README.md) |
| `cupAnalisys.png`, `laptopAnalysis.png` | Screenshots from demo runs |

v3 has no saved notebook here yet. Its method is described in [`../README.md`](../README.md).

## v1 — synthetic COCO

400 images, 192 obstructed / 208 clear, split 276 / 57 / 67. Class counts: vehicle 259, TV 45, traffic light 34, knife 27, fire hydrant 17, stop sign 12, scissors 5, toaster 1.

Segmentation on the 36 test images that had both a DINO box and an oracle box:

| Method | n | IoU | 95% CI | Dice | IoU < 0.1 |
|---|---|---|---|---|---|
| SAM zero-shot + DINO | 36 | 0.566 | 0.43–0.70 | 0.610 | 12 |
| Option B, transferred | 36 | 0.548 | 0.42–0.67 | 0.604 | 11 |
| SAM + oracle box | 36 | 0.858 | 0.82–0.89 | 0.920 | 0 |
| Option A, transferred | 36 | 0.081 | 0.03–0.15 | 0.108 | 29 |

Paired Option B − Option A is +0.467 (CI 0.35–0.59). Paired Option B − zero-shot DINO is −0.018 (the interval includes zero).

Per-class DINO IoU on that common set: stop sign 0.93, knife 0.79, TV 0.75, scissors 0.75 (n=1), fire hydrant 0.59, traffic light 0.30, vehicle 0.30. Oracle IoU on vehicles is 0.89.

Obstruction, same 36 images. With the ground-truth key mask and the true AR mask, tuned accuracy is 1.00. With the estimated AR mask at difference 30 and ratio 0.25:

| Key mask | Accuracy | F1 | False-negative rate |
|---|---|---|---|
| Ground truth | 0.806 | 0.774 | 0.368 |
| SAM + DINO | 0.722 | 0.643 | 0.526 |
| Option B | 0.722 | 0.643 | 0.526 |
| Option A | 0.556 | 0.273 | 0.842 |

Precision is 1.0 in every row. The errors are missed obstructions. The ViDDAR column of the comparison table in this notebook is empty: the CSV on Drive used method names that did not match the pivot. v2 fills that column from the Entrega 2 table.

## v2 — balanced COCO

321 images, 150 obstructed / 171 clear, split 223 / 45 / 53. Five classes have 50 images. Stop sign has 44, scissors 20, toaster 7. DINO returns a box on 319/321 images (99.4%). The test table uses all 53 test images.

| Method | n | IoU | 95% CI | Dice | IoU < 0.1 |
|---|---|---|---|---|---|
| SAM zero-shot + DINO | 53 | 0.657 | 0.56–0.75 | 0.713 | 11 |
| Option B, ViDDAR transfer | 53 | 0.626 | 0.53–0.72 | 0.689 | 11 |
| Option B, adapted on synthetic | 53 | 0.624 | 0.52–0.72 | 0.685 | 12 |
| SAM + oracle box | 53 | 0.835 | 0.80–0.86 | 0.905 | 0 |
| Option A, transferred | 53 | 0.154 | 0.09–0.22 | 0.202 | 35 |

Paired adapted B − zero-shot DINO: −0.033, CI [−0.071, −0.003]. Paired adapted B − transferred B: −0.002, CI includes zero. Paired adapted B − Option A: +0.470.

| Method | ViDDAR | Synth v1 | Synth v2 | v2 − v1 |
|---|---|---|---|---|
| SAM zero-shot + DINO | 0.869 | 0.566 | 0.657 | +0.091 |
| Option B, transfer | — | 0.548 | 0.626 | +0.078 |
| Option B, adapted | — | 0.548 | 0.624 | +0.076 |
| SAM + oracle box | 0.895 | 0.858 | 0.835 | −0.024 |

The “adapted” row uses v1’s transferred-B score as the v1 baseline, because v1 did not fine-tune.

Per-class zero-shot DINO IoU on the v2 test set: stop sign 0.95 (n=8), fire hydrant 0.87 (8), toaster 0.73 (2), traffic light 0.72 (8), TV 0.65 (8), knife 0.50 (8), vehicle 0.35 (8), scissors 0.34 (3). Oracle IoU on vehicles is 0.86. Traffic light moved from 0.30 in v1 to 0.72. Vehicles did not.

## How to read this

The gain from v1 to v2 is the detector query: specific COCO names, synonyms, and a lower threshold. SAM with an oracle box stays near its ViDDAR level, so the segmenter is not what breaks on COCO. Option B transferred from ViDDAR stays a bit below zero-shot SAM, and fine-tuning it on synthetic DINO boxes does not recover that. Option A, trained prompt-free on ViDDAR scenes, stays near the floor. The remaining vehicle errors are wrong boxes, which is what v3 ranks with SAM’s predicted IoU.
