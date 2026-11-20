# Entrega 4 — ViDDAR replication

This folder is the replication on the real ViDDAR obstruction set. The COCO follow-up lives in [`../FinalDelivery/`](../FinalDelivery/README.md). The paper that reports these experiments is [`ThirdRound/Project_vrva.tex`](ThirdRound/Project_vrva.tex).

## Rounds

The rounds are kept as they were handed in. Use the root notebook and the third-round paper. The earlier copies are the history.

| Round | What changed | Files |
|---|---|---|
| First | First Colab run of the replication | [`FirstRound/viddar_entrega2.ipynb`](FirstRound/viddar_entrega2.ipynb) and the exported PDF next to it |
| Second | The run behind the second write-up: fixed split, Option A repeated three times, Option B, obstruction metrics | [`SecondRound/viddar_entrega2_(1).ipynb`](SecondRound/viddar_entrega2_(1).ipynb), [`SecondRound/project_vrva.tex`](SecondRound/project_vrva.tex) |
| Third | Paper revised around the same experiment, plus the pointer to the COCO notebook | [`ThirdRound/Project_vrva.tex`](ThirdRound/Project_vrva.tex) |

[`viddar_entrega2.ipynb`](viddar_entrega2.ipynb) at this level is the notebook to run. [`viddar_entrega2_standalone.py`](viddar_entrega2_standalone.py) is the same pipeline as a script.

## What the notebook does

Section [0] mounts Drive, clones SAM and SAMed, and downloads the obstruction subset from the [ViDDAR Drive folder](https://drive.google.com/drive/folders/1aZ-2n-_DjUDqTJuQUbqMWiohOCxcp2UO). Labels come from [`YM-Xiu/ViDDAR-Dataset`](https://github.com/YM-Xiu/ViDDAR-Dataset) (`obstruction_labels.csv`). The split is written once to `MyDrive/viddar_samed/splits.json` and then reused. Regenerating it makes later numbers incomparable.

| Section | Work | Rough time on a T4 |
|---|---|---|
| [0] | Setup, dataset cache, Grounding DINO | 3–5 min the first time, then about 30 s |
| [1] | Option A, three runs, two settings | 70–75 min |
| [2] | One qualitative example | short |
| [4] | Option B, mask decoder only | 15–30 min for 40 epochs |
| [5] | Test-set segmentation | short |
| [6] | Obstruction from the raw/AR pixel difference | short |

Grounding DINO is `IDEA-Research/grounding-dino-tiny`, with `box_threshold=0.35` and `text_threshold=0.25`. The highest-scoring box is kept. ViDDAR used Grounding DINO 1.5, so this is a replication of the pipeline with public weights.

Option A is trained at 224 px with LoRA rank 4. The same seed still moves a few IoU points because GPU kernels are not deterministic, so each Option A setting is trained three times. Option B freezes the image encoder and the prompt encoder, trains the mask decoder with Dice + BCE, and is prompted with the DINO box of the ground-truth object name.

Obstruction uses α = 0.25. The AR mask is the set of pixels whose absolute difference between the raw and AR photos exceeds a threshold. The fixed setting is difference 30 and ratio 0.25. A second setting picks both thresholds on the 37-image validation split.

## How to run

1. Upload `viddar_entrega2.ipynb` to Colab.
2. **Runtime → Change runtime type → T4 GPU**.
3. Run all cells from the top. The first run downloads the zips (raw images are about 480 MB, AR images about 640 MB) into `MyDrive/viddar_samed/data/extracted/`.
4. Leave `splits.json` in place for every later session.

## Results used in the paper

Test set, 61 images (DINO returned no box for 1 of 62):

| Method | IoU | Dice |
|---|---|---|
| Option A (prompt-free) | 0.686 | 0.787 |
| SAM zero-shot + DINO box | 0.869 | 0.911 |
| Option B (DINO box + trained decoder) | 0.886 | 0.930 |
| SAM zero-shot + oracle box | 0.895 | 0.938 |

Paired Option B − Option A is +19.9 points, 95% CI [+13.9, +25.9]. Option B − zero-shot SAM is +1.7 points, and that interval includes zero. Most of the segmentation gain is the box. The decoder fine-tune is small.

Obstruction on the same 61 images (27 obstructed), AR mask estimated by pixel difference:

| Key mask | Thresholds | Accuracy | F1 | False-negative rate |
|---|---|---|---|---|
| Ground truth | fixed (30, 0.25) | 0.918 | 0.898 | 0.185 |
| Option B | fixed (30, 0.25) | 0.918 | 0.898 | 0.185 |
| Option A | fixed (30, 0.25) | 0.902 | 0.880 | 0.185 |
| Option B | tuned on validation | 0.967 | 0.964 | 0.000 |
| Option A | tuned on validation | 0.853 | 0.848 | 0.074 |

With the fixed thresholds, Option A is 1.6 points below Option B on the decision even though its IoU is much lower. The shared errors are missed obstructions. Tuned thresholds are fit on 37 validation images, so they show what the rule can reach. They are not comparable to the original paper, which does not estimate the AR mask this way.

On all 306 images, LLaVA-NeXT + Grounding DINO + SAM reaches 69.41% mIoU and 84.97% obstruction accuracy, against 71.85% and 89.21% reported for ViDDAR’s LLaVA-NeXT configuration.
