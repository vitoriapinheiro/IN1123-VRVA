# IN1123-VRVA

Replication and adaptation of [ViDDAR](https://github.com/YM-Xiu/ViDDAR-Dataset) for augmented-reality obstruction detection. The question is how much guidance SAM needs to find the task-relevant object and decide whether virtual content covers it.

Current paper: [`Entrega4/ThirdRound/Project_vrva.tex`](Entrega4/ThirdRound/Project_vrva.tex).

## What the pipeline does

ViDDAR names the key object, Grounding DINO localizes it, and SAM segments it. Obstruction is declared when the AR region covers at least 25% of that mask.

Two adaptations of SAM ViT-B were compared on the ViDDAR obstruction set (306 image pairs, 23 classes):

| Method | Input at inference | What is trained |
|---|---|---|
| Option A | The scene image only | LoRA in the image encoder, plus the prompt encoder and mask decoder (SAMed-style) |
| Option B | The scene plus a Grounding DINO box | Mask decoder only (MedSAM-style). The box is prompted with the ground-truth object name |

A later experiment rebuilds the same task from COCO val2017, where both the key-object mask and the AR-content mask are known.

## Repository map

| Path | What it is |
|---|---|
| [`Entrega4/`](Entrega4/README.md) | The ViDDAR replication, in the order it was done |
| [`FinalDelivery/`](FinalDelivery/README.md) | COCO comparison v1–v3, saved Colab runs, and the two demo notebooks |
| [`FinalDelivery/Demo/`](FinalDelivery/Demo/README.md) | Short demos: one photo in, a mask and an obstruction call out |
| [`FinalDelivery/Results/`](FinalDelivery/Results/README.md) | Notebooks with the Colab outputs already executed |

## Step by step

1. **Replicate ViDDAR.** [`Entrega4/viddar_entrega2.ipynb`](Entrega4/viddar_entrega2.ipynb) downloads the obstruction subset, trains Option A (three runs, because GPU noise moves the score), trains Option B, and evaluates segmentation and obstruction on a fixed 70/15/15 split. Details and the round history are in [`Entrega4/README.md`](Entrega4/README.md).
2. **Write the replication up.** The numbers in the paper come from that notebook: on the 61-image test set, Option A reaches 68.63% IoU and Option B 88.55%. Option B is 0.9 points below SAM with an oracle box. Decoder fine-tuning adds 1.7 points over zero-shot SAM with the same DINO box, and that difference includes zero in its confidence interval.
3. **Ask whether the result is specific to ViDDAR.** [`FinalDelivery/viddar_cross_dataset_comparison.ipynb`](FinalDelivery/viddar_cross_dataset_comparison.ipynb) (v1) builds a synthetic obstruction set from COCO and transfers the ViDDAR checkpoints onto it. The mean is dominated by vehicles, and DINO misses many images when the query is the coarse label `vehicle`.
4. **Fix the query and the sample.** [`viddar_cross_dataset_comparison2.ipynb`](FinalDelivery/viddar_cross_dataset_comparison2.ipynb) (v2) caps each class, queries the real COCO name with synonyms, and fine-tunes Option B on the synthetic train split. The query change is the gain. The fine-tune is not.
5. **Fix the remaining wrong boxes.** [`viddar_cross_dataset_comparison3.ipynb`](FinalDelivery/viddar_cross_dataset_comparison3.ipynb) (v3) keeps several DINO candidates and lets SAM’s predicted IoU choose the mask. It reuses the v2 images and does not train a decoder. This is the method to run if the goal is the best current segmentation.
6. **Show it on a photo.** [`FinalDelivery/Demo/`](FinalDelivery/Demo/README.md) has a COCO-method demo (upload any photo) and a ViDDAR-method demo (five real ViDDAR pairs).

## How to run

Everything that trains or segments is a Colab notebook and needs a GPU (T4 is enough).

1. In Colab: **Runtime → Change runtime type → T4 GPU**.
2. Upload the notebook named in the table below.
3. Run the cells from the top. The first setup cell mounts Google Drive at `MyDrive/viddar_samed` and reuses anything already cached there: the SAM ViT-B checkpoint, the Option A and Option B weights, the extracted ViDDAR images, and the COCO download.

| Goal | Notebook | Needs on Drive first |
|---|---|---|
| Full ViDDAR replication | `Entrega4/viddar_entrega2.ipynb` | Nothing. Section [0] downloads the dataset |
| Synthetic COCO set | `FinalDelivery/viddar_cross_dataset_comparison2.ipynb`, section [1] | The ViDDAR checkpoints, if Option A and B are part of the comparison |
| Best segmentation method | `FinalDelivery/viddar_cross_dataset_comparison3.ipynb` | The v2 folder `cross_dataset_v2/synthetic_coco_balanced/` |
| Demo on your own photo | `FinalDelivery/Demo/viddar_live_demo.ipynb` | Nothing. It downloads SAM and Grounding DINO |
| Demo on real ViDDAR scenes | `FinalDelivery/Demo/viddar_dataset_demo.ipynb` | Nothing. Upload the files in `Demo/viddar_images/` when Colab cannot see them |

Drive layout after a full run:

```
MyDrive/viddar_samed/
  checkpoints/sam_vit_b_01ec64.pth
  checkpoints/stat_v2_rep1/epoch_79.pth              # Option A
  checkpoints/option_b_maskdecoder_v2/mask_decoder_best.pth
  data/extracted/                                    # ViDDAR raw, AR, masks
  cross_dataset/                                     # v1
  cross_dataset_v2/synthetic_coco_balanced/          # v2 images, reused by v3
  cross_dataset_v3/
```

## Results in short

ViDDAR test set, 61 images. IoU:

| Method | IoU |
|---|---|
| Option A | 0.686 |
| SAM zero-shot + DINO box | 0.869 |
| Option B | 0.886 |
| SAM zero-shot + oracle box | 0.895 |

On the full 306 images, the LLaVA-NeXT + DINO + SAM replication reaches 69.41% mIoU and 84.97% obstruction accuracy. The paper reports 71.85% and 89.21% with a different DINO version.

Synthetic COCO, zero-shot SAM + DINO box:

| | v1 | v2 |
|---|---|---|
| Images scored | 36 of 67 test images | 53 of 53 |
| IoU | 0.566 | 0.657 |

Option B fine-tuned on the synthetic train split scores 0.624, below zero-shot SAM on the same boxes. The paired difference is −3.3 points and the 95% interval excludes zero. Vehicles stay the open failure in v2 (DINO IoU 0.35, oracle 0.86). v3 is the attempt to fix that by ranking boxes with SAM, and its Colab output is not saved in `Results/` yet.

The tables, confidence intervals, and per-class scores are in [`FinalDelivery/Results/README.md`](FinalDelivery/Results/README.md).
