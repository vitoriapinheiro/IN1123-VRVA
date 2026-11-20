# Demos

Two short Colab notebooks. Each one shows a mask and an obstruction call on a single scene. Neither trains anything.

| Notebook | Method | Photos |
|---|---|---|
| [`viddar_live_demo.ipynb`](viddar_live_demo.ipynb) | COCO v3. Synonym prompts, SAM keeps the mask it trusts most, and the notebook paints the AR card | [`test_images/`](test_images/) |
| [`viddar_dataset_demo.ipynb`](viddar_dataset_demo.ipynb) | ViDDAR replication. One object name, thresholds 0.35 / 0.25, highest DINO score, one SAM mask. The AR region is the pixel difference between the raw photo and the real AR photo | [`viddar_images/`](viddar_images/) |

[`Images/`](Images/) holds a few saved screenshots from demo runs (`CupAnalysis.png`, `DogAnalysis.png`).

## COCO-method demo

1. Upload `viddar_live_demo.ipynb` to Colab and switch the runtime to a T4.
2. Run section [1] once. It clones SAM and downloads Grounding DINO Tiny.
3. Run section [2] and pick a photo. The files in `test_images/` are free Unsplash photos, one object each.
4. Set `OBJECT_NAME` to the name in the table, then run section [3]. That cell is the frame to record.
5. Set `OBSTRUCT = False` to park the card in a corner. The card is painted by the notebook. The verdict is **OBSTRUCTED** when it covers at least 25% of the mask.

| File | `OBJECT_NAME` |
|---|---|
| `cup.jpg` | `cup` |
| `laptop.jpg` | `laptop` (a cup is also on the desk) |
| `dog.jpg` | `dog` |
| `car.jpg` | `car` |
| `bottle.jpg` | `bottle` |
| `phone.jpg` | `phone` |
| `chair.jpg` | `chair` |
| `cat.jpg` | `cat` |
| `tv.jpg` | `tv` |
| `person.jpg` | `person` |

`phone.jpg` includes the corner of a laptop. Query `phone` to get the phone. The saved run in [`../Results/viddar_live_demo.ipynb`](../Results/viddar_live_demo.ipynb) uploaded `car.jpg` with `OBJECT_NAME` still set to `cup`, so that output is a cup detection on a car photo.

## ViDDAR-method demo

The five scenes are real pairs from the ViDDAR obstruction set, chosen from classes that COCO does not contain. `raw/` is the photo. `ar/AR_<name>.png` is the same photo with the virtual content. `labels.csv` is the object name and the dataset obstruction label.

| File | Object name | Dataset label |
|---|---|---|
| `fire-extinguisher_016.png` | fire extinguisher | yes |
| `boiling-pot_005.png` | boiling pot | yes |
| `exit-sign_001.png` | exit sign | no |
| `coffee-maker_008.png` | coffee maker | no |
| `electrical-outlet_005.png` | electrical outlet | yes |

1. Upload `viddar_dataset_demo.ipynb` to Colab (T4).
2. Run section [1].
3. Run section [2]. On a laptop, with the notebook opened from this folder, it finds `viddar_images/` by itself. In Colab, select `labels.csv` and every png in `raw/` and `ar/` when asked.
4. Set `FILE_NAME` in section [3] and run it. The title reports the predicted call and whether it matches the dataset label.

The difference threshold is 30 and the overlap threshold is 0.25, the same fixed setting as the replication.
