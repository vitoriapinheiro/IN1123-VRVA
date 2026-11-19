#!/usr/bin/env python3
"""Build viddar_cross_dataset_comparison3.ipynb."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "viddar_cross_dataset_comparison3.ipynb"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)}


def code(s):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": s.strip("\n").splitlines(True),
    }


cells = []

cells.append(md("""# ViDDAR — Cross-dataset comparison **v3**

## What v2 taught us
- Better DINO queries fixed recall (99%) and raised IoU (~+9 pp).
- Fine-tuning Option B on synthetic data **did not help** (adapted ≤ zero-shot).
- Remaining errors are mostly **wrong boxes** (vehicle IoU ~0.34 with DINO vs ~0.86 with oracle).

## What v3 changes
1. **Reuse the balanced v2 dataset** (same images → fair v2 vs v3 comparison).
2. **SAM-ranked box selection (no GT):** collect top-K DINO boxes across synonym prompts; for each box run SAM with `multimask_output=True`; keep the (box, mask) with the highest SAM-predicted IoU. This replaces “trust DINO’s top score”.
3. **No decoder fine-tuning** (skipped on purpose after v2).
4. Report **micro IoU** and **macro IoU** (mean of per-class means) so one class cannot dominate.
5. Side-by-side **v1 / v2 / v3 / ViDDAR** table.

**Runtime (T4):** ~20–35 min if v2 data+COCO already on Drive (box ranking is the slow step).

**Before running:** Runtime → GPU.
"""))

cells.append(md("## [0] Setup"))

cells.append(code(r'''
import os, sys, json, random, shutil, zipfile, subprocess, copy
from collections import defaultdict

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

print("CUDA:", torch.cuda.is_available(), "| GPU:",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")

from google.colab import drive
drive.mount('/content/drive')

PROJECT_DIR = '/content/drive/MyDrive/viddar_samed'
V2_DIR = f'{PROJECT_DIR}/cross_dataset_v2'
SYNTH_DIR = f'{V2_DIR}/synthetic_coco_balanced'  # reuse v2 images
COMPARE_DIR = f'{PROJECT_DIR}/cross_dataset_v3'
os.makedirs(COMPARE_DIR, exist_ok=True)

assert os.path.exists(f'{SYNTH_DIR}/labels.csv'), (
    "v2 synthetic dataset not found. Run viddar_cross_dataset_comparison2.ipynb Section [1] first, "
    f"or check {SYNTH_DIR}")

shutil.rmtree(f'{PROJECT_DIR}/hf_cache', ignore_errors=True)
os.environ['HF_HOME'] = '/content/hf_cache'
os.makedirs(os.environ['HF_HOME'], exist_ok=True)

if not os.path.exists('/content/segment-anything'):
    subprocess.run(['git', 'clone', '-q',
                    'https://github.com/facebookresearch/segment-anything.git'], cwd='/content')
if not os.path.exists('/content/SAMed'):
    subprocess.run(['git', 'clone', '-q', 'https://github.com/hitachinsk/SAMed.git'], cwd='/content')
    subprocess.run(['pip', 'install', '-q', 'icecream', 'SimpleITK', 'medpy',
                    'tensorboardX', 'ml-collections'])

SAM_CKPT = f'{PROJECT_DIR}/checkpoints/sam_vit_b_01ec64.pth'
os.makedirs(f'{PROJECT_DIR}/checkpoints', exist_ok=True)
if not os.path.exists(SAM_CKPT):
    subprocess.run(['wget', '-q',
                    'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
                    '-O', SAM_CKPT])
os.makedirs('/content/SAMed/checkpoints', exist_ok=True)
shutil.copy(SAM_CKPT, '/content/SAMed/checkpoints/sam_vit_b_01ec64.pth')

sys.path.insert(0, '/content/SAMed')
sys.path.insert(0, '/content/segment-anything')

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
    "IDEA-Research/grounding-dino-tiny").to("cuda").eval()
print("Grounding DINO loaded.")

for mod_name in list(sys.modules.keys()):
    if mod_name == 'segment_anything' or mod_name.startswith('segment_anything.'):
        del sys.modules[mod_name]
sys.path.insert(0, '/content/segment-anything')
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

sam = sam_model_registry['vit_b'](checkpoint='/content/SAMed/checkpoints/sam_vit_b_01ec64.pth')
sam.cuda().eval()
print("SAM loaded.")


def compute_iou_dice(pred, target, eps=1e-6):
    pred, target = pred.astype(bool), target.astype(bool)
    inter = (pred & target).sum()
    union = (pred | target).sum()
    return float((inter + eps) / (union + eps)), float((2 * inter + eps) / (pred.sum() + target.sum() + eps))


DINO_SYNONYMS = {
    'car': ['car', 'automobile', 'vehicle'],
    'truck': ['truck', 'lorry', 'vehicle'],
    'bus': ['bus', 'vehicle'],
    'motorcycle': ['motorcycle', 'motorbike', 'bike'],
    'tv': ['tv', 'television', 'tv screen', 'monitor'],
    'stop sign': ['stop sign', 'stop'],
    'fire hydrant': ['fire hydrant', 'hydrant'],
    'traffic light': ['traffic light', 'traffic lights', 'stoplight'],
    'knife': ['knife', 'kitchen knife'],
    'scissors': ['scissors', 'pair of scissors'],
    'toaster': ['toaster'],
    'vehicle': ['car', 'truck', 'bus', 'motorcycle', 'vehicle'],
}


def dino_detect_all(image_pil, query, box_threshold=0.20, text_threshold=0.15, max_per_prompt=5):
    """Return list of (box_xyxy np, dino_score, prompt) across synonyms."""
    prompts = list(DINO_SYNONYMS.get(query.lower(), [query]))
    if query.lower() not in {p.lower() for p in prompts}:
        prompts = [query] + prompts
    # For vehicle-facing labels, always include all vehicle subtypes
    if query.lower() in ('vehicle', 'car', 'truck', 'bus', 'motorcycle'):
        for p in DINO_SYNONYMS['vehicle']:
            if p not in prompts:
                prompts.append(p)

    detections = []
    for prompt in prompts:
        inputs = dino_processor(images=image_pil, text=[[prompt.lower()]], return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = dino_model(**inputs)
        results = dino_processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=box_threshold, text_threshold=text_threshold,
            target_sizes=[image_pil.size[::-1]])[0]
        if len(results['boxes']) == 0:
            continue
        scores = results['scores'].cpu().numpy()
        boxes = results['boxes'].cpu().numpy()
        order = np.argsort(-scores)[:max_per_prompt]
        for i in order:
            detections.append((boxes[i], float(scores[i]), prompt))
    # dedupe near-duplicate boxes (IoU > 0.85 keep higher score)
    kept = []
    for box, sc, pr in sorted(detections, key=lambda x: -x[1]):
        x1, y1, x2, y2 = box
        drop = False
        for b2, s2, _ in kept:
            xx1, yy1 = max(x1, b2[0]), max(y1, b2[1])
            xx2, yy2 = min(x2, b2[2]), min(y2, b2[3])
            inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
            a1 = max(0, x2 - x1) * max(0, y2 - y1)
            a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
            iou = inter / (a1 + a2 - inter + 1e-6)
            if iou > 0.85:
                drop = True
                break
        if not drop:
            kept.append((box, sc, pr))
    return kept


OPTION_A_CKPT = f'{PROJECT_DIR}/checkpoints/stat_v2_rep1/epoch_79.pth'
HAS_OPTION_A = os.path.exists(OPTION_A_CKPT)
net_final = None
if HAS_OPTION_A:
    for mod_name in list(sys.modules.keys()):
        if mod_name == 'segment_anything' or mod_name.startswith('segment_anything.'):
            del sys.modules[mod_name]
    sys.path.insert(0, '/content/SAMed')
    from sam_lora_image_encoder import LoRA_Sam
    from segment_anything import sam_model_registry as sam_reg_samed
    args_a_img = 224
    sam_a, _ = sam_reg_samed['vit_b'](
        image_size=args_a_img, num_classes=2,
        checkpoint='/content/SAMed/checkpoints/sam_vit_b_01ec64.pth',
        pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1])
    net_final = LoRA_Sam(sam_a, 4).cuda()
    net_final.load_lora_parameters(OPTION_A_CKPT)
    net_final.eval()
    # restore original SAM on path for box methods
    for mod_name in list(sys.modules.keys()):
        if mod_name == 'segment_anything' or mod_name.startswith('segment_anything.'):
            del sys.modules[mod_name]
    sys.path.insert(0, '/content/segment-anything')
    from segment_anything import sam_model_registry
    from segment_anything.utils.transforms import ResizeLongestSide
    sam = sam_model_registry['vit_b'](checkpoint='/content/SAMed/checkpoints/sam_vit_b_01ec64.pth')
    sam.cuda().eval()
    print("Option A loaded.")
else:
    print("Option A checkpoint missing — will skip A.")

print("\n=== Setup complete ===")
'''))

cells.append(md("""## [1] Load v2 balanced dataset

Same 321 images / splits as v2 so IoU differences are due to the **box-ranking method**, not a new sample draw.
"""))

cells.append(code(r'''
labels_synth = pd.read_csv(f'{SYNTH_DIR}/labels.csv')
with open(f'{SYNTH_DIR}/splits.json') as f:
    splits_synth = json.load(f)

RAW_DIR_S = f'{SYNTH_DIR}/raw_img'
KEY_DIR_S = f'{SYNTH_DIR}/key_object_mask'
AR_DIR_S = f'{SYNTH_DIR}/ar_img'
AR_MASK_DIR_S = f'{SYNTH_DIR}/ar_content_mask'

print(f"Loaded v2 dataset: {len(labels_synth)} | "
      f"train={len(splits_synth['train'])} val={len(splits_synth['val'])} test={len(splits_synth['test'])}")
print(labels_synth['Object Name'].value_counts())
'''))

cells.append(md("""## [2] Embeddings + two box strategies

- **v2-style:** single best DINO confidence across synonyms (baseline on this same data).
- **v3 SAM-ranked:** among candidate DINO boxes, pick the one whose SAM multimask has the highest **predicted** IoU (no ground truth used).
"""))

cells.append(code(r'''
SAM_IMG_SIZE = sam.image_encoder.img_size
resize_transform = ResizeLongestSide(SAM_IMG_SIZE)
TOP_K_BOXES = 8  # max candidates to score with SAM


def load_gt_mask_s(fname, shape=None):
    m = (cv2.imread(os.path.join(KEY_DIR_S, f'mask_{fname}'), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    if shape is not None and m.shape != tuple(shape):
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m


def gt_box_from_mask(gt_mask):
    ys, xs = np.where(gt_mask > 0)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def build_cache(file_list):
    cache = {}
    sam.eval()
    with torch.no_grad():
        for fname in tqdm(file_list, desc='embeddings'):
            bgr = cv2.imread(os.path.join(RAW_DIR_S, fname))
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            gt = load_gt_mask_s(fname, shape=image.shape[:2])
            inp = resize_transform.apply_image(image)
            h, w = inp.shape[:2]
            x = torch.as_tensor(inp, device='cuda').permute(2, 0, 1).contiguous()[None]
            emb = sam.image_encoder(sam.preprocess(x))
            cache[fname] = {
                'image': image,
                'emb': emb.half().cpu(),
                'input_size': (h, w),
                'original_size': tuple(image.shape[:2]),
                'gt_full': gt,
            }
    return cache


def sam_masks_for_box(sample, box, multimask=True):
    """Return list of (mask_uint8, predicted_iou) for one box."""
    box_t = resize_transform.apply_boxes(np.asarray(box, dtype=np.float32)[None, :], sample['original_size'])
    box_t = torch.as_tensor(box_t, dtype=torch.float, device='cuda')
    with torch.no_grad():
        sparse_emb, dense_emb = sam.prompt_encoder(points=None, boxes=box_t, masks=None)
        low_res, iou_pred = sam.mask_decoder(
            image_embeddings=sample['emb'].cuda().float(),
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=multimask,
        )
        up = sam.postprocess_masks(low_res, input_size=sample['input_size'],
                                   original_size=sample['original_size'])
        # up: [1, C, H, W], iou_pred: [1, C]
        masks = (up[0].cpu().numpy() > 0).astype(np.uint8)
        scores = iou_pred[0].cpu().numpy()
    out = []
    for i in range(masks.shape[0]):
        out.append((masks[i], float(scores[i])))
    return out


def select_v2_style(detections):
    """Highest DINO score only."""
    if not detections:
        return None, 0.0, None
    box, sc, pr = max(detections, key=lambda x: x[1])
    return box, sc, pr


def select_v3_sam_ranked(sample, detections, top_k=TOP_K_BOXES):
    """Score each candidate box with SAM multimask predicted IoU; return best box + mask."""
    if not detections:
        return None, None, 0.0, None, 0.0
    cands = sorted(detections, key=lambda x: -x[1])[:top_k]
    best = None  # (sam_iou, dino_sc, box, mask, prompt)
    for box, dino_sc, prompt in cands:
        for mask, sam_iou in sam_masks_for_box(sample, box, multimask=True):
            # tie-break with DINO score
            key = (sam_iou, dino_sc)
            if best is None or key > (best[0], best[1]):
                best = (sam_iou, dino_sc, box, mask, prompt)
    sam_iou, dino_sc, box, mask, prompt = best
    return box, mask, sam_iou, prompt, dino_sc


# Build caches for val+test (and train only if needed — skip train for speed)
eval_files = list(dict.fromkeys(splits_synth['val'] + splits_synth['test']))
cache = build_cache(eval_files)

# Precompute detections + both selections
box_cache_path = f'{COMPARE_DIR}/boxes_v3.json'
if os.path.exists(box_cache_path):
    with open(box_cache_path) as f:
        box_cache = json.load(f)
    print(f"Loaded box cache ({len(box_cache)} files).")
else:
    box_cache = {}
    for fname in tqdm(eval_files, desc='DINO + SAM-rank boxes'):
        row = labels_synth[labels_synth['File Name'] == fname].iloc[0]
        query = str(row['dino_query']) if 'dino_query' in row and pd.notna(row['dino_query']) else str(row['Object Name'])
        pil = Image.fromarray(cache[fname]['image'])
        dets = dino_detect_all(pil, query)
        # also try Object Name if different
        if str(row['Object Name']).lower() != query.lower():
            dets = dets + dino_detect_all(pil, str(row['Object Name']))
            # re-dedupe roughly by keeping all; select functions handle ranking

        box_v2, sc_v2, pr_v2 = select_v2_style(dets)
        box_v3, mask_v3, sam_iou, pr_v3, dino_sc = select_v3_sam_ranked(cache[fname], dets)

        box_cache[fname] = {
            'query': query,
            'n_dets': len(dets),
            'v2_box': box_v2.tolist() if box_v2 is not None else None,
            'v2_dino_score': float(sc_v2) if box_v2 is not None else 0.0,
            'v2_prompt': pr_v2,
            'v3_box': box_v3.tolist() if box_v3 is not None else None,
            'v3_sam_iou': float(sam_iou) if box_v3 is not None else 0.0,
            'v3_dino_score': float(dino_sc) if box_v3 is not None else 0.0,
            'v3_prompt': pr_v3,
            # store mask only as optional — regenerate at eval for simplicity
        }
    with open(box_cache_path, 'w') as f:
        json.dump(box_cache, f)
    print(f"Saved {box_cache_path}")

n_v2 = sum(1 for f in eval_files if box_cache[f]['v2_box'] is not None)
n_v3 = sum(1 for f in eval_files if box_cache[f]['v3_box'] is not None)
print(f"Box recall on val+test ({len(eval_files)}): v2-style {n_v2} | v3 SAM-ranked {n_v3}")

# How often v3 picks a different box than v2
diff = 0
for f in eval_files:
    b2, b3 = box_cache[f]['v2_box'], box_cache[f]['v3_box']
    if b2 is None or b3 is None:
        continue
    if not np.allclose(b2, b3, atol=1.0):
        diff += 1
print(f"v3 chose a different box than v2 on {diff} images "
      f"({diff/max(1,n_v2):.1%} of those with both boxes)")
'''))

cells.append(md("## [3] Test segmentation"))

cells.append(code(r'''
def predict_from_box(sample, box, multimask=True):
    masks = sam_masks_for_box(sample, box, multimask=multimask)
    mask, _ = max(masks, key=lambda x: x[1])
    return mask


def predict_option_a(fname):
    raw = Image.open(os.path.join(RAW_DIR_S, fname)).convert('RGB')
    t = torch.from_numpy(cv2.resize(np.array(raw), (224, 224)) / 255.0).permute(2, 0, 1).unsqueeze(0).float().cuda()
    with torch.no_grad():
        out = net_final(t, True, 224)
        logits = F.interpolate(out['low_res_logits'], size=raw.size[::-1], mode='bilinear', align_corners=False)
        return torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)


test_files = splits_synth['test']
oracle_boxes = {f: gt_box_from_mask(cache[f]['gt_full']) for f in test_files if f in cache}
test_common = [f for f in test_files
               if f in box_cache and box_cache[f]['v3_box'] is not None and oracle_boxes.get(f) is not None]
# for fair comparison with v2-style, also require v2 box
test_common = [f for f in test_common if box_cache[f]['v2_box'] is not None]
print(f"Test common (v2 box + v3 box + oracle): {len(test_common)}/{len(test_files)}")

METHODS = {
    'ZS_V2': ('ZS DINO v2-style (best DINO score)',
              lambda f: predict_from_box(cache[f], box_cache[f]['v2_box'], multimask=True)),
    'ZS_V3': ('ZS DINO v3 (SAM-ranked box)',
              lambda f: predict_from_box(cache[f], box_cache[f]['v3_box'], multimask=True)),
    'ORACLE': ('SAM + oracle box',
               lambda f: predict_from_box(cache[f], oracle_boxes[f], multimask=True)),
}
if HAS_OPTION_A and net_final is not None:
    METHODS['A'] = ('Option A ViDDAR→synth', lambda f: predict_option_a(f))


def bootstrap_ci(values, n_boot=10000, seed=0):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


rows = []
for fname in tqdm(test_common, desc='Test segmentation v3'):
    gt = cache[fname]['gt_full']
    row = labels_synth[labels_synth['File Name'] == fname].iloc[0]
    rec = {'file': fname, 'class': row['Object Name'],
           'dino_query': row.get('dino_query', row['Object Name']),
           'v3_changed_box': int(not np.allclose(box_cache[fname]['v2_box'], box_cache[fname]['v3_box'], atol=1.0))}
    for key, (_, fn) in METHODS.items():
        pred = fn(fname)
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        rec[f'{key}_iou'], rec[f'{key}_dice'] = compute_iou_dice(pred, gt)
    rows.append(rec)

per_image = pd.DataFrame(rows)
per_image.to_csv(f'{COMPARE_DIR}/test_per_image_segmentation.csv', index=False)

summary_rows = []
for key, (name, _) in METHODS.items():
    ious = per_image[f'{key}_iou'].values
    lo, hi = bootstrap_ci(ious)
    # macro = mean of per-class means
    macro = per_image.groupby('class')[f'{key}_iou'].mean().mean()
    summary_rows.append({
        'método': name, 'n': len(ious),
        'IoU micro': ious.mean(), 'IoU macro': macro,
        'IoU IC95% inf': lo, 'IoU IC95% sup': hi,
        'Dice': per_image[f'{key}_dice'].mean(),
        'falhas (IoU<0.1)': int((ious < 0.1).sum()),
    })
summary = pd.DataFrame(summary_rows).set_index('método')
summary.to_csv(f'{COMPARE_DIR}/test_segmentation_summary.csv')
print(summary.round(4).to_string())

# Paired tests
for a, b in [('ZS_V3', 'ZS_V2'), ('ZS_V3', 'A'), ('ZS_V3', 'ORACLE')]:
    if a not in METHODS or b not in METHODS:
        continue
    diffs = per_image[f'{a}_iou'] - per_image[f'{b}_iou']
    lo, hi = bootstrap_ci(diffs)
    print(f"Paired {METHODS[a][0]} − {METHODS[b][0]}: {diffs.mean():+.4f} "
          f"(IC95% [{lo:+.4f}, {hi:+.4f}]"
          f"{' — significativa' if lo > 0 or hi < 0 else ' — inclui zero'})")

# Gain restricted to images where v3 changed the box
changed = per_image[per_image['v3_changed_box'] == 1]
if len(changed):
    d = changed['ZS_V3_iou'] - changed['ZS_V2_iou']
    print(f"\nOn {len(changed)} images where v3 ≠ v2 box: mean Δ IoU = {d.mean():+.4f} "
          f"(wins={(d>0.01).sum()}, losses={(d<-0.01).sum()}, ties={((d>=-0.01)&(d<=0.01)).sum()})")

per_class = per_image.groupby('class')[[f'{k}_iou' for k in METHODS]].mean()
per_class.columns = [METHODS[k][0] for k in METHODS]
per_class.insert(0, 'n', per_image.groupby('class').size())
per_class.to_csv(f'{COMPARE_DIR}/test_per_class_iou.csv')
print('\nIoU by class:')
print(per_class.round(3).to_string())

fig, ax = plt.subplots(figsize=(9, 4))
ax.boxplot([per_image[f'{k}_iou'] for k in METHODS], showmeans=True)
ax.set_xticks(range(1, len(METHODS) + 1))
ax.set_xticklabels([METHODS[k][0].replace(' (', '\n(').replace(' + ', '\n+ ') for k in METHODS], fontsize=7)
ax.set_ylabel('IoU'); ax.set_title(f'v3 test ({len(test_common)} images, same as v2 data)')
plt.tight_layout()
plt.savefig(f'{COMPARE_DIR}/fig_test_iou_boxplot.png', dpi=200, bbox_inches='tight')
plt.show()
'''))

cells.append(md("## [4] Obstruction detection"))

cells.append(code(r'''
DIFF_GRID = [10, 20, 30, 40, 50]
RATIO_GRID = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]


def overlap_ratio(key_mask, content_mask):
    if content_mask.shape != key_mask.shape:
        content_mask = cv2.resize(content_mask, (key_mask.shape[1], key_mask.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
    return (key_mask.astype(bool) & content_mask.astype(bool)).sum() / (key_mask.sum() + 1e-6)


def obstruction_metrics(gts, preds):
    tn, fp, fn, tp = confusion_matrix(gts, preds, labels=[False, True]).ravel()
    return {'acc': float((preds == gts).mean()),
            'precision': float(precision_score(gts, preds, zero_division=0)),
            'recall': float(recall_score(gts, preds, zero_division=0)),
            'f1': float(f1_score(gts, preds, zero_division=0)),
            'fnr': float(fn / (fn + tp) if (fn + tp) > 0 else 0.0),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}


def collect_obs(file_list, key_fns):
    ratios, gts = defaultdict(list), []
    for fname in tqdm(file_list, desc='Obstruction'):
        gts.append(labels_synth.loc[labels_synth['File Name'] == fname, 'obstruction'].iloc[0] == 'yes')
        raw = cv2.imread(os.path.join(RAW_DIR_S, fname))
        ar = cv2.imread(os.path.join(AR_DIR_S, f'AR_{fname}'))
        if ar.shape != raw.shape:
            ar = cv2.resize(ar, (raw.shape[1], raw.shape[0]))
        dg = cv2.cvtColor(cv2.absdiff(raw, ar), cv2.COLOR_BGR2GRAY)
        gt_c = (cv2.imread(os.path.join(AR_MASK_DIR_S, f'mask_{fname}'), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        for kn, kfn in key_fns.items():
            km = kfn(fname)
            for d in DIFF_GRID:
                ratios[(kn, f'diff{d}')].append(overlap_ratio(km, (dg > d).astype(np.uint8)))
            ratios[(kn, 'gt')].append(overlap_ratio(km, gt_c))
    return {k: np.array(v) for k, v in ratios.items()}, np.array(gts)


val_files = [f for f in splits_synth['val']
             if f in box_cache and box_cache[f]['v3_box'] is not None and f in cache]
key_fns = {
    'GT': lambda f: cache[f]['gt_full'],
    'ZS_V2': lambda f: predict_from_box(cache[f], box_cache[f]['v2_box']),
    'ZS_V3': lambda f: predict_from_box(cache[f], box_cache[f]['v3_box']),
}
if HAS_OPTION_A and net_final is not None:
    key_fns['A'] = predict_option_a

val_ratios, val_gts = collect_obs(val_files, key_fns)
test_ratios, test_gts = collect_obs(test_common, key_fns)


def tune(kn, content):
    diffs = DIFF_GRID if content == 'estimated' else [None]
    best = None
    for d in diffs:
        r = val_ratios[(kn, f'diff{d}' if d is not None else 'gt')]
        for t in RATIO_GRID:
            m = obstruction_metrics(val_gts, r >= t)
            score = (m['f1'], m['acc'])
            if best is None or score > best[0]:
                best = (score, d, t)
    return best[1], best[2]


results = []
for kn in key_fns:
    for how, content, d, t in [
        ('fixed (30, 0.25)', 'estimated', 30, 0.25),
        ('tuned on val', 'estimated', *tune(kn, 'estimated')),
        ('tuned on val', 'gt', *tune(kn, 'gt')),
    ]:
        r = test_ratios[(kn, f'diff{d}' if content == 'estimated' else 'gt')]
        m = obstruction_metrics(test_gts, r >= t)
        results.append({'key mask': kn, 'AR content': content, 'thresholds': how,
                        'diff_thr': d, 'ratio_thr': t, **m})

obs = pd.DataFrame(results)
obs.to_csv(f'{COMPARE_DIR}/obstruction_metrics.csv', index=False)
print(obs.round(4).to_string(index=False))
'''))

cells.append(md("## [5] Compare v1 / v2 / v3 / ViDDAR"))

cells.append(code(r'''
# Reference numbers from prior runs / paper
REF = pd.DataFrame([
    # method, dataset, IoU
    ('Option A', 'ViDDAR', 0.6863),
    ('ZS DINO', 'ViDDAR', 0.8686),
    ('Option B', 'ViDDAR', 0.8855),
    ('Oracle box', 'ViDDAR', 0.8945),
    ('Option A', 'Synth v1', 0.0813),
    ('ZS DINO', 'Synth v1', 0.5657),
    ('Option B transfer', 'Synth v1', 0.5481),
    ('Oracle box', 'Synth v1', 0.8582),
    ('Option A', 'Synth v2', 0.1537),
    ('ZS DINO', 'Synth v2', 0.6566),
    ('Option B transfer', 'Synth v2', 0.6257),
    ('Option B adapted', 'Synth v2', 0.6240),
    ('Oracle box', 'Synth v2', 0.8345),
], columns=['método', 'dataset', 'IoU'])

def iou_of(key):
    return float(per_image[f'{key}_iou'].mean())

cmp_v3 = pd.DataFrame([
    {'método': 'ZS DINO (v2-style queries)', 'Synth v3': iou_of('ZS_V2'),
     'Synth v2': 0.6566, 'Synth v1': 0.5657, 'ViDDAR': 0.8686},
    {'método': 'ZS DINO (v3 SAM-ranked)', 'Synth v3': iou_of('ZS_V3'),
     'Synth v2': np.nan, 'Synth v1': np.nan, 'ViDDAR': np.nan},
    {'método': 'Oracle box', 'Synth v3': iou_of('ORACLE'),
     'Synth v2': 0.8345, 'Synth v1': 0.8582, 'ViDDAR': 0.8945},
])
if 'A' in METHODS:
    cmp_v3 = pd.concat([cmp_v3, pd.DataFrame([{
        'método': 'Option A', 'Synth v3': iou_of('A'),
        'Synth v2': 0.1537, 'Synth v1': 0.0813, 'ViDDAR': 0.6863}])], ignore_index=True)

cmp_v3['Δ v3−v2 (ZS)'] = np.nan
# highlight main gain: SAM-ranked vs v2-style on same data
gain = iou_of('ZS_V3') - iou_of('ZS_V2')
print(f"=== Main result (same images) ===")
print(f"ZS v2-style IoU: {iou_of('ZS_V2'):.4f}")
print(f"ZS v3 SAM-ranked IoU: {iou_of('ZS_V3'):.4f}")
print(f"Δ (v3 − v2-style): {gain:+.4f}")
print(f"Oracle: {iou_of('ORACLE'):.4f} | gap to oracle v3: {iou_of('ORACLE')-iou_of('ZS_V3'):+.4f}")

cmp_v3.to_csv(f'{COMPARE_DIR}/comparison_v1_v2_v3.csv', index=False)
print('\n', cmp_v3.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4))
labels = ['ZS v2-style\n(on v2 data)', 'ZS v3\nSAM-ranked', 'Oracle']
vals = [iou_of('ZS_V2'), iou_of('ZS_V3'), iou_of('ORACLE')]
colors = ['#9ca3af', '#dc2626', '#16a34a']
ax.bar(labels, vals, color=colors)
ax.set_ylim(0, 1); ax.set_ylabel('IoU')
ax.set_title(f'v3 box ranking vs v2-style (n={len(test_common)})')
for i, v in enumerate(vals):
    ax.text(i, v + 0.02, f'{v:.3f}', ha='center')
plt.tight_layout()
plt.savefig(f'{COMPARE_DIR}/fig_v3_vs_v2style_iou.png', dpi=200, bbox_inches='tight')
plt.show()
print(f'\nOutputs: {COMPARE_DIR}')
'''))

cells.append(md("## [6] Qualitative — cases where v3 changed the box"))

cells.append(code(r'''
def overlay_mask(image_rgb, mask, color=(255, 140, 0), alpha=0.45):
    img = np.array(image_rgb).copy(); over = img.copy()
    over[mask.astype(bool)] = color
    return cv2.addWeighted(over, alpha, img, 1 - alpha, 0)


df = per_image.copy()
df['delta'] = df['ZS_V3_iou'] - df['ZS_V2_iou']
changed = df[df['v3_changed_box'] == 1].copy()
if len(changed) == 0:
    print("v3 never changed the box — qualitative falls back to largest deltas.")
    changed = df.copy()

gain_f = changed.sort_values('delta', ascending=False).iloc[0]['file']
loss_f = changed.sort_values('delta', ascending=True).iloc[0]['file']
# hard vehicle if any
veh = changed[changed['class'] == 'vehicle']
hard_f = veh.sort_values('ZS_V3_iou', ascending=True).iloc[0]['file'] if len(veh) else loss_f
selected = [gain_f, loss_f, hard_f]
row_labels = ['Largest v3 gain over v2-style', 'Largest v3 loss vs v2-style', 'Hard case']

fig, axes = plt.subplots(3, 5, figsize=(14, 9))
titles = ['Raw', 'GT key', 'v2-style mask', 'v3 SAM-ranked', 'Oracle']
for i, fname in enumerate(selected):
    raw = Image.fromarray(cache[fname]['image'])
    gt = cache[fname]['gt_full']
    panels = [
        np.array(raw),
        overlay_mask(raw, gt, (0, 200, 0)),
        overlay_mask(raw, predict_from_box(cache[fname], box_cache[fname]['v2_box']), (0, 120, 255)),
        overlay_mask(raw, predict_from_box(cache[fname], box_cache[fname]['v3_box']), (255, 140, 0)),
        overlay_mask(raw, predict_from_box(cache[fname], oracle_boxes[fname]), (180, 0, 180)),
    ]
    for j, p in enumerate(panels):
        axes[i, j].imshow(p); axes[i, j].axis('off')
        if i == 0:
            axes[i, j].set_title(titles[j], fontsize=10)
    info = df[df.file == fname].iloc[0]
    axes[i, 0].text(-0.1, 0.5,
                    f"{row_labels[i]}\n{fname}\nv2={info['ZS_V2_iou']:.2f} "
                    f"v3={info['ZS_V3_iou']:.2f}\noracle={info['ORACLE_iou']:.2f}",
                    transform=axes[i, 0].transAxes, fontsize=7, va='center', ha='right')
plt.tight_layout()
plt.savefig(f'{COMPARE_DIR}/fig_qualitative_v3.png', dpi=200, bbox_inches='tight')
plt.show()
'''))

nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "cells": cells,
}
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Wrote", OUT)
