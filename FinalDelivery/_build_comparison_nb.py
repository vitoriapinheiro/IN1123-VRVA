#!/usr/bin/env python3
"""Build viddar_cross_dataset_comparison.ipynb for Entrega4/ThirdRound."""
import json
from pathlib import Path

# Output notebook next to this script (FinalDelivery/)
OUT = Path(__file__).resolve().parent / "viddar_cross_dataset_comparison.ipynb"


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

cells.append(md("""# ViDDAR — Cross-dataset comparison (synthetic COCO)

Self-contained Colab notebook that:

1. Builds a **synthetic ViDDAR-style obstruction dataset** from COCO (raw + AR overlay + key-object mask + **ground-truth AR-content mask** + obstruction labels).
2. Evaluates the same methods as the ViDDAR delivery: Option A (prompt-free SAMed, if checkpoint exists), zero-shot SAM + DINO box, Option B (box-conditioned decoder from ViDDAR), SAM + oracle box.
3. Runs obstruction detection with **GT AR masks** (synthetic advantage) and with the estimated pixel-difference mask.
4. Writes comparison tables against the ViDDAR test numbers from Entrega 2.

**Runtime:** GPU (T4). Section [0] ~2 min; [1] download COCO val2017 once (~1 GB) then generate ~5–10 min; [2]–[4] ~15–30 min depending on `N_SYNTH`.

**Before running:** Runtime → Change runtime type → GPU.
"""))

cells.append(md("## [0] Setup (Drive, SAM, Grounding DINO, optional SAMed)"))

cells.append(code(r'''
import os, sys, json, random, shutil, zipfile, subprocess, copy
from collections import defaultdict
from pathlib import Path

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
COMPARE_DIR = f'{PROJECT_DIR}/cross_dataset'
SYNTH_DIR = f'{COMPARE_DIR}/synthetic_coco'
os.makedirs(COMPARE_DIR, exist_ok=True)
os.makedirs(f'{COMPARE_DIR}/checkpoints', exist_ok=True)
os.makedirs(SYNTH_DIR, exist_ok=True)

# HF cache on local disk (avoid Drive mmap / safetensors issues)
shutil.rmtree(f'{PROJECT_DIR}/hf_cache', ignore_errors=True)
os.environ['HF_HOME'] = '/content/hf_cache'
os.makedirs(os.environ['HF_HOME'], exist_ok=True)

# --- SAM repos + checkpoint ---
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
    print("Downloading SAM ViT-B (~375MB)...")
    subprocess.run(['wget', '-q',
                    'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
                    '-O', SAM_CKPT])
os.makedirs('/content/SAMed/checkpoints', exist_ok=True)
shutil.copy(SAM_CKPT, '/content/SAMed/checkpoints/sam_vit_b_01ec64.pth')

# Original SAM first on sys.path
sys.path.insert(0, '/content/SAMed')
sys.path.insert(0, '/content/segment-anything')

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
dino_model_id = "IDEA-Research/grounding-dino-tiny"
dino_processor = AutoProcessor.from_pretrained(dino_model_id)
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_id).to("cuda").eval()
print("Grounding DINO loaded.")

for mod_name in list(sys.modules.keys()):
    if mod_name == 'segment_anything' or mod_name.startswith('segment_anything.'):
        del sys.modules[mod_name]
sys.path.insert(0, '/content/segment-anything')
from segment_anything import sam_model_registry as sam_model_registry_orig, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide

sam_baseline = sam_model_registry_orig['vit_b'](
    checkpoint='/content/SAMed/checkpoints/sam_vit_b_01ec64.pth')
sam_baseline.cuda().eval()
print("SAM original loaded.")


def reload_original_sam():
    for mod_name in list(sys.modules.keys()):
        if mod_name == 'segment_anything' or mod_name.startswith('segment_anything.'):
            del sys.modules[mod_name]
    sys.path.insert(0, '/content/segment-anything')
    from segment_anything import sam_model_registry as _reg, SamPredictor as _Pred
    sam = _reg['vit_b'](checkpoint='/content/SAMed/checkpoints/sam_vit_b_01ec64.pth')
    sam.cuda()
    return sam, _Pred(sam)


def compute_iou_dice(pred, target, eps=1e-6):
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = (pred & target).sum()
    union = (pred | target).sum()
    iou = (inter + eps) / (union + eps)
    dice = (2 * inter + eps) / (pred.sum() + target.sum() + eps)
    return float(iou), float(dice)


def get_best_dino_box(image_pil, class_name, box_threshold=0.35, text_threshold=0.25):
    text_labels = [[class_name.lower()]]
    inputs = dino_processor(images=image_pil, text=text_labels, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = dino_model(**inputs)
    results = dino_processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=box_threshold, text_threshold=text_threshold,
        target_sizes=[image_pil.size[::-1]])[0]
    if len(results['boxes']) == 0:
        return None
    best_idx = results['scores'].argmax().item()
    return results['boxes'][best_idx].cpu().numpy()


# --- Optional SAMed (Option A) if LoRA checkpoint exists ---
OPTION_A_CKPT = f'{PROJECT_DIR}/checkpoints/stat_v2_rep1/epoch_79.pth'
HAS_OPTION_A = os.path.exists(OPTION_A_CKPT)
OPTION_B_CKPT = f'{PROJECT_DIR}/checkpoints/option_b_maskdecoder_v2/mask_decoder_best.pth'
HAS_OPTION_B = os.path.exists(OPTION_B_CKPT)
print(f"Option A checkpoint: {'FOUND' if HAS_OPTION_A else 'MISSING'} ({OPTION_A_CKPT})")
print(f"Option B checkpoint: {'FOUND' if HAS_OPTION_B else 'MISSING'} ({OPTION_B_CKPT})")

net_final = None
if HAS_OPTION_A:
    for mod_name in list(sys.modules.keys()):
        if mod_name == 'segment_anything' or mod_name.startswith('segment_anything.'):
            del sys.modules[mod_name]
    sys.path.insert(0, '/content/SAMed')
    from sam_lora_image_encoder import LoRA_Sam
    from segment_anything import sam_model_registry
    import argparse
    args_a = argparse.Namespace(
        vit_name='vit_b', num_classes=2, img_size=224, rank=4,
        ckpt='/content/SAMed/checkpoints/sam_vit_b_01ec64.pth')
    sam_a, _ = sam_model_registry[args_a.vit_name](
        image_size=args_a.img_size, num_classes=args_a.num_classes, checkpoint=args_a.ckpt,
        pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1])
    net_final = LoRA_Sam(sam_a, args_a.rank).cuda()
    net_final.load_lora_parameters(OPTION_A_CKPT)
    net_final.eval()
    multimask_output_final = True
    print("Option A (SAMed LoRA) loaded from ViDDAR training.")

sam_b = None
if HAS_OPTION_B:
    sam_b, _ = reload_original_sam()
    state = torch.load(OPTION_B_CKPT, map_location='cuda')
    sam_b.mask_decoder.load_state_dict(state)
    sam_b.eval()
    print("Option B (mask decoder) loaded from ViDDAR training.")

print("\n=== Setup complete ===")
'''))

cells.append(md("""## [1] Build synthetic ViDDAR-style dataset from COCO

Uses **COCO val2017** images whose categories overlap ViDDAR key-object classes.
For each sample we:

- keep the original image as **raw**;
- paste a semi-transparent AR panel (colored rectangle) either overlapping the key object (`obstruction=yes`) or elsewhere (`obstruction=no`);
- save the key-object instance mask and the **exact** AR-content mask (synthetic advantage over ViDDAR).

Classes without a COCO match (e.g. fire extinguisher, boiling pot) are skipped here; they remain ViDDAR-only.
"""))

cells.append(code(r'''
# --- Config ---
N_SYNTH = 400          # total synthetic samples (increase if you have time)
RATIO_OBSTRUCTED = 0.48  # ~same as ViDDAR (145/306)
SEED = 42
MIN_AREA_FRAC = 0.002    # skip tiny instances
MAX_AREA_FRAC = 0.35

# ViDDAR class name -> COCO category name(s)
VIDDAR_TO_COCO = {
    'stop sign': ['stop sign'],
    'fire hydrant': ['fire hydrant'],
    'traffic light': ['traffic light'],
    'knife': ['knife'],
    'scissors': ['scissors'],
    'toaster': ['toaster'],
    'TV': ['tv'],
    'vehicle': ['car', 'truck', 'bus', 'motorcycle'],
    'parking sign': [],          # no direct COCO class
    'warning sign': [],
    'pedestrian sign': [],
    'caution sign': [],
    'exit sign': [],
    'fire extinguisher': [],
    'electrical outlet': [],
    'electrical box': [],
    'electrical pot': [],
    'boiling pot': [],
    'coffee maker': [],
    'ceiling fan': [],
    'help station': [],
    'rules board': [],
}

RAW_OUT = f'{SYNTH_DIR}/raw_img'
AR_OUT = f'{SYNTH_DIR}/ar_img'
KEY_OUT = f'{SYNTH_DIR}/key_object_mask'
AR_MASK_OUT = f'{SYNTH_DIR}/ar_content_mask'
for d in (RAW_OUT, AR_OUT, KEY_OUT, AR_MASK_OUT):
    os.makedirs(d, exist_ok=True)

manifest_path = f'{SYNTH_DIR}/labels.csv'
meta_path = f'{SYNTH_DIR}/meta.json'
splits_path = f'{SYNTH_DIR}/splits.json'

if os.path.exists(manifest_path) and os.path.exists(splits_path):
    labels_synth = pd.read_csv(manifest_path)
    with open(splits_path) as f:
        splits_synth = json.load(f)
    print(f"Synthetic dataset already on Drive: {len(labels_synth)} samples "
          f"(train={len(splits_synth['train'])} val={len(splits_synth['val'])} "
          f"test={len(splits_synth['test'])}). Skipping generation.")
else:
    # Download COCO val2017 + annotations (cached under Drive)
    coco_root = f'{COMPARE_DIR}/coco'
    os.makedirs(coco_root, exist_ok=True)
    ann_zip = f'{coco_root}/annotations_trainval2017.zip'
    img_zip = f'{coco_root}/val2017.zip'
    ann_dir = f'{coco_root}/annotations'
    img_dir = f'{coco_root}/val2017'

    if not os.path.exists(f'{ann_dir}/instances_val2017.json'):
        if not os.path.exists(ann_zip):
            print("Downloading COCO annotations (~241MB)...")
            subprocess.run(['wget', '-q',
                            'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
                            '-O', ann_zip], check=True)
        print("Extracting annotations...")
        with zipfile.ZipFile(ann_zip) as z:
            z.extractall(coco_root)

    if not os.path.exists(img_dir) or len(os.listdir(img_dir)) < 100:
        if not os.path.exists(img_zip):
            print("Downloading COCO val2017 images (~1GB)...")
            subprocess.run(['wget', '-q',
                            'http://images.cocodataset.org/zips/val2017.zip',
                            '-O', img_zip], check=True)
        print("Extracting val2017...")
        with zipfile.ZipFile(img_zip) as z:
            z.extractall(coco_root)

    subprocess.run(['pip', 'install', '-q', 'pycocotools'], check=True)
    from pycocotools.coco import COCO
    from pycocotools import mask as mask_utils

    coco = COCO(f'{ann_dir}/instances_val2017.json')
    cat_name_to_id = {c['name']: c['id'] for c in coco.loadCats(coco.getCatIds())}

    usable = {v: [cat_name_to_id[c] for c in coco_names if c in cat_name_to_id]
              for v, coco_names in VIDDAR_TO_COCO.items()}
    usable = {k: v for k, v in usable.items() if v}
    print("Usable ViDDAR↔COCO classes:", list(usable.keys()))

    # Collect candidate (image_id, ann_id, viddar_class)
    rng = random.Random(SEED)
    candidates = []
    for v_cls, cat_ids in usable.items():
        ann_ids = coco.getAnnIds(catIds=cat_ids, iscrowd=False)
        for aid in ann_ids:
            ann = coco.loadAnns([aid])[0]
            img = coco.loadImgs(ann['image_id'])[0]
            area_frac = ann['area'] / (img['width'] * img['height'])
            if MIN_AREA_FRAC <= area_frac <= MAX_AREA_FRAC:
                candidates.append((ann['image_id'], aid, v_cls))
    rng.shuffle(candidates)
    print(f"Candidates after area filter: {len(candidates)}")

    def ann_to_mask(ann, h, w):
        rle = coco.annToRLE(ann)
        m = mask_utils.decode(rle)
        if m.ndim == 3:
            m = m.any(axis=2)
        return (m > 0).astype(np.uint8)

    def paste_ar_panel(img, key_mask, obstruct, rng):
        """Return (ar_bgr, ar_content_mask, obstruction_yes)."""
        h, w = img.shape[:2]
        ar = img.copy()
        content = np.zeros((h, w), dtype=np.uint8)
        color = tuple(int(x) for x in rng.integers(0, 256, size=3))
        alpha = float(rng.uniform(0.45, 0.75))

        ys, xs = np.where(key_mask > 0)
        if len(xs) == 0:
            return ar, content, False
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        bw, bh = max(8, x1 - x0 + 1), max(8, y1 - y0 + 1)

        if obstruct:
            # panel covering a controlled fraction of the key object
            target_ratio = float(rng.uniform(0.28, 0.70))
            pw = int(max(bw * 0.5, min(w, bw * np.sqrt(target_ratio) * rng.uniform(0.9, 1.4))))
            ph = int(max(bh * 0.5, min(h, bh * np.sqrt(target_ratio) * rng.uniform(0.9, 1.4))))
            cx = int(rng.integers(x0, max(x0 + 1, x1 + 1)))
            cy = int(rng.integers(y0, max(y0 + 1, y1 + 1)))
            px0 = int(np.clip(cx - pw // 2, 0, w - 1))
            py0 = int(np.clip(cy - ph // 2, 0, h - 1))
            px1 = int(np.clip(px0 + pw, 0, w))
            py1 = int(np.clip(py0 + ph, 0, h))
        else:
            # panel away from key object (try a few times)
            pw = int(rng.integers(max(20, w // 10), max(21, w // 3) + 1))
            ph = int(rng.integers(max(20, h // 10), max(21, h // 3) + 1))
            for _ in range(20):
                px0 = int(rng.integers(0, max(1, w - pw + 1)))
                py0 = int(rng.integers(0, max(1, h - ph + 1)))
                px1, py1 = px0 + pw, py0 + ph
                patch = np.zeros_like(key_mask)
                patch[py0:py1, px0:px1] = 1
                overlap = (patch & key_mask).sum() / (key_mask.sum() + 1e-6)
                if overlap < 0.08:
                    break
            else:
                px0 = py0 = 0
                px1, py1 = min(pw, w), min(ph, h)

        content[py0:py1, px0:px1] = 1
        overlay = ar.copy()
        overlay[py0:py1, px0:px1] = color
        ar = cv2.addWeighted(overlay, alpha, ar, 1 - alpha, 0)

        ratio = (content & key_mask).sum() / (key_mask.sum() + 1e-6)
        return ar, content, bool(ratio >= 0.25)

    rows = []
    used_imgs = set()
    n_yes = int(N_SYNTH * RATIO_OBSTRUCTED)
    n_no = N_SYNTH - n_yes
    counts = {'yes': 0, 'no': 0}
    targets = {'yes': n_yes, 'no': n_no}

    for img_id, aid, v_cls in tqdm(candidates, desc='Generating synthetic samples'):
        if counts['yes'] >= targets['yes'] and counts['no'] >= targets['no']:
            break
        if img_id in used_imgs:
            continue
        img_info = coco.loadImgs([img_id])[0]
        img_path = os.path.join(img_dir, img_info['file_name'])
        bgr = cv2.imread(img_path)
        if bgr is None:
            continue
        ann = coco.loadAnns([aid])[0]
        key = ann_to_mask(ann, img_info['height'], img_info['width'])
        if key.shape[:2] != bgr.shape[:2]:
            key = cv2.resize(key, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

        want_yes = counts['yes'] < targets['yes']
        want_no = counts['no'] < targets['no']
        if want_yes and (not want_no or rng.random() < 0.5):
            obstruct_intent = True
        elif want_no:
            obstruct_intent = False
        else:
            continue

        ar, content, is_obs = paste_ar_panel(bgr, key, obstruct_intent, np.random.default_rng(rng.randint(0, 10**9)))
        label = 'yes' if is_obs else 'no'
        if counts[label] >= targets[label]:
            # flip intent once if we accidentally filled the other bucket
            ar, content, is_obs = paste_ar_panel(bgr, key, not obstruct_intent,
                                                 np.random.default_rng(rng.randint(0, 10**9)))
            label = 'yes' if is_obs else 'no'
            if counts[label] >= targets[label]:
                continue

        idx = counts['yes'] + counts['no']
        fname = f"{v_cls.replace(' ', '-')}_{idx:04d}.png"
        cv2.imwrite(os.path.join(RAW_OUT, fname), bgr)
        cv2.imwrite(os.path.join(AR_OUT, f'AR_{fname}'), ar)
        cv2.imwrite(os.path.join(KEY_OUT, f'mask_{fname}'), key * 255)
        cv2.imwrite(os.path.join(AR_MASK_OUT, f'mask_{fname}'), content * 255)
        rows.append({'File Name': fname, 'Object Name': v_cls, 'obstruction': label,
                     'coco_image_id': img_id, 'coco_ann_id': aid})
        counts[label] += 1
        used_imgs.add(img_id)

    labels_synth = pd.DataFrame(rows)
    labels_synth.to_csv(manifest_path, index=False)

    # stratified split 70/15/15 by class
    rng.seed(SEED)
    by_class = defaultdict(list)
    for _, r in labels_synth.iterrows():
        by_class[r['Object Name']].append(r['File Name'])
    train, val, test = [], [], []
    for cls, files in by_class.items():
        rng.shuffle(files)
        n = len(files)
        n_tr = max(1, int(0.70 * n))
        n_va = max(1, int(0.15 * n)) if n >= 3 else 0
        train += files[:n_tr]
        val += files[n_tr:n_tr + n_va]
        test += files[n_tr + n_va:]
    splits_synth = {'train': train, 'val': val, 'test': test}
    with open(splits_path, 'w') as f:
        json.dump(splits_synth, f, indent=2)
    with open(meta_path, 'w') as f:
        json.dump({'n': len(labels_synth), 'counts': counts, 'usable_classes': list(usable.keys()),
                   'seed': SEED, 'source': 'COCO val2017'}, f, indent=2)
    print(f"Generated {len(labels_synth)} samples | obstruction yes={counts['yes']} no={counts['no']}")
    print(f"Split: train={len(train)} val={len(val)} test={len(test)}")
    print(labels_synth['Object Name'].value_counts())
'''))

cells.append(md("""## [2] Segmentation evaluation on synthetic test set

Same protocol as ViDDAR Entrega 2: all methods on the **same images**, original resolution, bootstrap CIs.
Methods depend on available checkpoints (zero-shot always; A/B if found on Drive).
"""))

cells.append(code(r'''
SAM_IMG_SIZE = sam_baseline.image_encoder.img_size
LOW_RES_SAM = SAM_IMG_SIZE // 4
resize_transform = ResizeLongestSide(SAM_IMG_SIZE)

labels_synth = pd.read_csv(f'{SYNTH_DIR}/labels.csv')
with open(f'{SYNTH_DIR}/splits.json') as f:
    splits_synth = json.load(f)

RAW_DIR_S = f'{SYNTH_DIR}/raw_img'
KEY_DIR_S = f'{SYNTH_DIR}/key_object_mask'
AR_DIR_S = f'{SYNTH_DIR}/ar_img'
AR_MASK_DIR_S = f'{SYNTH_DIR}/ar_content_mask'

test_files_s = list(splits_synth['test'])
print(f"Synthetic test files: {len(test_files_s)}")


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


def precompute_boxes_s(file_list, cache_path):
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    boxes = {}
    for fname in tqdm(file_list, desc='DINO boxes (synthetic)'):
        row = labels_synth[labels_synth['File Name'] == fname].iloc[0]
        pil = Image.open(os.path.join(RAW_DIR_S, fname)).convert('RGB')
        box = get_best_dino_box(pil, row['Object Name'])
        boxes[fname] = box.tolist() if box is not None else None
    with open(cache_path, 'w') as f:
        json.dump(boxes, f)
    return boxes


def build_cache(sam, file_list):
    cache = {}
    sam.eval()
    with torch.no_grad():
        for fname in tqdm(file_list, desc='embeddings'):
            image = cv2.cvtColor(cv2.imread(os.path.join(RAW_DIR_S, fname)), cv2.COLOR_BGR2RGB)
            gt = load_gt_mask_s(fname, shape=image.shape[:2])
            inp = resize_transform.apply_image(image)
            h, w = inp.shape[:2]
            x = torch.as_tensor(inp, device='cuda').permute(2, 0, 1).contiguous()[None]
            emb = sam.image_encoder(sam.preprocess(x))
            cache[fname] = {
                'emb': emb.half().cpu(),
                'input_size': (h, w),
                'original_size': tuple(image.shape[:2]),
                'gt_full': gt,
            }
    return cache


def decode_box(sam, sample, box):
    box_t = resize_transform.apply_boxes(np.asarray(box, dtype=np.float32)[None, :], sample['original_size'])
    box_t = torch.as_tensor(box_t, dtype=torch.float, device='cuda')
    with torch.no_grad():
        sparse_emb, dense_emb = sam.prompt_encoder(points=None, boxes=box_t, masks=None)
        dense_pe = sam.prompt_encoder.get_dense_pe()
        low_res, _ = sam.mask_decoder(
            image_embeddings=sample['emb'].cuda().float(), image_pe=dense_pe,
            sparse_prompt_embeddings=sparse_emb, dense_prompt_embeddings=dense_emb,
            multimask_output=False)
    return low_res


def predict_mask_box(sam, sample, box):
    with torch.no_grad():
        low = decode_box(sam, sample, box)
        up = sam.postprocess_masks(low, input_size=sample['input_size'],
                                   original_size=sample['original_size'])
    return (up.squeeze().cpu().numpy() > 0).astype(np.uint8)


def predict_mask_option_a(fname):
    if net_final is None:
        raise RuntimeError('Option A not loaded')
    raw = Image.open(os.path.join(RAW_DIR_S, fname)).convert('RGB')
    t = torch.from_numpy(cv2.resize(np.array(raw), (224, 224)) / 255.0).permute(2, 0, 1).unsqueeze(0).float().cuda()
    with torch.no_grad():
        out = net_final(t, True, 224)
        logits = F.interpolate(out['low_res_logits'], size=raw.size[::-1], mode='bilinear', align_corners=False)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred


test_boxes_s = precompute_boxes_s(test_files_s, f'{COMPARE_DIR}/synth_test_boxes.json')
test_common = [f for f in test_files_s if test_boxes_s[f] is not None]
cache_test_s = build_cache(sam_baseline, test_files_s)
oracle_boxes_s = {f: gt_box_from_mask(cache_test_s[f]['gt_full']) for f in test_files_s}
test_common = [f for f in test_common if oracle_boxes_s[f] is not None]
print(f"Common test images with DINO+oracle box: {len(test_common)} / {len(test_files_s)}")

METHODS = {
    'ZS_DINO': ('SAM zero-shot + DINO box',
                lambda f: predict_mask_box(sam_baseline, cache_test_s[f], test_boxes_s[f])),
    'ORACLE': ('SAM zero-shot + oracle box',
               lambda f: predict_mask_box(sam_baseline, cache_test_s[f], oracle_boxes_s[f])),
}
if HAS_OPTION_B and sam_b is not None:
    METHODS['B'] = ('Option B (ViDDAR decoder → synthetic)',
                    lambda f: predict_mask_box(sam_b, cache_test_s[f], test_boxes_s[f]))
if HAS_OPTION_A and net_final is not None:
    METHODS['A'] = ('Option A (ViDDAR LoRA → synthetic)', lambda f: predict_mask_option_a(f))


def bootstrap_ci(values, n_boot=10000, seed=0):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


rows = []
for fname in tqdm(test_common, desc='Segmentation (synthetic test)'):
    gt = cache_test_s[fname]['gt_full']
    row = {'file': fname,
           'class': labels_synth.loc[labels_synth['File Name'] == fname, 'Object Name'].iloc[0]}
    for key, (_, fn) in METHODS.items():
        pred = fn(fname)
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        row[f'{key}_iou'], row[f'{key}_dice'] = compute_iou_dice(pred, gt)
    rows.append(row)

per_image_synth = pd.DataFrame(rows)
per_image_synth.to_csv(f'{COMPARE_DIR}/synth_test_per_image_segmentation.csv', index=False)

summary_rows = []
for key, (name, _) in METHODS.items():
    ious = per_image_synth[f'{key}_iou'].values
    lo, hi = bootstrap_ci(ious)
    summary_rows.append({
        'método': name, 'n': len(ious), 'IoU': ious.mean(),
        'IoU IC95% inf': lo, 'IoU IC95% sup': hi,
        'Dice': per_image_synth[f'{key}_dice'].mean(),
        'falhas (IoU<0.1)': int((ious < 0.1).sum()),
    })
summary_synth = pd.DataFrame(summary_rows).set_index('método')
summary_synth.to_csv(f'{COMPARE_DIR}/synth_test_segmentation_summary.csv')
print('\n=== Synthetic test segmentation ===')
print(summary_synth.round(4).to_string())

if 'B' in METHODS and 'A' in METHODS:
    diffs = per_image_synth['B_iou'] - per_image_synth['A_iou']
    lo, hi = bootstrap_ci(diffs)
    print(f"\nPaired B−A: {diffs.mean():+.4f} (IC95% [{lo:+.4f}, {hi:+.4f}])")
if 'B' in METHODS:
    diffs = per_image_synth['B_iou'] - per_image_synth['ZS_DINO_iou']
    lo, hi = bootstrap_ci(diffs)
    print(f"Paired B−ZS_DINO: {diffs.mean():+.4f} (IC95% [{lo:+.4f}, {hi:+.4f}])")

per_class = per_image_synth.groupby('class')[[f'{k}_iou' for k in METHODS]].mean()
per_class.columns = [METHODS[k][0] for k in METHODS]
per_class.insert(0, 'n', per_image_synth.groupby('class').size())
per_class.to_csv(f'{COMPARE_DIR}/synth_test_per_class_iou.csv')
print('\nIoU by class:')
print(per_class.round(3).to_string())

fig, ax = plt.subplots(figsize=(max(7, 1.8 * len(METHODS)), 4))
ax.boxplot([per_image_synth[f'{k}_iou'] for k in METHODS], showmeans=True)
ax.set_xticks(range(1, len(METHODS) + 1))
ax.set_xticklabels([METHODS[k][0].replace(' + ', '\n+ ').replace(' (', '\n(') for k in METHODS], fontsize=8)
ax.set_ylabel('IoU'); ax.set_title(f'Synthetic COCO test ({len(test_common)} images)')
plt.tight_layout()
plt.savefig(f'{COMPARE_DIR}/fig_synth_iou_boxplot.png', dpi=200, bbox_inches='tight')
plt.show()
'''))

cells.append(md("""## [3] Obstruction detection (GT AR mask vs estimated diff)

On synthetic data we know the true AR content mask. This separates segmentation errors from AR-content estimation errors — the analysis that was skipped on ViDDAR when masks were not found.
"""))

cells.append(code(r'''
DIFF_GRID = [10, 20, 30, 40, 50]
RATIO_GRID = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
DEFAULT_DIFF, DEFAULT_RATIO = 30, 0.25


def overlap_ratio(key_mask, content_mask):
    if content_mask.shape != key_mask.shape:
        content_mask = cv2.resize(content_mask, (key_mask.shape[1], key_mask.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
    return (key_mask.astype(bool) & content_mask.astype(bool)).sum() / (key_mask.sum() + 1e-6)


def obstruction_metrics(gts, preds):
    tn, fp, fn, tp = confusion_matrix(gts, preds, labels=[False, True]).ravel()
    return {
        'acc': float((preds == gts).mean()),
        'precision': float(precision_score(gts, preds, zero_division=0)),
        'recall': float(recall_score(gts, preds, zero_division=0)),
        'f1': float(f1_score(gts, preds, zero_division=0)),
        'fnr': float(fn / (fn + tp) if (fn + tp) > 0 else 0.0),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def collect_obs(file_list, key_fns):
    ratios = defaultdict(list)
    gts = []
    for fname in tqdm(file_list, desc='Obstruction ratios'):
        gts.append(labels_synth.loc[labels_synth['File Name'] == fname, 'obstruction'].iloc[0] == 'yes')
        raw = cv2.imread(os.path.join(RAW_DIR_S, fname))
        ar = cv2.imread(os.path.join(AR_DIR_S, f'AR_{fname}'))
        if ar.shape != raw.shape:
            ar = cv2.resize(ar, (raw.shape[1], raw.shape[0]))
        dg = cv2.cvtColor(cv2.absdiff(raw, ar), cv2.COLOR_BGR2GRAY)
        gt_c = (cv2.imread(os.path.join(AR_MASK_DIR_S, f'mask_{fname}'),
                           cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        for key_name, key_fn in key_fns.items():
            km = key_fn(fname)
            for d in DIFF_GRID:
                ratios[(key_name, f'diff{d}')].append(overlap_ratio(km, (dg > d).astype(np.uint8)))
            ratios[(key_name, 'gt')].append(overlap_ratio(km, gt_c))
    return {k: np.array(v) for k, v in ratios.items()}, np.array(gts)


val_boxes_s = precompute_boxes_s(splits_synth['val'], f'{COMPARE_DIR}/synth_val_boxes.json')
cache_val_s = build_cache(sam_baseline, splits_synth['val'])
val_common = [f for f in splits_synth['val']
              if val_boxes_s.get(f) is not None
              and gt_box_from_mask(cache_val_s[f]['gt_full']) is not None]
print(f"Validation images for threshold tuning: {len(val_common)}")


def key_fns_for(cache, boxes):
    fns = {
        'GT': lambda f: cache[f]['gt_full'],
        'ZS': lambda f: predict_mask_box(sam_baseline, cache[f], boxes[f]),
    }
    if HAS_OPTION_B and sam_b is not None:
        fns['B'] = lambda f: predict_mask_box(sam_b, cache[f], boxes[f])
    if HAS_OPTION_A and net_final is not None:
        fns['A'] = predict_mask_option_a
    return fns


key_fns_val = key_fns_for(cache_val_s, val_boxes_s)
key_fns_test = key_fns_for(cache_test_s, test_boxes_s)
val_ratios, val_gts = collect_obs(val_common, key_fns_val)
test_ratios, test_gts = collect_obs(test_common, key_fns_test)


def tune(val_ratios, val_gts, key_name, content):
    diffs = DIFF_GRID if content == 'estimated' else [None]
    best = None
    for d in diffs:
        r = val_ratios[(key_name, f'diff{d}' if d is not None else 'gt')]
        for t in RATIO_GRID:
            m = obstruction_metrics(val_gts, r >= t)
            score = (m['f1'], m['acc'])
            if best is None or score > best[0]:
                best = (score, d, t)
    return best[1], best[2]


KEY_LABELS = {'GT': 'key GT', 'ZS': 'SAM+DINO', 'A': 'Option A', 'B': 'Option B'}
results = []
for key_name in key_fns_test.keys():
    configs = [('fixed (30, 0.25)', 'estimated', DEFAULT_DIFF, DEFAULT_RATIO)]
    for content in ('estimated', 'gt'):
        d, t = tune(val_ratios, val_gts, key_name, content)
        configs.append(('tuned on val', content, d, t))
    for how, content, d, t in configs:
        r = test_ratios[(key_name, f'diff{d}' if content == 'estimated' else 'gt')]
        m = obstruction_metrics(test_gts, r >= t)
        results.append({
            'key mask': KEY_LABELS.get(key_name, key_name),
            'AR content': 'estimated (diff)' if content == 'estimated' else 'GT (synthetic)',
            'thresholds': how, 'diff_thr': d, 'ratio_thr': t, **m,
        })

obs_synth = pd.DataFrame(results)
obs_synth.to_csv(f'{COMPARE_DIR}/synth_obstruction_metrics.csv', index=False)
print(obs_synth.round(4).to_string(index=False))
'''))

cells.append(md("""## [4] Side-by-side comparison with ViDDAR Entrega 2 numbers

Loads ViDDAR CSVs from Drive if present; otherwise uses the published paper numbers from the SecondRound run.
"""))

cells.append(code(r'''
# ViDDAR reference numbers (SecondRound / paper Table)
VIDDAR_SEG = pd.DataFrame([
    {'dataset': 'ViDDAR', 'método': 'Option A (promptless)', 'IoU': 0.6863, 'Dice': 0.7874, 'n': 61},
    {'dataset': 'ViDDAR', 'método': 'SAM zero-shot + DINO box', 'IoU': 0.8686, 'Dice': 0.9114, 'n': 61},
    {'dataset': 'ViDDAR', 'método': 'Option B (DINO + decoder)', 'IoU': 0.8855, 'Dice': 0.9300, 'n': 61},
    {'dataset': 'ViDDAR', 'método': 'SAM zero-shot + oracle box', 'IoU': 0.8945, 'Dice': 0.9377, 'n': 61},
])

# Prefer CSVs written by Entrega 2 if available
viddar_csv = f'{PROJECT_DIR}/test_segmentation_summary.csv'
if os.path.exists(viddar_csv):
    v = pd.read_csv(viddar_csv)
    # normalize column names from previous notebook
    if 'método' in v.columns or 'Unnamed: 0' in v.columns:
        if 'Unnamed: 0' in v.columns:
            v = v.rename(columns={'Unnamed: 0': 'método'})
        v['dataset'] = 'ViDDAR'
        VIDDAR_SEG = v[['dataset', 'método', 'IoU', 'Dice', 'n']].copy()
        print(f"Loaded ViDDAR metrics from {viddar_csv}")
else:
    print("Using hardcoded ViDDAR SecondRound numbers (CSV not found on Drive).")

# Map synthetic summary names to comparable rows
name_map = {
    'Option A (ViDDAR LoRA → synthetic)': 'Option A (promptless)',
    'SAM zero-shot + DINO box': 'SAM zero-shot + DINO box',
    'Option B (ViDDAR decoder → synthetic)': 'Option B (DINO + decoder)',
    'SAM zero-shot + oracle box': 'SAM zero-shot + oracle box',
}

synth_rows = []
for metodo, row in summary_synth.iterrows():
    canon = name_map.get(metodo, metodo)
    synth_rows.append({'dataset': 'Synthetic COCO', 'método': canon,
                       'IoU': row['IoU'], 'Dice': row['Dice'], 'n': int(row['n'])})
SYNTH_SEG = pd.DataFrame(synth_rows)

comparison = pd.concat([VIDDAR_SEG, SYNTH_SEG], ignore_index=True)
pivot = comparison.pivot_table(index='método', columns='dataset', values='IoU', aggfunc='first')
if 'ViDDAR' in pivot.columns and 'Synthetic COCO' in pivot.columns:
    pivot['Δ IoU (synth − ViDDAR)'] = pivot['Synthetic COCO'] - pivot['ViDDAR']
pivot = pivot.reindex([
    'Option A (promptless)',
    'SAM zero-shot + DINO box',
    'Option B (DINO + decoder)',
    'SAM zero-shot + oracle box',
])
pivot.to_csv(f'{COMPARE_DIR}/comparison_viddar_vs_synth_iou.csv')
print('\n=== IoU: ViDDAR test vs Synthetic COCO test (models trained on ViDDAR) ===')
print(pivot.round(4).to_string())

# Obstruction quick compare (fixed thresholds, estimated AR)
VIDDAR_OBS = pd.DataFrame([
    {'dataset': 'ViDDAR', 'key': 'Option A', 'acc': 0.9016, 'f1': 0.8800, 'fnr': 0.1852},
    {'dataset': 'ViDDAR', 'key': 'Option B', 'acc': 0.9180, 'f1': 0.8980, 'fnr': 0.1852},
    {'dataset': 'ViDDAR', 'key': 'key GT', 'acc': 0.9180, 'f1': 0.8980, 'fnr': 0.1852},
])
obs_fixed = obs_synth[(obs_synth['thresholds'] == 'fixed (30, 0.25)') &
                      (obs_synth['AR content'] == 'estimated (diff)')].copy()
obs_fixed['dataset'] = 'Synthetic COCO'
obs_fixed = obs_fixed.rename(columns={'key mask': 'key'})[['dataset', 'key', 'acc', 'f1', 'fnr']]

obs_gt_ar = obs_synth[(obs_synth['thresholds'].str.startswith('tuned')) &
                      (obs_synth['AR content'] == 'GT (synthetic)')].copy()
print('\n=== Obstruction (synthetic) with GT AR content + tuned thresholds ===')
print(obs_gt_ar[['key mask', 'diff_thr', 'ratio_thr', 'acc', 'f1', 'fnr']].round(4).to_string(index=False))

print('\n=== Obstruction fixed thresholds (estimated AR): ViDDAR vs Synthetic ===')
print(pd.concat([VIDDAR_OBS, obs_fixed], ignore_index=True).round(4).to_string(index=False))

# Bar chart
fig, ax = plt.subplots(figsize=(9, 4))
methods_order = [m for m in pivot.index if m in pivot.index and not pivot.loc[m].isna().all()]
x = np.arange(len(methods_order))
w = 0.35
if 'ViDDAR' in pivot.columns:
    ax.bar(x - w/2, [pivot.loc[m, 'ViDDAR'] if pd.notna(pivot.loc[m, 'ViDDAR']) else 0
                     for m in methods_order], w, label='ViDDAR test', color='#2563eb')
if 'Synthetic COCO' in pivot.columns:
    ax.bar(x + w/2, [pivot.loc[m, 'Synthetic COCO'] if pd.notna(pivot.loc[m, 'Synthetic COCO']) else 0
                     for m in methods_order], w, label='Synthetic COCO test', color='#dc2626')
ax.set_xticks(x)
ax.set_xticklabels([m.replace(' + ', '\n+ ').replace(' (', '\n(') for m in methods_order], fontsize=8)
ax.set_ylabel('IoU'); ax.set_ylim(0, 1)
ax.set_title('Cross-dataset transfer: models trained on ViDDAR')
ax.legend()
plt.tight_layout()
plt.savefig(f'{COMPARE_DIR}/fig_viddar_vs_synth_iou.png', dpi=200, bbox_inches='tight')
plt.show()

print(f"\nAll outputs under: {COMPARE_DIR}")
print("  - synth_test_segmentation_summary.csv")
print("  - synth_obstruction_metrics.csv")
print("  - comparison_viddar_vs_synth_iou.csv")
print("  - fig_synth_iou_boxplot.png / fig_viddar_vs_synth_iou.png")
'''))

cells.append(md("""## [5] Qualitative examples (synthetic)

Three rows: largest Option B gain over A (if A exists), a DINO failure, and a strong case for box-prompted SAM.
"""))

cells.append(code(r'''
def overlay_mask(image_rgb, mask, color=(255, 140, 0), alpha=0.45):
    img = np.array(image_rgb).copy()
    over = img.copy()
    over[mask.astype(bool)] = color
    return cv2.addWeighted(over, alpha, img, 1 - alpha, 0)


case_df = per_image_synth.copy()
if 'A_iou' in case_df.columns and 'B_iou' in case_df.columns:
    case_df['gap'] = case_df['B_iou'] - case_df['A_iou']
    gain_f = case_df.sort_values('gap', ascending=False).iloc[0]['file']
    loss_f = case_df.sort_values('gap', ascending=True).iloc[0]['file']
elif 'B_iou' in case_df.columns:
    case_df['gap'] = case_df['B_iou'] - case_df['ZS_DINO_iou']
    gain_f = case_df.sort_values('gap', ascending=False).iloc[0]['file']
    loss_f = case_df.sort_values('ZS_DINO_iou', ascending=True).iloc[0]['file']
else:
    gain_f = case_df.sort_values('ZS_DINO_iou', ascending=False).iloc[0]['file']
    loss_f = case_df.sort_values('ZS_DINO_iou', ascending=True).iloc[0]['file']

case_df['zs'] = case_df['ZS_DINO_iou']
best_f = case_df.sort_values('zs', ascending=False).iloc[0]['file']
selected = [gain_f, loss_f, best_f]
labels_rows = ['Largest box-prompted gain', 'Hard / failure case', 'Strong zero-shot case']

ncols = 5 if 'B' in METHODS else 4
fig, axes = plt.subplots(len(selected), ncols, figsize=(3.2 * ncols, 3.0 * len(selected)))
col_titles = ['Raw', 'AR', 'GT key']
if 'A' in METHODS:
    col_titles.append('Option A')
if 'B' in METHODS:
    col_titles.append('Option B')
else:
    col_titles.append('SAM+DINO')

for i, fname in enumerate(selected):
    raw = Image.open(os.path.join(RAW_DIR_S, fname)).convert('RGB')
    ar = Image.open(os.path.join(AR_DIR_S, f'AR_{fname}')).convert('RGB')
    gt = cache_test_s[fname]['gt_full']
    panels = [np.array(raw), np.array(ar), overlay_mask(raw, gt, (0, 200, 0))]
    if 'A' in METHODS:
        panels.append(overlay_mask(raw, predict_mask_option_a(fname), (0, 120, 255)))
    if 'B' in METHODS:
        panels.append(overlay_mask(raw, predict_mask_box(sam_b, cache_test_s[fname], test_boxes_s[fname]),
                                   (255, 140, 0)))
    else:
        panels.append(overlay_mask(raw, predict_mask_box(sam_baseline, cache_test_s[fname], test_boxes_s[fname]),
                                   (255, 140, 0)))
    for j, panel in enumerate(panels):
        ax = axes[i, j] if len(selected) > 1 else axes[j]
        ax.imshow(panel); ax.axis('off')
        if i == 0:
            ax.set_title(col_titles[j], fontsize=10)
    left = axes[i, 0]
    info = case_df[case_df['file'] == fname].iloc[0]
    txt = f"{labels_rows[i]}\n{fname}\n"
    if 'A_iou' in info:
        txt += f"A={info['A_iou']:.2f} "
    if 'B_iou' in info:
        txt += f"B={info['B_iou']:.2f} "
    txt += f"ZS={info['ZS_DINO_iou']:.2f}"
    left.text(-0.12, 0.5, txt, transform=left.transAxes, fontsize=8, va='center', ha='right')

plt.tight_layout()
plt.savefig(f'{COMPARE_DIR}/fig_synth_qualitative.png', dpi=200, bbox_inches='tight')
plt.show()
print("Saved fig_synth_qualitative.png")
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