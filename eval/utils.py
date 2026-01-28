import os
import json
import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

from backbone.maskrcnn import get_maskrcnn_model
from data import build_coco_gt_json, build_coco_pred_list

import torch
from torchvision.transforms import Compose, ToTensor, Normalize

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# -------------------------
# Utilities
# -------------------------
def grad_norm(params):
    total = 0.0
    count = 0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.data.norm(2).item() ** 2)
            count += 1
    return math.sqrt(total) if count > 0 else 0.0

def _strip_parallel_state_dict(state_dict):
    """Remove 'module.' prefix if present (DataParallel checkpoint)."""
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    new_state = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.", "") if k.startswith("module.") else k
        new_state[new_k] = v
    return new_state

def load_checkpoint_weights(checkpoint_path, model, device=torch.device("cpu"), strict=False):
    """
    Load checkpoint into model. Accepts either:
      - a raw state_dict saved by torch.save(model.state_dict())
      - a checkpoint dict with 'model_state_dict' (your EarlyStopping format)
    Returns model (on device).
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    else:
        state = ckpt

    state = _strip_parallel_state_dict(state)
    model.load_state_dict(state, strict=strict)
    model.to(device)
    model.eval()
    return model

# -------------------------
# Prediction helpers
# -------------------------
def predict_image(model, pil_img, device, score_thresh=0.5):
    """
    Run model on a single PIL image (batch size 1).
    Returns dict with numpy arrays: boxes (Nx4), labels (N,), scores (N,), masks (N,H,W) or None.
    """
    model.eval()
    transform = ToTensor()
    img_t = transform(pil_img).to(device)

    with torch.no_grad():
        outputs = model([img_t])[0]

    # Extract tensors safely (may be on GPU)
    boxes_t  = outputs.get('boxes', torch.zeros((0, 4), device=device))
    labels_t = outputs.get('labels', torch.zeros((0,), dtype=torch.int64, device=device))
    scores_t = outputs.get('scores', torch.tensor([], device=device))
    masks_t  = outputs.get('masks', None)  # may be None or tensor [N,1,H,W]

    # compute keep indices as a torch index (on same device)
    if scores_t.numel() == 0:
        # no detections
        return {"boxes": np.zeros((0,4)), "labels": np.zeros((0,), dtype=int), "scores": np.zeros((0,)), "masks": None}

    keep_mask = scores_t >= score_thresh
    if keep_mask.sum().item() == 0:
        return {"boxes": np.zeros((0,4)), "labels": np.zeros((0,), dtype=int), "scores": np.zeros((0,)), "masks": None}

    keep_idx = keep_mask.nonzero(as_tuple=True)[0]  # torch tensor on same device

    # Index while still tensors (safe on GPU)
    boxes_sel = boxes_t[keep_idx]        # [K,4] tensor
    labels_sel = labels_t[keep_idx]      # [K] tensor
    scores_sel = scores_t[keep_idx]      # [K] tensor

    if masks_t is not None:
        # masks_t: [N,1,H,W] -> select, then squeeze channel dim
        masks_sel = masks_t[keep_idx].squeeze(1)  # [K, H, W]
    else:
        masks_sel = None

    # Move to CPU and convert to numpy
    boxes_np = boxes_sel.detach().cpu().numpy()
    labels_np = labels_sel.detach().cpu().numpy()
    scores_np = scores_sel.detach().cpu().numpy()
    masks_np = masks_sel.detach().cpu().numpy() if masks_sel is not None else None

    return {"boxes": boxes_np, "labels": labels_np, "scores": scores_np, "masks": masks_np}

def visualize_and_save(pil_img, preds, class_map=None, save_path=None, score_format="{:.2f}", alpha=0.4):
    """
    Overlay boxes and masks on image and save/show.
    preds: dict from predict_image
    class_map: dict mapping class token->id or id->name. We assume preds labels are ints.
    """
    img_arr = np.array(pil_img)
    H, W = img_arr.shape[:2]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img_arr)
    boxes = preds['boxes']
    labels = preds['labels']
    scores = preds['scores']
    masks = preds.get('masks', None)

    for i, (box, lbl, scr) in enumerate(zip(boxes, labels, scores)):
        xmin, ymin, xmax, ymax = box
        w = xmax - xmin
        h = ymax - ymin
        ax.add_patch(plt.Rectangle((xmin, ymin), w, h, fill=False, edgecolor='yellow', linewidth=2))
        # label text
        label_name = str(int(lbl))
        if class_map is not None:
            # class_map might map string tokens to ints or ints to names — handle both
            if isinstance(class_map, dict):
                # if keys are names and values ints, invert
                if all(isinstance(k, str) and isinstance(v, int) for k, v in class_map.items()):
                    inv = {v: k for k, v in class_map.items()}
                    label_name = inv.get(int(lbl), str(int(lbl)))
                else:
                    label_name = class_map.get(int(lbl), str(int(lbl)))
        ax.text(xmin, ymin - 5, f"{label_name}: {score_format.format(scr)}", fontsize=10, color='white', backgroundcolor='black')

        # overlay mask if present
        if masks is not None and i < masks.shape[0]:
            mask = masks[i] > 0.5
            if mask.sum() > 0:
                # create an RGB colored mask - here red channel
                colored_mask = np.zeros((H, W, 3), dtype=float)
                colored_mask[..., 0] = mask.astype(float)  # red
                ax.imshow(colored_mask, alpha=alpha, cmap=None, extent=(0, W, H, 0))
    ax.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    else:
        plt.show()

def crop_and_save_instances(pil_img, masks, out_dir, basename):
    """
    Save each instance crop masked to PNG to out_dir with basename.
    masks: (N,H,W) binary array
    """
    os.makedirs(out_dir, exist_ok=True)
    arr = np.array(pil_img)
    for i in range(masks.shape[0]):
        mask = masks[i].astype(bool)
        if mask.sum() == 0:
            continue
        masked = arr.copy()
        masked[~mask] = 0
        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        crop = masked[y0:y1+1, x0:x1+1]
        out_path = os.path.join(out_dir, f"{basename}_inst{i}.png")
        Image.fromarray(crop).save(out_path)

# -------------------------
# Inference collector
# -------------------------
def run_inference_on_folder(
    checkpoint_path,
    images_dir,
    output_dir,
    num_classes,
    score_thresh=0.5,
    class_map=None,
    use_pretrained=False,
    save_crops=False
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(output_dir, exist_ok=True)
    # instantiate model architecture (must match training)
    model = get_maskrcnn_model(num_classes=num_classes, pretrained=use_pretrained)
    # load weights
    model = load_checkpoint_weights(checkpoint_path, model, device=device, strict=False)

    # important: if you changed image_mean/std during training, set same here:
    # model.transform.image_mean = [..]
    # model.transform.image_std  = [..]

    img_files = [f for f in sorted(os.listdir(images_dir)) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    results = []
    for fn in img_files:
        path = os.path.join(images_dir, fn)
        pil = Image.open(path).convert("RGB")
        preds = predict_image(model, pil, device=device, score_thresh=score_thresh)

        # save visualization
        vis_path = os.path.join(output_dir, fn.replace('.', '_pred.'))
        # ensure extension .png
        vis_path = os.path.splitext(vis_path)[0] + ".png"
        visualize_and_save(pil, preds, class_map=class_map, save_path=vis_path)

        # optionally save crops
        if save_crops and preds.get('masks') is not None:
            crop_dir = os.path.join(output_dir, "crops")
            crop_and_save_instances(pil, preds['masks'], crop_dir, os.path.splitext(fn)[0])

        # collect numeric results (for CSV/JSON)
        results.append({
            "image": fn,
            "boxes": preds['boxes'].tolist(),
            "labels": preds['labels'].tolist(),
            "scores": preds['scores'].tolist()
        })

    # save results json
    with open(os.path.join(output_dir, "predictions.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"Inference finished. Visualizations + predictions saved to {output_dir}")

def run_inference_and_collect(model, dataset, device, score_thresh=0.0):
    """
    Runs inference on dataset (iterable returning (img_tensor, target) or use dataset.image_files).
    Returns:
      preds_per_image: dict image_id -> {boxes: Nx4 xyxy, labels: N, scores: N, masks: N,H,W binary}
      gt_per_image: dict image_id -> list of gt ann dicts (category_id, bbox xywh, mask HxW binary)
      image_info_list: list of image dicts for COCO 'images' (id, file_name, width, height)
    """
    transform = ToTensor()
    preds_per_image = {}
    image_info_list = []
    gt_per_image = {}

    # Use dataset.image_files list if available
    for idx in tqdm(range(len(dataset)), desc="Running inference"):
        img_name = dataset.image_files[idx]
        img_path = os.path.join(dataset.img_dir, img_name) if hasattr(dataset, "img_dir") else img_name
        pil = Image.open(img_path).convert("RGB")
        W, H = pil.size
        image_info_list.append({"id": idx, "file_name": img_name, "width": W, "height": H})

        # --- ground truth from dataset (use dataset parsing if available) ---
        # Try to re-use dataset internals to build GT masks/boxes/labels
        # We'll call dataset._parse_label_file if present, else fallback to dataset[idx]
        gt_anns = []
        if hasattr(dataset, "_parse_label_file"):
            lbl_path = os.path.join(dataset.lbl_dir, os.path.splitext(img_name)[0] + ".txt")
            boxes, labels, masks = dataset._parse_label_file(lbl_path, W, H)
            for b, lbl, mask in zip(boxes, labels, masks):
                # convert boxes from [xmin,ymin,xmax,ymax] -> xywh
                x1, y1, x2, y2 = b
                gt_anns.append({
                    "category_id": int(lbl),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "mask": mask.astype(np.uint8)
                })
        else:
            # fallback: try dataset[idx] to obtain target with masks
            try:
                img_t, target = dataset[idx]
                # target["boxes"], target["labels"], target["masks"]
                boxes = target.get("boxes", torch.zeros((0,4))).numpy()
                labels = target.get("labels", torch.zeros((0,))).numpy()
                masks = target.get("masks", np.zeros((0, H, W)))
                for b, lbl, mask in zip(boxes, labels, masks):
                    x1, y1, x2, y2 = b
                    gt_anns.append({
                        "category_id": int(lbl),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "mask": mask.astype(np.uint8)
                    })
            except Exception:
                gt_anns = []
        gt_per_image[idx] = gt_anns

        # --- prediction ---
        img_t = transform(pil).to(device)
        with torch.no_grad():
            out = model([img_t])[0]

        # outputs are tensors on device; do indexing on tensors then convert
        scores_t = out.get("scores", torch.tensor([], device=device))
        if scores_t.numel() == 0:
            preds_per_image[idx] = {"boxes": np.zeros((0,4)), "labels": np.zeros((0,), dtype=int), "scores": np.zeros((0,)), "masks": None}
            continue

        keep_mask = scores_t >= score_thresh
        if keep_mask.sum().item() == 0:
            preds_per_image[idx] = {"boxes": np.zeros((0,4)), "labels": np.zeros((0,), dtype=int), "scores": np.zeros((0,)), "masks": None}
            continue

        keep_idx = keep_mask.nonzero(as_tuple=True)[0]

        boxes_t = out.get("boxes", torch.zeros((0,4), device=device))[keep_idx]
        labels_t = out.get("labels", torch.zeros((0,), dtype=torch.int64, device=device))[keep_idx]
        scores_sel = scores_t[keep_idx]
        masks_t = out.get("masks", None)
        if masks_t is not None:
            masks_sel = masks_t[keep_idx].squeeze(1)  # [K,H,W]
            masks_np = masks_sel.detach().cpu().numpy()
            # threshold to binary
            masks_np = (masks_np > 0.5).astype(np.uint8)
        else:
            masks_np = None

        boxes_np = boxes_t.detach().cpu().numpy()
        labels_np = labels_t.detach().cpu().numpy()
        scores_np = scores_sel.detach().cpu().numpy()

        preds_per_image[idx] = {"boxes": boxes_np, "labels": labels_np, "scores": scores_np, "masks": masks_np}

    return preds_per_image, gt_per_image, image_info_list

# -------------------------
# Run COCO eval
# -------------------------
def run_coco_eval(gt_json_path, pred_json_path, iou_type='bbox'):
    coco_gt = COCO(gt_json_path)
    coco_dt = coco_gt.loadRes(pred_json_path)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    stats = coco_eval.stats  # numpy array
    return {
        "mAP_50_95": float(stats[0]),
        "mAP_50": float(stats[1]),
        "mAP_75": float(stats[2])
    }

# -------------------------
# Greedy precision/recall at IoU threshold
# -------------------------
from torchvision.ops import box_iou
def iou_mask_np(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return (inter / union) if union > 0 else 0.0

def compute_pr_at_threshold(gt_per_image, preds_per_image, iou_thresh=0.5, score_thresh=0.5, iou_type='bbox'):
    TP = 0; FP = 0; FN = 0
    for img_id, gts in gt_per_image.items():
        preds = preds_per_image.get(img_id, {"boxes": np.zeros((0,4)), "labels": np.zeros((0,), dtype=int), "scores": np.zeros((0,)), "masks": None})
        # filter preds by score
        keep = preds["scores"] >= score_thresh
        boxes_p = preds["boxes"][keep]
        labels_p = preds["labels"][keep]
        masks_p = preds["masks"][keep] if preds["masks"] is not None else None

        matched_gt = [False] * len(gts)
        # for each pred, find best matching GT of same class
        for j in range(len(boxes_p)):
            p_cat = int(labels_p[j])
            best_iou = 0.0; best_idx = -1
            for k, g in enumerate(gts):
                if matched_gt[k] or int(g["category_id"]) != p_cat:
                    continue
                if iou_type == 'bbox':
                    px1, py1, px2, py2 = boxes_p[j]
                    gx, gy, gw, gh = g["bbox"]
                    gx2 = gx + gw; gy2 = gy + gh
                    p_box = torch.tensor([px1, py1, px2, py2]).unsqueeze(0)
                    g_box = torch.tensor([gx, gy, gx2, gy2]).unsqueeze(0)
                    iou = float(box_iou(p_box, g_box).item())
                else:
                    iou = iou_mask_np(masks_p[j], g["mask"])
                if iou > best_iou:
                    best_iou = iou; best_idx = k
            if best_iou >= iou_thresh and best_idx >= 0:
                TP += 1
                matched_gt[best_idx] = True
            else:
                FP += 1
        # remaining unmatched GT are FN
        FN += sum(1 for mk in matched_gt if not mk)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    return precision, recall, TP, FP, FN

# -------------------------
# Main evaluation orchestration
# -------------------------
def evaluate_model_on_dataset(
    checkpoint_path,
    dataset,
    output_dir,
    device=None,
    num_classes=2,
    score_thresh_inference=0.0,
    score_thresh_pr=0.5,
    iou_thresh_pr=0.5,
    class_map = None
):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    os.makedirs(output_dir, exist_ok=True)

    model = get_maskrcnn_model(num_classes, pretrained = False)
    model = load_checkpoint_weights(checkpoint_path, model=model, device=device)

    preds_per_image, gt_per_image, image_info_list = run_inference_and_collect(model, dataset, device, score_thresh=score_thresh_inference)

    # build coco GT and save
    coco_gt_dict = build_coco_gt_json(image_info_list, gt_per_image, getattr(dataset, "class_map", {1: "1"}))
    gt_json_path = os.path.join(output_dir, "gt_coco.json")
    with open(gt_json_path, "w") as f:
        json.dump(coco_gt_dict, f)

    # build preds lists and save
    preds_bbox_list, preds_segm_list = build_coco_pred_list(preds_per_image)
    preds_bbox_path = os.path.join(output_dir, "preds_bbox.json")
    preds_segm_path = os.path.join(output_dir, "preds_segm.json")
    with open(preds_bbox_path, "w") as f:
        json.dump(preds_bbox_list, f)
    with open(preds_segm_path, "w") as f:
        json.dump(preds_segm_list, f)

    # run COCO eval for bbox and segm
    print("Running COCO eval for bbox")
    bbox_summary = run_coco_eval(gt_json_path, preds_bbox_path, iou_type='bbox')
    print("Running COCO eval for segm")
    segm_summary = run_coco_eval(gt_json_path, preds_segm_path, iou_type='segm')

    # compute PR at IoU=0.5 score_thresh_pr
    print(f"Computing PR at IoU={iou_thresh_pr}, score={score_thresh_pr} (bbox)")
    prec_box, rec_box, tp, fp, fn = compute_pr_at_threshold(gt_per_image, preds_per_image, iou_thresh=iou_thresh_pr, score_thresh=score_thresh_pr, iou_type='bbox')
    print(f"Box PR -> precision: {prec_box:.4f}, recall: {rec_box:.4f} (TP {tp}, FP {fp}, FN {fn})")

    print(f"Computing PR at IoU={iou_thresh_pr}, score={score_thresh_pr} (mask)")
    prec_mask, rec_mask, tp_m, fp_m, fn_m = compute_pr_at_threshold(gt_per_image, preds_per_image, iou_thresh=iou_thresh_pr, score_thresh=score_thresh_pr, iou_type='mask')
    print(f"Mask PR -> precision: {prec_mask:.4f}, recall: {rec_mask:.4f} (TP {tp_m}, FP {fp_m}, FN {fn_m})")

    # return structured results
    return {
        "bbox_mAP": bbox_summary,
        "segm_mAP": segm_summary,
        "box_PR_at_thresh": {"precision": prec_box, "recall": rec_box, "TP": tp, "FP": fp, "FN": fn},
        "mask_PR_at_thresh": {"precision": prec_mask, "recall": rec_mask, "TP": tp_m, "FP": fp_m, "FN": fn_m},
        "preds_bbox_path": preds_bbox_path,
        "preds_segm_path": preds_segm_path,
        "gt_json_path": gt_json_path
    }