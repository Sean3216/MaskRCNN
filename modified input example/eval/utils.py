import os
import json
import math
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from backbone.maskrcnn import AnomalyAwareMaskRCNN
from data import build_coco_gt_json, build_coco_pred_list

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize

from anomalib.deploy import TorchInferencer

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
def predict_image(model, pil_img, heatmap_feat, device, score_thresh=0.5):
    """
    Run model on a single PIL image (batch size 1).
    Returns dict with numpy arrays: boxes (Nx4), labels (N,), scores (N,), masks (N,H,W) or None.
    """
    model.eval()
    transform = ToTensor()
    img_t = transform(pil_img).to(device)
    heatmap_feat = heatmap_feat.to(device)

    with torch.no_grad():
        outputs = model([img_t], [heatmap_feat])[0]

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
    model = AnomalyAwareMaskRCNN(
        num_classes=num_classes, 
        #proj_channels=1, 
        pretrained=use_pretrained
    )
    # load weights
    model = load_checkpoint_weights(checkpoint_path, model, device=device, strict=False)
    if os.path.exists("anomaly_mod") and len(os.listdir("anomaly_mod")) == 1:
        anomaly_mod_name = os.listdir('anomaly_mod')[0]
        os.environ['TRUST_REMOTE_CODE'] = '1'
        anom_mod = TorchInferencer(
            path = os.path.join('anomaly_mod', anomaly_mod_name),
            device = 'auto'
        )
    else:
        raise ValueError(
            """
            Anomaly model folder does not exist or model folder contains more than one model!\n
            Expected model folder "anomaly_mod" folder in directory
            """
        )

    # important: if you changed image_mean/std during training, set same here:
    # model.transform.image_mean = [..]
    # model.transform.image_std  = [..]

    img_files = [f for f in sorted(os.listdir(images_dir)) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    results = []
    time_accumulated = []
    time_anom = []
    time_maskrcnn = []
    for fn in img_files:
        path = os.path.join(images_dir, fn)
        pil = Image.open(path).convert("RGB")
        W, H = pil.size  # Get original dimensions

        # 1. Get raw heatmap (Numpy array, usually small e.g., 256x256)
        infer_start = time.perf_counter()
        anom_start = time.perf_counter()
        heat_raw = anom_mod.predict(path).anomaly_map 
        anom_end = time.perf_counter()

        # 2. CRITICAL FIX: Resize heatmap to match the original image exactly
        # We use OpenCV here for speed, or you can use PIL.
        # This ensures the heatmap pixels align 1:1 with the original image pixels.
        heat_raw = heat_raw.squeeze().detach().cpu().numpy()
        heat_resized = cv2.resize(heat_raw, (W, H)) 

        # 3. Convert to Tensor and add Channel dimension (1, H, W)
        # predict_image expects a tensor on the device
        heatmap_t = torch.from_numpy(heat_resized).unsqueeze(0).float()

        # Pass the processed tensor, not the raw numpy array
        maskrcnn_start = time.perf_counter()
        preds = predict_image(model, pil, heatmap_feat=heatmap_t, device=device, score_thresh=score_thresh)
        maskrcnn_end = time.perf_counter()
        infer_end = time.perf_counter()

        time_accumulated.append(infer_end-infer_start)
        time_anom.append(anom_end-anom_start)
        time_maskrcnn.append(maskrcnn_end-maskrcnn_start)
        
        # dbg = run_one_debug_case(model, path, anom_mod, device, out_dir= os.path.join("debug_folder",fn))
        # analyze_dbg_and_try_variants(model, dbg, pil,out_dir=os.path.join("debug_folder",fn,'additional_debug'))

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
    avg_time = sum(time_accumulated)/len(time_accumulated)
    avg_anom = sum(time_anom)/len(time_anom)
    avg_maskrcnn = sum(time_maskrcnn)/len(time_maskrcnn)
    print("="*100)
    print(f"Average total inference time (seconds): {avg_time} seconds")
    print(f"Average total inference time (milliseconds): {avg_time*1000} ms")
    print("="*100)
    print(f"Average anomaly model (fastflow) inference time (seconds): {avg_anom} seconds")
    print(f"Average anomaly model (fastflow) inference time (milliseconds): {avg_anom*1000} ms")
    print("="*100)
    print(f"Average instance segmentation (maskrcnn) inference time (seconds): {avg_maskrcnn} seconds")
    print(f"Average instance segmentation (maskrcnn) inference time (milliseconds): {avg_maskrcnn*1000} ms")

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

        img_t, target, heatmap_t = dataset[idx]
        img_t = img_t.to(device)
        #target = target.to(device)
        heatmap_t = heatmap_t.to(device)

        #continuing image processing
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
        #img_t = transform(pil).to(device)
        with torch.no_grad():
            out = model([img_t],[heatmap_t])[0]

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

    model = AnomalyAwareMaskRCNN(
        num_classes=num_classes,
        pretrained = False
    )
    model = load_checkpoint_weights(checkpoint_path, model=model, device=device, strict = False)
    
    preds_per_image, gt_per_image, image_info_list = run_inference_and_collect(
        model, 
        dataset, 
        device, 
        score_thresh=score_thresh_inference
    )

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

# -------------------------
# Debugging
# -------------------------
def forward_debug(model, pil_img, heatmap, device):
    """
    Run one forward pass but return intermediate items for debugging alignment.
    Returns a dict with:
    - image_orig_size: (H_orig, W_orig)
    - image_list_tensors: padded transformed image tensor (B=1,3,H_pad,W_pad) - cpu
    - image_list_image_sizes: list of per-image (h_i,w_i) after transform (unpadded)
    - heatmaps_prepared: padded heatmap tensor (B=1,1,H_pad,W_pad) - cpu
    - features: dict of FPN features (cpu)
    - features_for_rpn: dict of modified features used by rpn (cpu)
    - rpn_proposals: proposals returned by rpn (list of Boxes or tensors) - CPU
    - detections_before_post: outputs from roi_heads (in resized coords) - CPU
    - detections_after_post: outputs from transform.postprocess(...) (in original image coords) - CPU
    """
    model.eval()
    transform = ToTensor()
    img_t = transform(pil_img).to(device)  # (3,H,W)
    hm_t = heatmap
    # normalize heatmap to tensor like your forward expects (H,W) or (1,H,W)
    if isinstance(hm_t, np.ndarray):
        hm_t = torch.from_numpy(hm_t).float()
    if hm_t.dim() == 2:
        pass
    elif hm_t.dim() == 3 and hm_t.shape[0] == 1:
        hm_t = hm_t.squeeze(0)
    hm_t = hm_t.to(device).float()
    with torch.no_grad():
        # replicate what your forward does but stop at points and collect variables
        was_tensor = isinstance(img_t, torch.Tensor)
        images_list = [img_t]  # list of (3,H,W)
        image_list, _ = model.model.transform(images_list)  # use inner maskrcnn transform (ImageList)
        image_tensors = image_list.tensors  # B,3,H_pad,W_pad
        device_used = image_tensors.device
        dtype_used = image_tensors.dtype
        per_image_sizes = image_list.image_sizes  # list of (h_i,w_i)
        H_pad, W_pad = image_tensors.shape[-2:]

        # --- prepare heatmaps exactly like model.forward does (per-image resize + pad to H_pad,W_pad) ---
        hm_tensors = []
        for i in range(len(per_image_sizes)):
            cur = hm_t.clone().float()
            # cur shape (H_src,W_src) or (1,H_src,W_src)
            if cur.dim() == 3 and cur.shape[0] == 1:
                cur = cur.squeeze(0)
            if cur.max() > 1.5:
                cur = (cur / 255.0).clamp(0.0, 1.0)
            # resize to per-image size (unpadded)
            h_i, w_i = per_image_sizes[i]
            cur_res = F.interpolate(cur.unsqueeze(0).unsqueeze(0), size=(h_i, w_i), mode='bilinear', align_corners=False).squeeze(0)
            # pad to H_pad,W_pad
            pad = torch.zeros((1, H_pad, W_pad), dtype=cur_res.dtype, device=cur_res.device)
            pad[:, :h_i, :w_i] = cur_res
            hm_tensors.append(pad)
        heatmaps_tensor = torch.stack(hm_tensors, dim=0).to(device=device_used, dtype=dtype_used)  # (B,1,H_pad,W_pad)

        # 2) backbone features and features_for_rpn
        features = model.model.backbone(image_tensors)  # OrderedDict
        # build features_for_rpn via your _prepare_features_for_rpn
        features_for_rpn = model._prepare_features_for_rpn(features, heatmaps_tensor)

        # 3) call rpn with modified features_for_rpn (capture proposals)
        proposals, proposal_losses = model.model.rpn(image_list, features_for_rpn, targets=None)

        # 4) call roi_heads using ORIGINAL features (unchanged)
        detections_resized, detector_losses = model.model.roi_heads(features, proposals, image_list.image_sizes, targets=None)

        # 5) postprocess -> convert detections_resized -> original image sizes
        # compute original_image_sizes from the original PIL
        orig_h, orig_w = pil_img.size[1], pil_img.size[0]  # PIL .size => (W,H)
        original_image_sizes = [(orig_h, orig_w)]
        detections_post = model.model.transform.postprocess(detections_resized, image_list.image_sizes, original_image_sizes)

    # move outputs to cpu for inspection
    def tocpu(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    out = {
        "image_orig_size": (int(orig_h), int(orig_w)),
        "image_list_tensors": image_tensors.detach().cpu(),
        "image_list_image_sizes": per_image_sizes,
        "heatmaps_prepared": heatmaps_tensor.detach().cpu(),
        "features_shapes": {k: tuple(v.shape) for k, v in features.items()},
        "features_for_rpn_shapes": {k: tuple(v.shape) for k, v in features_for_rpn.items()},
        "rpn_proposals": proposals,            # keep as-is (likely list of Boxes / Tensors)
        "detections_before_post": detections_resized,
        "detections_after_post": detections_post
    }
    return out
def run_one_debug_case(model, image_path, anom_mod, device, out_dir="debug_out"):
    os.makedirs(out_dir, exist_ok=True)
    pil = Image.open(image_path).convert("RGB")
    heat_raw = anom_mod.predict(image_path).anomaly_map  # whatever you previously used

    # forward debug
    dbg = forward_debug(model, pil, heat_raw, device)

    # PRINT diagnostics
    print("=== DIAGNOSTICS ===")
    print("Original image (H,W):", dbg["image_orig_size"])
    print("ImageList padded tensor shape (B,3,H_pad,W_pad):", tuple(dbg["image_list_tensors"].shape))
    print("Per-image transformed sizes (unpadded):", dbg["image_list_image_sizes"])
    print("Prepared heatmap shape (B,1,H_pad,W_pad):", tuple(dbg["heatmaps_prepared"].shape))
    print("Feature maps shapes (FPN):")
    for k, s in dbg["features_shapes"].items():
        print(" ", k, s)
    print("Features-for-RPN shapes:")
    for k, s in dbg["features_for_rpn_shapes"].items():
        print(" ", k, s)

    # Save images for visual inspection
    # 1) padded transformed image (top-left region is unpadded)
    img_pad = dbg["image_list_tensors"][0].permute(1,2,0).numpy()  # H_pad,W_pad,3 in 0..1
    plt.imsave(os.path.join(out_dir, "img_padded.png"), np.clip(img_pad, 0, 1))

    # 2) prepared heatmap (first channel)
    hm = dbg["heatmaps_prepared"][0,0].numpy()  # H_pad,W_pad
    plt.imsave(os.path.join(out_dir, "heat_prep.png"), hm, cmap="viridis")

    # 3) overlay heat on padded image
    overlay = img_pad.copy()
    # normalize hm to 0..1
    hm_norm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    overlay[..., 0] = np.clip(overlay[..., 0] * 0.6 + hm_norm * 0.4, 0, 1)
    plt.imsave(os.path.join(out_dir, "overlay_padded.png"), np.clip(overlay, 0, 1))

    # Print RPN proposals numeric samples (they are in resized coords: per-image unpadded size)
    proposals = dbg["rpn_proposals"]
    print("RPN proposals type:", type(proposals))
    # proposals is typically a List[Tensor] or Boxes; inspect the first element
    try:
        p0 = proposals[0]
        if hasattr(p0, "bbox"):
            pboxes = p0.bbox.detach().cpu().numpy()
        else:
            pboxes = p0.detach().cpu().numpy()
        print("Number of proposals (first image):", len(pboxes))
        print("First 10 proposals (x1,y1,x2,y2):")
        for i, pb in enumerate(pboxes[:10]):
            print(" ", i, pb)
    except Exception as e:
        print("Could not print proposals:", e)

    # Print detections before and after postprocess
    det_before = dbg["detections_before_post"]
    det_after = dbg["detections_after_post"]
    print("DETECTIONS before postprocess (resized coords):")
    try:
        if isinstance(det_before, (list, tuple)):
            db0 = det_before[0]
        else:
            db0 = det_before
        boxes_b = db0["boxes"].detach().cpu().numpy()
        print(" boxes (first 10):", boxes_b[:10])
    except Exception as e:
        print("Could not read detections_before_post boxes:", e)

    print("DETECTIONS after postprocess (original image coords):")
    try:
        if isinstance(det_after, (list, tuple)):
            da0 = det_after[0]
        else:
            da0 = det_after
        boxes_a = da0["boxes"].detach().cpu().numpy()
        masks_a = da0.get("masks", None)
        if masks_a is not None:
            masks_a = masks_a.detach().cpu().numpy()
        print(" boxes (first 10):", boxes_a[:10])
        if masks_a is not None:
            print(" masks shape:", masks_a.shape)
    except Exception as e:
        print("Could not read detections_after_post boxes/masks:", e)

    print("Saved debug images to", out_dir)
    return dbg

# ANALYZE dbg returned by your run_one_debug_case() earlier
def analyze_dbg_and_try_variants(model, dbg, pil_img, out_dir="dbg_analysis"):
    os.makedirs(out_dir, exist_ok=True)

    # get data from dbg
    orig_h, orig_w = dbg["image_orig_size"]
    img_pad = dbg["image_list_tensors"][0].permute(1,2,0).numpy()  # H_pad x W_pad x 3 (0..1)
    hm_prep = dbg["heatmaps_prepared"][0,0].numpy()                # H_pad x W_pad
    per_image_sizes = dbg["image_list_image_sizes"]                # [(h_i,w_i)]
    proposals = dbg["rpn_proposals"]
    det_before = dbg["detections_before_post"]
    det_after = dbg["detections_after_post"]

    # normalize det_before/after reading (handle list vs dict)
    def extract_first_det(d):
        if isinstance(d, (list, tuple)):
            d0 = d[0]
        else:
            d0 = d
        return d0

    db0 = extract_first_det(det_before)
    da0 = extract_first_det(det_after)

    # ensure boxes and masks are tensors -> numpy
    def to_numpy(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    boxes_before = to_numpy(db0.get("boxes", None))
    masks_before = to_numpy(db0.get("masks", None))
    boxes_after = to_numpy(da0.get("boxes", None))
    masks_after = to_numpy(da0.get("masks", None))

    print("orig image (H,W):", (orig_h, orig_w))
    print("per-image transformed (unpadded):", per_image_sizes)
    print("image_list padded shape (H_pad,W_pad):", img_pad.shape[:2])
    print("prepared heatmap shape:", hm_prep.shape)
    print("boxes_before shape:", None if boxes_before is None else boxes_before.shape)
    print("boxes_after shape:", None if boxes_after is None else boxes_after.shape)
    print("masks_after shape:", None if masks_after is None else masks_after.shape)

    # --- compute mask bbox vs box for detections_after_post ---
    if boxes_after is None or masks_after is None:
        print("No boxes/masks available to compare.")
    else:
        N = boxes_after.shape[0]
        print(f"Comparing {N} detections (boxes vs mask bboxes):")
        deltas = []
        for i in range(N):
            bx = boxes_after[i]  # [x1,y1,x2,y2]
            # masks_after shape (N, 1, H_orig, W_orig) or (N, H, W)
            m = masks_after[i]
            if m.ndim == 3 and m.shape[0] == 1:
                m = m[0]
            # ensure bool
            mask_bool = (m > 0.5)
            ys, xs = np.where(mask_bool)
            if ys.size == 0:
                print(f"  det {i}: mask empty")
                continue
            mx0, my0, mx1, my1 = xs.min(), ys.min(), xs.max(), ys.max()
            bx0, by0, bx1, by1 = bx[0], bx[1], bx[2], bx[3]
            dx = mx0 - bx0
            dy = my0 - by0
            deltas.append((dx, dy))
            print(f"  det {i}: box=[{bx0:.1f},{by0:.1f},{bx1:.1f},{by1:.1f}] mask_bbox=[{mx0},{my0},{mx1},{my1}] dx={dx:.1f} dy={dy:.1f}")
        if len(deltas) > 0:
            avg_dx = np.mean([d[0] for d in deltas])
            avg_dy = np.mean([d[1] for d in deltas])
            print(f"Average delta (mask_bbox - box_top_left): dx={avg_dx:.3f}, dy={avg_dy:.3f}")
        else:
            print("No positive masks found to compute deltas.")

    # --- Visualize: overlay boxes_after and masks_after on original PIL image (no matplotlib axis tricks) ---
    orig_arr = np.array(pil_img).copy()  # H,W,3
    viz = orig_arr.astype(np.float32) / 255.0

    # draw masks as red overlay (alpha)
    alpha = 0.45
    if masks_after is not None:
        for i in range(min(masks_after.shape[0], 10)):
            m = masks_after[i]
            if m.ndim == 3 and m.shape[0] == 1:
                m = m[0]
            mask_bool = (m > 0.5)
            if mask_bool.sum() == 0:
                continue
            # overlay red
            viz[..., 0][mask_bool] = viz[..., 0][mask_bool] * (1 - alpha) + 1.0 * alpha
            viz[..., 1][mask_bool] = viz[..., 1][mask_bool] * (1 - alpha) + 0.0 * alpha
            viz[..., 2][mask_bool] = viz[..., 2][mask_bool] * (1 - alpha) + 0.0 * alpha

    # draw boxes
    import cv2
    vimg = (viz * 255).astype(np.uint8).copy()
    if boxes_after is not None:
        for i, b in enumerate(boxes_after):
            x1,y1,x2,y2 = [int(round(v)) for v in b]
            cv2.rectangle(vimg, (x1,y1), (x2,y2), color=(255,255,0), thickness=2)

    out_vpath = os.path.join(out_dir, "overlay_boxes_masks_on_original.png")
    Image.fromarray(vimg).save(out_vpath)
    print("Saved overlay (boxes+masks on original) to", out_vpath)

    # --- Quick test: try alternate postprocess mapping by swapping (H,W) -> (W,H) and visualizing ---
    # This tests whether original_image_sizes ordering is incorrect in your forward.
    from torchvision.models.detection.image_list import ImageList
    # reconstruct detections_resized from dbg (we have det_before as original variable)
    # We'll call transform.postprocess with swapped sizes to see effect
    try:
        # fetch detections_resized (the outputs before you called postprocess)
        dets_resized = det_before
        # Try mapping with original sizes (H,W) [this is what you used]
        orig_sizes_h_w = [(orig_h, orig_w)]
        det_after_test = model.model.transform.postprocess(dets_resized, dbg["image_list_image_sizes"], orig_sizes_h_w)
        da0_test = det_after_test[0]
        btest = da0_test["boxes"].detach().cpu().numpy()
        print("postprocess with (H,W) produced boxes sample:", btest[:3])
        # Now try swapped (W,H)
        orig_sizes_w_h = [(orig_w, orig_h)]
        det_swapped = model.model.transform.postprocess(dets_resized, dbg["image_list_image_sizes"], orig_sizes_w_h)
        bsw = det_swapped[0]["boxes"].detach().cpu().numpy()
        print("postprocess with swapped (W,H) produced boxes sample:", bsw[:3])

        # Save overlay for swapped mapping using boxes bsw
        vimg2 = orig_arr.copy()
        vimg2 = vimg2.astype(np.float32)/255.0
        vimg2 = (vimg2*255).astype(np.uint8)
        for b in bsw[:10]:
            x1,y1,x2,y2 = [int(round(v)) for v in b]
            cv2.rectangle(vimg2, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)
        out_vpath2 = os.path.join(out_dir, "overlay_boxes_swapped_postprocess.png")
        Image.fromarray(vimg2).save(out_vpath2)
        print("Saved swapped-postprocess overlay:", out_vpath2)
    except Exception as e:
        print("Could not run postprocess experiments:", e)

    return None