# Paste this in a new cell / file and run it.
# It expects:
#  - `model` your AnomalyAwareMaskRCNN instance (on device)
#  - `anom_mod` your TorchInferencer instance (to produce heatmaps)
#  - `image_path` a path to one problematic image
#  - `device` set as torch.device(...)
#
# If you already used run_one_debug_case and have `dbg` and `pil`, set `USE_EXISTING_DBG=True`
# otherwise it will run forward_debug to get dbg.

import numpy as np
import torch
import os, math
from PIL import Image
import matplotlib.pyplot as plt
import cv2

from eval.utils import forward_debug, load_checkpoint_weights
from backbone.maskrcnn import AnomalyAwareMaskRCNN
from anomalib.deploy import TorchInferencer

USE_EXISTING_DBG = False  # set True if you have dbg/pil in scope
image_path = "data/raw/testNG/(P)1044491-00-F(1T)ABD25316C014580-1-1-0-0-5000-RAW.png"  # <- change this
outdir = "diag_match_out"
os.makedirs(outdir, exist_ok=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# instantiate model architecture (must match training)
model = AnomalyAwareMaskRCNN(num_classes=3, proj_channels=1, pretrained=False)
# load weights
model = load_checkpoint_weights('exported_models/best_models/best_epoch_6.pth', model, device=device, strict=False)
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

def run_and_get_dbg(model, anom_mod, image_path, device):
    pil = Image.open(image_path).convert("RGB")
    heat = anom_mod.predict(image_path).anomaly_map
    # forward_debug must be available (from previous helper). If not, recreate minimal one here.
    dbg = forward_debug(model, pil, heat, device)  # uses your forward_debug helper
    return dbg, pil

if USE_EXISTING_DBG:
    # assume dbg and pil exist in current scope
    pass
else:
    dbg, pil = run_and_get_dbg(model, anom_mod, image_path, device)

# helper conversions
def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

# extract key items
orig_h, orig_w = dbg["image_orig_size"]           # (H_orig, W_orig)
H_pad, W_pad = dbg["image_list_tensors"].shape[1], dbg["image_list_tensors"].shape[2]  # careful: image_list_tensors is (B,3,H,W) in dbg: we saved permuted form
# note: earlier debug saved image_list_tensors as (B,3,H_pad,W_pad) moved to CPU
img_pad = dbg["image_list_tensors"][0].permute(1,2,0).numpy()  # H_pad x W_pad x 3
resized_h, resized_w = dbg["image_list_image_sizes"][0]        # (h_i, w_i) after transform, unpadded
heat_prep = dbg["heatmaps_prepared"][0,0].numpy()             # H_pad x W_pad
proposals = dbg["rpn_proposals"]                              # list structure, proposals[0] is per-image

# normalize proposals to numpy array of shape (N,4) in resized coords
p0 = proposals[0]
if hasattr(p0, "bbox"):
    prop_boxes = to_np(p0.bbox)
elif isinstance(p0, torch.Tensor):
    prop_boxes = to_np(p0)
else:
    # maybe torchvision returns list of tensors
    try:
        prop_boxes = to_np(p0[0])
    except Exception:
        raise RuntimeError("Unknown proposals format; inspect type:", type(p0))

# detections before postprocess and after postprocess
det_before = dbg["detections_before_post"]
det_after = dbg["detections_after_post"]
# normalize to dict format for first image
def pick_first(det):
    if isinstance(det, (list, tuple)):
        return det[0]
    return det
db0 = pick_first(det_before)
da0 = pick_first(det_after)

boxes_before = to_np(db0.get("boxes"))      # should be (K,4), likely in resized OR original coords depending on behavior
boxes_after = to_np(da0.get("boxes"))       # (K,4) presumably original coords
masks_after = to_np(da0.get("masks"))       # (K,1,H_orig,W_orig) or (K,H,W)

print("ORIG size (H,W):", (orig_h, orig_w))
print("RESIZED (unpadded) size (h_i,w_i):", (resized_h, resized_w))
print("IMAGE_LIST padded size (H_pad,W_pad):", img_pad.shape[:2])
print("prop_boxes sample (first 5):\n", np.round(prop_boxes[:5], 2))
print("boxes_before sample (first 5):\n", np.round(boxes_before[:5], 2))
print("boxes_after sample (first 5):\n", np.round(boxes_after[:5], 2))

# scale factor from resized -> original
scale_h = orig_h / resized_h
scale_w = orig_w / resized_w
print("scale factors (H_scale, W_scale):", (scale_h, scale_w))

# Apply scale to proposals (resized -> original)
prop_boxes_scaled = prop_boxes.copy().astype(np.float64)
prop_boxes_scaled[:, [0,2]] *= scale_w
prop_boxes_scaled[:, [1,3]] *= scale_h

print("prop_boxes_scaled sample (first 5):\n", np.round(prop_boxes_scaled[:5], 2))

# Function: IoU between boxes arrays
def iou_boxes(boxA, boxB):
    # boxes in x1,y1,x2,y2
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA + 1)
    interH = max(0, yB - yA + 1)
    inter = interW * interH
    areaA = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    areaB = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    union = areaA + areaB - inter
    if union <= 0:
        return 0.0
    return inter / union

# For each box_after (final detection), find best matching scaled proposal (by IoU),
# report IoU and dx,dy between proposal_scaled top-left and detection top-left.
results = []
for i, det in enumerate(boxes_after):
    best_iou = 0.0
    best_j = -1
    for j, pb in enumerate(prop_boxes_scaled):
        iou = iou_boxes(det, pb)
        if iou > best_iou:
            best_iou = iou
            best_j = j
    if best_j == -1:
        print(f"det {i}: no matching proposal found")
        continue
    pb = prop_boxes_scaled[best_j]
    dx = (pb[0] - det[0])
    dy = (pb[1] - det[1])
    results.append({"det_i": i, "best_prop_j": best_j, "iou": best_iou, "dx": dx, "dy": dy, "det": det, "prop_scaled": pb})

# Print summary
print("Matched proposal -> detection summary (first 10):")
for r in results[:10]:
    print(f" det#{r['det_i']} prop#{r['best_prop_j']} iou={r['iou']:.3f} dx={r['dx']:.2f} dy={r['dy']:.2f}")

if len(results) > 0:
    mean_dx = np.mean([r['dx'] for r in results])
    mean_dy = np.mean([r['dy'] for r in results])
    print("Mean dx, dy (proposal_scaled top-left - detection top-left):", mean_dx, mean_dy)
else:
    print("No matched pairs found; cannot compute offsets.")

# Visualize: overlay prop_boxes_scaled (cyan), boxes_after (yellow), and masks_after (red) on original image
orig_arr = np.array(pil).copy()
vis = orig_arr.copy()
# draw masks (if present)
if masks_after is not None:
    for k in range(min(masks_after.shape[0], 10)):
        m = masks_after[k]
        if m.ndim == 4 or m.ndim == 3 and m.shape[0] == 1:
            # accept (1,H,W) or (H,W)
            if m.ndim == 4:
                m = m[0,0]
            elif m.ndim == 3 and m.shape[0] == 1:
                m = m[0]
        mask_bool = (m > 0.5)
        if mask_bool.sum() == 0:
            continue
        vis[mask_bool] = (vis[mask_bool] * 0.4 + np.array([255,0,0]) * 0.6).astype(np.uint8)

# draw boxes_after (yellow)
for b in boxes_after[:50]:
    x1,y1,x2,y2 = [int(round(v)) for v in b]
    cv2.rectangle(vis, (x1,y1), (x2,y2), color=(255,255,0), thickness=2)

# draw prop_boxes_scaled (cyan)
for pb in prop_boxes_scaled[:50]:
    x1,y1,x2,y2 = [int(round(v)) for v in pb]
    cv2.rectangle(vis, (x1,y1), (x2,y2), color=(255,255,255), thickness=1)  # white thin

outv = os.path.join(outdir, "match_overlay.png")
Image.fromarray(vis).save(outv)
print("Saved overlay:", outv)

# Save a zoom crop around first detection for close inspection
if len(boxes_after) > 0:
    b = boxes_after[0]
    x1,y1,x2,y2 = [int(max(0,min(orig_w,round(v)))) for v in b]
    pad = 30
    x0 = max(0, x1-pad); y0 = max(0, y1-pad); x3 = min(orig_w, x2+pad); y3 = min(orig_h, y2+pad)
    crop = vis[y0:y3, x0:x3]
    Image.fromarray(crop).save(os.path.join(outdir, "match_overlay_crop0.png"))
    print("Saved crop:", os.path.join(outdir, "match_overlay_crop0.png"))

print("Done. Inspect 'match_overlay.png' and printed matching stats.")
