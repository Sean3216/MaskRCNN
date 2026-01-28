import os
from PIL import Image, ImageDraw
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from pycocotools import mask as mask_utils


# -------------------------
# Main data classes
# -------------------------

class Image_Dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.img_dir = os.path.join(data_dir, 'images')
        self.lbl_dir = os.path.join(data_dir,'labels')
        self.transform = transform
        
        # Getting the images
        self.image_files = sorted([
            f for f in os.listdir(self.img_dir)
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ])
        self._ensure_class_map()

    def _ensure_class_map(self):
        seen = set()
        for im in self.image_files:
            lbl = os.path.splitext(im)[0] + '.txt'
            p = os.path.join(self.lbl_dir, lbl)
            if not os.path.exists(p):
                continue
            with open(p, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    tok = line.split()
                    seen.add(tok[0])
        self.class_map = {k: i+1 for i, k in enumerate(sorted(seen))}
        print("Built class_map:", self.class_map)
        
    def __len__(self):
        return len(self.image_files)
    
    def _parse_label_file(self, filepath, width, height):
        boxes = []
        labels = []
        masks = []

        if not os.path.exists(filepath):
            return boxes, labels, masks
        
        with open(filepath, 'r') as f:
            for line in f:
                line_inside = line.strip()
                if not line_inside:
                    continue
                tokens = line_inside.split()

                cls_token = tokens[0]

                if cls_token not in self.class_map:
                    #it means it's the first time we're seeing this class
                    self.class_map[cls_token] = max(self.class_map.values(), default = 0) + 1
                
                label = self.class_map[cls_token]

                coords = [float(x) for x in tokens[1:]]
                if len(coords) % 2 != 0 or len(coords) < 6:
                    #not enough points to make a polygon (skip)
                    continue

                # convert normalized coords to pixel coords

                pts = []
                for i in range(0, len(coords), 2):
                    nx = coords[i]
                    ny = coords[i+1]

                    x = float(nx) * float(width)
                    y = float(ny) * float(height)
                    pts.append((x,y))

                #rasterize polygon --> mask
                mask_img = Image.new("L", (width, height), 0)
                ImageDraw.Draw(mask_img).polygon([p for xy in pts for p in xy], outline = 1, fill=1)
                mask_arr = np.array(mask_img, dtype = np.uint(8))
                
                #compute bounding box from mask
                ys, xs = np.where(mask_arr > 0)
                if ys.size == 0:
                    #polygon is weird
                    continue 
                xmin = float(xs.min())
                xmax = float(xs.max())
                ymin = float(ys.min())
                ymax = float(ys.max())

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)
                masks.append(mask_arr)
        return boxes, labels, masks
    
    def __getitem__(self, idx):
        # Get the file name from the image list
        img_file_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_file_name)
        lbl_path = os.path.join(self.lbl_dir,os.path.splitext(img_file_name)[0] + '.txt')

        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        boxes, labels, masks = self._parse_label_file(lbl_path, width, height)

        #if we detect no instance, create empty targets
        if len(boxes) == 0:
            boxes_t = torch.zeros((0,4), dtype = torch.float32)
            labels_t = torch.zeros((0,), dtype = torch.int64)
            masks_t = torch.zeros((0, height, width), dtype = torch.uint8)
            area = torch.zeros((0,), dtype = torch.float32)
            iscrowd = torch.zeros((0,), dtype = torch.int64)
        else:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype = torch.int64)
            masks_t = torch.as_tensor(np.stack(masks, axis = 0), dtype=torch.uint8)
            area = (boxes_t[:,3] - boxes_t[:,1]) * (boxes_t[:, 2] - boxes_t[:,0])
            iscrowd = torch.zeros((len(boxes),), dtype = torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "masks": masks_t,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd
        }
        # --- Apply Synchronized Transforms ---
        # We must apply the *same* random transformations (like flips)
        # to both the image and the mask. We do this by setting a
        # fixed random seed before each transform call.
        if self.transform:
            torch.manual_seed(42)
            image = self.transform(image)

        # --- Return Image and Target---        
        return image, target
    
def load_Image_Dataloader(data_dir, batch_size = 4, transform_func = None):
    dataset = Image_Dataset(data_dir, transform = transform_func)
    def collate_fn(batch):
        return tuple(zip(*batch))
    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = True,
        collate_fn=collate_fn
    )
    num_classes = len(dataset.class_map) + 1 #considering 1 more class detecting background
    print("Detected class_map:")
    print(dataset.class_map)
    print("Detected number of class:")
    print(num_classes)
    return dataloader, num_classes


# -------------------------
# Eval data util
# -------------------------

# Convert binary mask -> COCO RLE (counts must be ascii string)
def mask_to_rle(mask):
    # mask: HxW binary (0/1) numpy
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    # pycocotools returns counts as bytes on py3; decode for json
    if isinstance(rle['counts'], bytes):
        rle['counts'] = rle['counts'].decode('ascii')
    return rle

# Convert GT / preds to COCO json lists
def build_coco_gt_json(image_info_list, gt_per_image, category_map):
    """
    category_map: dict name->id or id->name? We expect model labels (ints) are same as category ids.
    Returns full COCO-style dict
    """
    annotations = []
    ann_id = 1
    for img in image_info_list:
        img_id = img["id"]
        gts = gt_per_image.get(img_id, [])
        for g in gts:
            mask = g["mask"]
            rle = mask_to_rle(mask)
            annotations.append({
                "id": ann_id,
                "image_id": int(img_id),
                "category_id": int(g["category_id"]),
                "bbox": [float(x) for x in g["bbox"]],
                "area": float(np.sum(mask > 0)),
                "iscrowd": 0,
                "segmentation": rle
            })
            ann_id += 1

    # build categories from category_map: assume {token: id} or {id: name}
    categories = []
    if isinstance(category_map, dict):
        # try detect format
        # if keys are strings (tokens) and values ints -> invert to id->name
        if all(isinstance(k, str) and isinstance(v, int) for k, v in category_map.items()):
            inv = {v: k for k, v in category_map.items()}
            for cid, name in inv.items():
                categories.append({"id": int(cid), "name": str(name)})
        else:
            # assume keys are ints
            for k, v in category_map.items():
                categories.append({"id": int(k), "name": str(v)})
    else:
        # fallback single category
        categories.append({"id": 1, "name": "1"})

    coco = {"images": image_info_list, "annotations": annotations, "categories": categories, "info": "Custom Image Dataset Ground Truth"}
    return coco

def build_coco_pred_list(preds_per_image):
    """
    Build two lists:
      preds_bbox_list: entries with fields image_id, category_id, bbox (xywh), score
      preds_segm_list: same plus segmentation (RLE)
    """
    preds_bbox_list = []
    preds_segm_list = []
    for img_id, pred in preds_per_image.items():
        boxes = pred["boxes"]
        labels = pred["labels"]
        scores = pred["scores"]
        masks = pred["masks"]
        N = boxes.shape[0]
        for i in range(N):
            x1, y1, x2, y2 = boxes[i].tolist()
            w = x2 - x1
            h = y2 - y1
            bbox = [float(x1), float(y1), float(w), float(h)]
            det = {"image_id": int(img_id), "category_id": int(labels[i]), "bbox": bbox, "score": float(scores[i])}
            preds_bbox_list.append(det)
            if masks is not None:
                rle = mask_to_rle(masks[i])
                det_segm = det.copy()
                det_segm["segmentation"] = rle
                preds_segm_list.append(det_segm)
    return preds_bbox_list, preds_segm_list
