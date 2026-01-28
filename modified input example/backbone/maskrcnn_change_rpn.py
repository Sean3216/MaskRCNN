import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import torchvision
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_V2_Weights,
    mask_rcnn,
    faster_rcnn,
)

import numpy as np


def get_maskrcnn_model(num_classes, pretrained=True):
    model = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None,
        min_size=480,
        max_size=800,
    )
    # replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    # replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 128
    model.roi_heads.mask_predictor = mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model


def _replace_first_conv_of_rpn_head_with_extra_in_channels(rpn_head, proj_extra_channels=1):
    """
    In-place: replace the first Conv2d inside rpn_head.conv so that it accepts
    (orig_in + proj_extra_channels) input channels. Copy old weights into first channels
    and zero the extra channels.
    Returns True if replaced, False otherwise.
    """
    # Helper: try to replace conv when it's embedded as child[0] in a Sequential wrapper (Conv2dNormActivation)
    def try_replace_in_sequential(seq):
        for idx, child in enumerate(list(seq)):  # iterate over direct children
            # case A: child is a Sequential-like wrapper and its first element is Conv2d
            if isinstance(child, nn.Sequential) and len(child) > 0 and isinstance(child[0], nn.Conv2d):
                conv_old = child[0]
                # build new conv with extra input channels
                orig_in = conv_old.in_channels
                orig_out = conv_old.out_channels
                ksize = conv_old.kernel_size
                stride = conv_old.stride
                padding = conv_old.padding
                bias = conv_old.bias is not None
                new_conv = nn.Conv2d(orig_in + proj_extra_channels, orig_out,
                                     kernel_size=ksize, stride=stride, padding=padding, bias=bias)
                # copy weights & bias
                with torch.no_grad():
                    # copy old kernel into first orig_in channels
                    new_conv.weight[:, :orig_in, :, :].copy_(conv_old.weight)
                    # zero extra channels
                    if proj_extra_channels > 0:
                        new_conv.weight[:, orig_in:, :, :].zero_()
                    if bias and conv_old.bias is not None:
                        new_conv.bias.copy_(conv_old.bias)
                # place new_conv into the same wrapper (preserve norm/activation)
                child[0] = new_conv
                seq[idx] = child
                return True
            # case B: child itself is a Conv2d (unlikely for recent torchvision, but handle)
            if isinstance(child, nn.Conv2d):
                conv_old = child
                orig_in = conv_old.in_channels
                orig_out = conv_old.out_channels
                ksize = conv_old.kernel_size
                stride = conv_old.stride
                padding = conv_old.padding
                bias = conv_old.bias is not None
                new_conv = nn.Conv2d(orig_in + proj_extra_channels, orig_out,
                                     kernel_size=ksize, stride=stride, padding=padding, bias=bias)
                with torch.no_grad():
                    new_conv.weight[:, :orig_in, :, :].copy_(conv_old.weight)
                    if proj_extra_channels > 0:
                        new_conv.weight[:, orig_in:, :, :].zero_()
                    if bias and conv_old.bias is not None:
                        new_conv.bias.copy_(conv_old.bias)
                seq[idx] = new_conv
                return True
        return False

    # 1) If rpn_head has attribute 'conv' and is a Sequential, try replacing inside it
    if hasattr(rpn_head, "conv"):
        conv_attr = rpn_head.conv
        # case: conv_attr is a Sequential (typical)
        if isinstance(conv_attr, nn.Sequential):
            replaced = try_replace_in_sequential(conv_attr)
            if replaced:
                return True

        # If conv_attr is some custom module, try to find and replace direct child convs
        # try named_children of conv_attr
        for name, child in conv_attr.named_children():
            if isinstance(child, nn.Sequential):
                replaced = try_replace_in_sequential(child)
                if replaced:
                    return True
            if isinstance(child, nn.Conv2d):
                # replace this conv directly on conv_attr
                conv_old = child
                orig_in = conv_old.in_channels
                orig_out = conv_old.out_channels
                ksize = conv_old.kernel_size
                stride = conv_old.stride
                padding = conv_old.padding
                bias = conv_old.bias is not None
                new_conv = nn.Conv2d(orig_in + proj_extra_channels, orig_out,
                                     kernel_size=ksize, stride=stride, padding=padding, bias=bias)
                with torch.no_grad():
                    new_conv.weight[:, :orig_in, :, :].copy_(conv_old.weight)
                    if proj_extra_channels > 0:
                        new_conv.weight[:, orig_in:, :, :].zero_()
                    if bias and conv_old.bias is not None:
                        new_conv.bias.copy_(conv_old.bias)
                setattr(conv_attr, name, new_conv)
                return True

    # 2) Fallback: scan rpn_head for first Conv2d and replace it directly at parent
    for parent in rpn_head.children():
        # we need to find the direct parent module / attribute name or index
        for cname, child in parent.named_children():
            if isinstance(child, nn.Conv2d):
                conv_old = child
                orig_in = conv_old.in_channels
                orig_out = conv_old.out_channels
                ksize = conv_old.kernel_size
                stride = conv_old.stride
                padding = conv_old.padding
                bias = conv_old.bias is not None
                new_conv = nn.Conv2d(orig_in + proj_extra_channels, orig_out,
                                     kernel_size=ksize, stride=stride, padding=padding, bias=bias)
                with torch.no_grad():
                    new_conv.weight[:, :orig_in, :, :].copy_(conv_old.weight)
                    if proj_extra_channels > 0:
                        new_conv.weight[:, orig_in:, :, :].zero_()
                    if bias and conv_old.bias is not None:
                        new_conv.bias.copy_(conv_old.bias)
                setattr(parent, cname, new_conv)
                return True

    return False


class AnomalyAwareMaskRCNN(nn.Module):
    """
    Mask R-CNN wrapper that fuses a per-image heatmap into the RPN by projecting
    the heatmap to each FPN level and concatenating (or you can change to 'mul').
    """
    def __init__(self, num_classes, proj_channels=1, pretrained=True):
        super().__init__()
        self.model = get_maskrcnn_model(num_classes=num_classes, pretrained=pretrained)
        self.proj_channels = proj_channels
        # per-level 1x1 convs (lazy-created and registered)
        self.hmap_projs = nn.ModuleDict()
        # Modify RPN head to include our heatmap-aware head
        _replace_first_conv_of_rpn_head_with_extra_in_channels(self.model.rpn.head, proj_extra_channels=self.proj_channels)

    def _prepare_features_for_rpn(self, features: OrderedDict, heatmaps_tensor: torch.Tensor, per_image_sizes):
        """
        features: OrderedDict(level_name -> tensor(B,C,H_l,W_l))
        heatmaps_tensor: (B,1,H_pad,W_pad) where each per-image heatmap is top-left aligned
                        (unpadded region sits at [:, :h_i, :w_i])
        per_image_sizes: list of (h_i, w_i) unpadded sizes from image_list.image_sizes
        Returns OrderedDict with same keys where each value is modified (concat or mul).
        """
        def downsample_integer_safe(cur_1ch, target_h, target_w, src_h, src_w):
            """
            cur_1ch: Tensor (1, src_h, src_w)
            Returns: Tensor (1, target_h, target_w)
            Uses avg_pool2d when integer stride is available and equal for H/W, else bilinear interpolate.
            """
            H_src, W_src = cur_1ch.shape[-2], cur_1ch.shape[-1]
            # If not matching the declared unpadded size, first resize to src_h,src_w with bilinear
            if (H_src, W_src) != (src_h, src_w):
                cur_1ch = F.interpolate(cur_1ch.unsqueeze(0), size=(src_h, src_w),
                                        mode='bilinear', align_corners=False).squeeze(0)
                H_src, W_src = src_h, src_w

            # compute integer stride relative to unpadded src
            stride_h = src_h // target_h if (src_h % target_h == 0) else None
            stride_w = src_w // target_w if (src_w % target_w == 0) else None

            if stride_h is not None and stride_w is not None and (stride_h == stride_w) and stride_h >= 1:
                # avg_pool2d expects (N,C,H,W)
                pooled = F.avg_pool2d(cur_1ch.unsqueeze(0), kernel_size=stride_h, stride=stride_h)
                return pooled.squeeze(0)  # (1, target_h, target_w)
            else:
                # fallback: bilinear
                return F.interpolate(cur_1ch.unsqueeze(0), size=(target_h, target_w),
                                    mode='bilinear', align_corners=False).squeeze(0)

        new_feats = OrderedDict()
        # features: OrderedDict with consistent keys/order -> preserve that
        for lvl, feat in features.items():
            B, C, H_l, W_l = feat.shape

            # lazy-create projection conv (register module if missing)
            if lvl not in self.hmap_projs:
                self.hmap_projs[lvl] = nn.Conv2d(1, self.proj_channels, kernel_size=1)
            proj = self.hmap_projs[lvl]
            # ensure proj params are on same device as feature (move once)
            if next(proj.parameters()).device != feat.device:
                proj = proj.to(feat.device)
                self.hmap_projs[lvl] = proj

            # Build per-image hm_up for this level, using integer-safe pooling when possible
            hm_up_list = []
            for i in range(B):
                # per_image_sizes provides unpadded size for image i
                h_i, w_i = per_image_sizes[i]
                # extract the unpadded region from the padded heatmaps_tensor
                # heatmaps_tensor has shape (B,1,H_pad,W_pad)
                # defensive clamp (in case h_i/w_i equals 0 or larger than padded dims)
                H_pad, W_pad = heatmaps_tensor.shape[-2], heatmaps_tensor.shape[-1]
                hi = min(max(1, h_i), H_pad)
                wi = min(max(1, w_i), W_pad)
                cur = heatmaps_tensor[i:i+1, :, :hi, :wi]  # shape (1,1,hi,wi)
                # remove channel dim for helper: want (1, hi, wi)
                cur_1ch = cur.squeeze(0)  # (1, hi, wi)

                # normalize numeric range if needed (optional; safe)
                if cur_1ch.max() > 1.5:
                    cur_1ch = (cur_1ch / 255.0).clamp(0.0, 1.0)
                else:
                    cur_1ch = cur_1ch.clamp(0.0, 1.0)

                # downsample to H_l x W_l using integer-safe helper
                hm_lvl_i = downsample_integer_safe(cur_1ch, H_l, W_l, src_h=h_i, src_w=w_i)  # (1, H_l, W_l)
                hm_up_list.append(hm_lvl_i)

            # stack per-image to (B,1,H_l,W_l) and move to same device as feat
            hm_up = torch.stack(hm_up_list, dim=0).to(feat.device, dtype=feat.dtype)

            # Apply 1x1 projection conv and fuse
            hm_proj = proj(hm_up)  # (B, proj_channels, H_l, W_l)

            new_feat = torch.cat([feat, hm_proj], dim=1)

            new_feats[lvl] = new_feat

        return new_feats

    def forward(self, images, heatmaps, targets=None):
        """
        images: tensor (B,3,H,W) or list of Tensors/PIL images
        heatmaps: iterable of per-image arrays/tensors or a single template array/tensor
        """
        was_tensor = isinstance(images, torch.Tensor)
        if was_tensor:
            if images.dim() != 4 or images.shape[1] != 3:
                raise ValueError("images tensor must have shape (B,3,H,W)")
            images_list = [img for img in images]
        else:
            images_list = images

        # 1) transform images -> ImageList
        image_list, _ = self.model.transform(images_list)
        image_tensors = image_list.tensors  # (B,3,H_pad,W_pad)
        device = image_tensors.device
        dtype = image_tensors.dtype
        B_img = image_tensors.shape[0]
        target_spatial = image_tensors.shape[-2:]  # (H_pad, W_pad)
        
        # 2) prepare heatmaps -> (B,1,H_pad,W_pad) by per-image resize & pad using unpadded sizes
        # Accept heatmaps as tensor or iterable
        if isinstance(heatmaps, torch.Tensor):
            hm = heatmaps
        else:
            # allow iterable of arrays/ tensors or single template
            try:
                hm = torch.stack([torch.from_numpy(h).float() if isinstance(h, np.ndarray) else h.float()
                                  for h in heatmaps], dim=0)
            except Exception:
                hm = torch.from_numpy(heatmaps).float() if isinstance(heatmaps, np.ndarray) else torch.tensor(heatmaps).float()

        # normalize dims to (B,1,H_src,W_src)
        if hm.dim() == 2:
            hm = hm.unsqueeze(0).unsqueeze(0)
        elif hm.dim() == 3:
            # (1,H,W) or (B,H,W)
            if hm.shape[0] == 1 and hm.shape[1] != 1:
                hm = hm.unsqueeze(1)  # (1,1,H,W)
            else:
                hm = hm.unsqueeze(1)  # (B,1,H,W)
        elif hm.dim() == 4:
            if hm.shape[1] != 1:
                hm = hm.mean(dim=1, keepdim=True)
        else:
            raise TypeError(f"Unsupported heatmaps tensor shape: {tuple(hm.shape)}")

        # replicate template if needed
        if hm.shape[0] == 1 and B_img > 1:
            hm = hm.expand(B_img, -1, -1, -1)

        # move heatmaps to device/dtype (float)
        hm = hm.to(device=device, dtype=dtype)

        # pad/resize per-image to match image_list.image_sizes then pad to (H_pad, W_pad)
        per_sizes = image_list.image_sizes  # list of (h_i, w_i)
        H_pad, W_pad = target_spatial
        hm_padded_list = []
        for i in range(B_img):
            cur = hm[i] if hm.shape[0] == B_img else hm[0]  # (1, H_src, W_src)
            # normalize range if needed
            if cur.max() > 1.5:
                cur = (cur / 255.0).clamp(0.0, 1.0)
            else:
                cur = cur.clamp(0.0, 1.0)

            h_i, w_i = per_sizes[i]
            cur_resized = F.interpolate(cur.unsqueeze(0), size=(h_i, w_i), mode='bilinear', align_corners=False).squeeze(0)
            # pad bottom/right to H_pad,W_pad
            pad_t = torch.zeros((1, H_pad, W_pad), dtype=cur_resized.dtype, device=cur_resized.device)
            pad_t[:, :h_i, :w_i] = cur_resized
            hm_padded_list.append(pad_t)

        heatmaps_tensor = torch.stack(hm_padded_list, dim=0).to(device=device, dtype=dtype)  # (B,1,H_pad,W_pad)

        # 3) backbone features (for ROI heads)
        features = self.model.backbone(image_tensors)  # OrderedDict of FPN features

        # 4) prepare modified features for RPN by fusing heatmap per-level
        # print("Image_tensors shape: (to check alignment with heatmap)")
        # print(image_tensors.shape)
        # print(heatmaps_tensor.shape)
        features_for_rpn = self._prepare_features_for_rpn(features, heatmaps_tensor, image_list.image_sizes)
        # for k, v in features.items():
        #     print("Before modifying:")
        #     print("Level: ", str(k))
        #     print("Shape: ", v.size())

        # for k, v in features_for_rpn.items():
        #     print("After modifying:")
        #     print("Level: ", str(k))
        #     print("Shape: ", v.size())
        
        # 5) call RPN with modified features_for_rpn
        proposals, proposal_losses = self.model.rpn(image_list, features_for_rpn, targets)

        # 6) call ROI heads with ORIGINAL features (unchanged)
        detections, detector_losses = self.model.roi_heads(features, proposals, image_list.image_sizes, targets)

        # 7) compute original_image_sizes robustly (height, width)
        if was_tensor:
            original_image_sizes = [tuple(img.shape[-2:]) for img in images_list]
        else:
            original_image_sizes = []
            for orig in images_list:
                if isinstance(orig, torch.Tensor):
                    original_image_sizes.append(tuple(orig.shape[-2:]))
                else:
                    w, h = orig.size
                    original_image_sizes.append((h, w))

        # 8) decide whether to postprocess: only if boxes appear in resized coords
        do_postprocess = True
        try:
            sample_boxes = None
            for det in detections:
                if "boxes" in det and getattr(det["boxes"], "numel", lambda: 0)() > 0:
                    sample_boxes = det["boxes"]
                    break
            if sample_boxes is None:
                do_postprocess = False
            else:
                bmax = float(sample_boxes.max().item())
                max_resized = max(h for (h, w) in image_list.image_sizes)
                eps = 1e-3
                do_postprocess = (bmax <= max_resized + eps)
        except Exception:
            do_postprocess = True

        if do_postprocess:
            detections = self.model.transform.postprocess(detections, image_list.image_sizes, original_image_sizes)
        # else: keep detections as-is (already in original coords)

        # 9) combine losses
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return losses if self.training else detections
