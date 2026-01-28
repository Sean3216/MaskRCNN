# losses.py
import torch
from torch import nn
from torch.nn import functional as F

# --- stable BCE-with-logits for adversarial losses (default) --------------
bce_logits = nn.BCEWithLogitsLoss(reduction="mean")

# --- LSGAN (least-squares) losses ----------------------------------------
mse_loss = nn.MSELoss(reduction="mean")

def discriminator_loss_bce(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """
    Standard discriminator loss using BCEWithLogits:
      L_D = BCE(logits_real, 1) + BCE(logits_fake, 0)
    logits_* are *raw* discriminator outputs (not passed through sigmoid).
    """
    ones = torch.ones_like(logits_real)
    zeros = torch.zeros_like(logits_fake)
    loss_real = bce_logits(logits_real, ones)
    loss_fake = bce_logits(logits_fake, zeros)
    return loss_real + loss_fake


def generator_loss_bce_from_logits(logits_fake: torch.Tensor) -> torch.Tensor:
    """
    Generator adversarial objective (BCE): encourage D(fake) -> 1
    """
    ones = torch.ones_like(logits_fake)
    return bce_logits(logits_fake, ones)


def discriminator_loss_lsgan(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """
    Least-squares discriminator loss (LSGAN):
      L_D = 0.5 * ( (D(real)-1)^2 + (D(fake)-0)^2 )
    We omit the 0.5 constant (it just scales gradients); keep consistent with MSELoss.
    """
    ones = torch.ones_like(logits_real)
    zeros = torch.zeros_like(logits_fake)
    return mse_loss(logits_real, ones) + mse_loss(logits_fake, zeros)


def generator_loss_lsgan_from_logits(logits_fake: torch.Tensor) -> torch.Tensor:
    """
    Least-squares generator objective: push D(fake) -> 1
    """
    ones = torch.ones_like(logits_fake)
    return mse_loss(logits_fake, ones)


# --- Cycle consistency (pixel L1) ----------------------------------------
def cycle_consistency_loss(reconstructed_norm: torch.Tensor,
                           input_norm: torch.Tensor,
                           reconstructed_abnl: torch.Tensor,
                           input_abnl: torch.Tensor) -> torch.Tensor:
    """
    Sum of L1 losses for both cycles:
      L_cyc = ||F(G(x)) - x||_1 + ||G(F(y)) - y||_1
    Returns a scalar tensor (mean reduction).
    """
    l1 = F.l1_loss(reconstructed_norm, input_norm, reduction="mean")
    l2 = F.l1_loss(reconstructed_abnl, input_abnl, reduction="mean")
    return l1 + l2


# --- Identity loss (optional; used in CycleGAN when preserving color) ---
def identity_loss(generator_apply, target_images: torch.Tensor) -> torch.Tensor:
    """
    Apply generator to images from target domain and penalize deviation:
      L_id = ||G(y) - y||_1
    Accepts either a callable generator (nn.Module) or precomputed output tensor.
    If `generator_apply` is a tensor, it is treated as G(y) already computed.
    """
    if isinstance(generator_apply, torch.Tensor):
        out = generator_apply
    else:
        out = generator_apply(target_images)
    return F.l1_loss(out, target_images, reduction="mean")


# --- Helpers for label smoothing / soft labels --------------------------------
def smooth_labels(tensor: torch.Tensor, offset: float = 0.1, positive: bool = True):
    """
    Convert hard 1/0 labels to smoothed labels.
    If positive=True: smooth ones to uniform range [1 - offset, 1]
    If positive=False: smooth zeros to [0, offset]
    Expects input tensor shape for target; returns tensor with same shape & device.
    Usage:
      targets = torch.ones_like(logits)
      targets = smooth_labels(targets, offset=0.1, positive=True)
    """
    if positive:
        return tensor * (1.0 - offset) + offset * torch.rand_like(tensor) * 0.0 + (offset - offset * 0.0)  # returns ones*(1-offset)+offset
    else:
        return tensor * offset  # zeros stay near 0 (this function keeps shape/device)


# --- Convenience wrapper to switch easily between BCE and LSGAN ------------
# You can pick `use_lsgan = True` in your training script and call these.
def discriminator_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor, use_lsgan: bool = False) -> torch.Tensor:
    return discriminator_loss_lsgan(logits_real, logits_fake) if use_lsgan else discriminator_loss_bce(logits_real, logits_fake)


def generator_loss_from_logits(logits_fake: torch.Tensor, use_lsgan: bool = False) -> torch.Tensor:
    return generator_loss_lsgan_from_logits(logits_fake) if use_lsgan else generator_loss_bce_from_logits(logits_fake)
