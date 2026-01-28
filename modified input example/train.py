# train.py (with LR scheduler + EarlyStopping)
import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from tqdm import tqdm
import copy
from datetime import datetime

from backbone.maskrcnn import AnomalyAwareMaskRCNN
# from losses import (
#     discriminator_loss,
#     generator_loss_from_logits,
#     cycle_consistency_loss,
#     identity_loss,
# )
from eval.utils import grad_norm
from data import * 
import pandas as pd

# -------------------------
# EarlyStopping helper
# -------------------------
class EarlyStopping:
    """
    Simple early stopping that monitors a metric and stops when it doesn't improve.
    mode: 'min' (lower is better) or 'max' (higher is better)
    """
    def __init__(self, patience=10, delta=1e-4, mode='min', verbose=False, save_best=False, save_dir='best_models'):
        self.patience = int(patience)
        self.delta = float(delta)
        self.mode = mode
        self.verbose = verbose
        self.save_best = save_best
        self.save_dir = save_dir
        self.best_score = None
        self.num_bad_epochs = 0
        self.best_epoch = None
        if self.save_best:
            os.makedirs(self.save_dir, exist_ok=True)

    def is_improvement(self, current):
        if self.best_score is None:
            return True
        if self.mode == 'min':
            return current < (self.best_score - self.delta)
        else:
            return current > (self.best_score + self.delta)

    def step(self, current, epoch=None, trainer_obj: object = None, scheduler: dict = {}):
        """
        current: scalar metric value (float)
        trainer_obj: MaskRCNNTrainer instance (optional) to save models when improved
        returns: (should_stop: bool, improved: bool)
        """
        improved = False
        if self.is_improvement(current):
            improved = True
            self.best_score = float(current)
            self.num_bad_epochs = 0
            self.best_epoch = int(epoch) if epoch is not None else None
            if self.save_best and trainer_obj is not None:
                # Save generator and discriminator state dicts with epoch and metric in filename
                prefix = f"epoch{epoch:03d}_{self.mode}_{current:.6f}"
                ckpt = {
                    "save_dir": self.save_dir,
                    "model_state_dict": trainer_obj.maskrcnn_mod.state_dict(),
                    "optim_state_dict": trainer_obj.maskrcnn_mod_opt.state_dict(),
                    "scheduler_state_dicts": {k: v.state_dict() for k, v in scheduler.items()}
                }
                torch.save(ckpt, os.path.join(self.save_dir, f"best_epoch_{epoch}.pth"))
                if self.verbose:
                    print(f"[EarlyStopping] Improved metric -> saved models to {self.save_dir} (epoch {epoch})")
        else:
            self.num_bad_epochs += 1

        should_stop = self.num_bad_epochs >= self.patience
        return should_stop, improved

# -------------------------
# MaskRCNN trainer with scheduler + early stopping
# -------------------------
class MaskRCNNTrainer:
    def __init__(
        self,
        device: torch.device = None,
        lr: float = 0.005,     
        momentum: float = 0.9,
        weight_decay: float = 0.0005, 
        # scheduler / early stopping arguments (defaults can be tuned)
        lr_decay_start: int = 50,               # epoch after which linear decay starts
        early_stopping_patience: int = 20,
        early_stopping_delta: float = 1e-4,
        early_stopping_monitor: str = "epoch_train_loss",
        early_stopping_mode: str = "min",
        early_stopping_save_best: bool = False,
        scheduler_enabled: bool = True,
        num_classes: int = 2,
        use_pretrained: bool = True
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model confs
        assert num_classes >= 2, "num_classes should include background (>=2)"
        self.maskrcnn_mod = AnomalyAwareMaskRCNN(
            num_classes=num_classes,
            #proj_channels=1,
            pretrained = use_pretrained
        )
        
        # Note: original code used lr*100 for some optimizers; keep that mapping but user can pass lr accordingly
        self.maskrcnn_mod_opt = torch.optim.SGD(
            self.maskrcnn_mod.parameters(), 
            lr=lr, 
            momentum = momentum,
            weight_decay=weight_decay
        )

        # Summaries
        self.record = []

        # Scheduler & early stopping config
        self.scheduler_enabled = bool(scheduler_enabled)
        self.lr_decay_start = int(lr_decay_start)
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            delta=early_stopping_delta,
            mode=early_stopping_mode,
            verbose=True,
            save_best=early_stopping_save_best,
            save_dir="exported_models/best_models",
        )
        self.early_stopping_monitor = early_stopping_monitor
        self.early_stopping_mode = early_stopping_mode

        # track schedulers (created at train start)
        self._schedulers = None

    def _make_linear_lr_scheduler(self, opt, total_epochs, decay_start):
        """
        returns a LambdaLR that keeps lr constant for decay_start epochs and
        linearly decays it to zero over (total_epochs - decay_start) epochs.
        """
        def lambda_rule(epoch):
            if epoch < decay_start:
                return 1.0
            else:
                # linear decay
                return max(0.0, 1.0 - float(epoch - decay_start) / float(max(1, total_epochs - decay_start)))
        return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda_rule)

    def train(self, trainloader, epochs=100, print_every=10):
        # set train mode
        self.maskrcnn_mod.train()
        self.maskrcnn_mod.to(self.device)

        # create schedulers now that we know epochs
        if self.scheduler_enabled:
            self._schedulers = {
                "maskrcnn": self._make_linear_lr_scheduler(self.maskrcnn_mod_opt, total_epochs=epochs, decay_start=self.lr_decay_start)
            }
        else:
            self._schedulers = {}

        for epoch in range(epochs):
            loop = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch}", unit="batch")
            running_total = 0.0
            for b, (images, targets, heatmaps) in loop:
                self.maskrcnn_mod_opt.zero_grad()
                images = [img.to(self.device) for img in images]
                heatmaps = [hm.to(self.device) for hm in heatmaps]
                targets = [
                    {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()
                    } 
                    for t in targets
                ]
                loss_dict = self.maskrcnn_mod(images, heatmaps, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                losses.backward()

                # check gradients (before optimizer.step())
                maskrcnn_grad_norm = grad_norm(self.maskrcnn_mod.parameters())

                self.maskrcnn_mod_opt.step()

                batch_loss = float(losses.item())
                running_total += batch_loss
                running_average = running_total / (b+1)
                # -------------------------
                # Diagnostics & recording (per-batch)
                # -------------------------
                current_lr = self.maskrcnn_mod_opt.param_groups[0]['lr']
                batch_report = {
                    "batch_loss": f"{batch_loss:.4f}",
                    "running_avg": f"{running_average:.4f}",
                    "grad_norm": f"{maskrcnn_grad_norm:.4f}",
                    "lr": f"{current_lr:.6f}"
                }

                # Logging to tqdm with compact view
                if b % print_every == 0:
                    loop.set_postfix(batch_report)

            # end of epoch: persist per-batch averaged record so far (safe to crash)
            epoch_report = copy.deepcopy(batch_report)
            epoch_report['timestamp'] = datetime.utcnow().isoformat()
            epoch_report['epoch'] = int(epoch)
            epoch_report['epoch_train_loss'] = running_total/len(trainloader)
            self.record.append(epoch_report)

            try:
                record_df = pd.DataFrame(self.record)
                record_df.to_csv("train_record.csv", index=False)
            except Exception as e:
                print("Warning: failed to write train_record.csv:", e)

            # Step schedulers (if enabled)
            if self.scheduler_enabled and self._schedulers:
                for s in self._schedulers.values():
                    s.step()

            # Early stopping check (monitor a chosen metric from report)
            monitor_key = self.early_stopping_monitor
            if monitor_key in epoch_report:
                current = float(epoch_report[monitor_key])
                should_stop, improved = self.early_stopping.step(current, epoch=epoch, trainer_obj=self, scheduler = self._schedulers)
                if self.early_stopping.verbose:
                    print(f"[Epoch {epoch}] Monitor {monitor_key} = {current:.6f} | improved={improved} | bad_epochs={self.early_stopping.num_bad_epochs}/{self.early_stopping.patience}")
                if should_stop:
                    print(f"[EarlyStopping] stopping training at epoch {epoch} (no improvement for {self.early_stopping.patience} epochs)")
                    break
            else:
                # metric not present (unlikely) -> continue
                if self.early_stopping.verbose:
                    print(f"[EarlyStopping] monitor key '{monitor_key}' not in epoch metrics; skipping early-stopping evaluation.")

        print("Training finished.")
