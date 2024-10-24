
from typing import Any

import lightning as L
import torch

from pytorch_optimizer import create_optimizer
from pytorch_lightning.utilities.types import OptimizerLRScheduler

# define the LightningModule
class TrainModule(L.LightningModule):
    
    def __init__(self,
                 model : torch.nn.Module,
                 optimizer_params,
                 loss_functions : list[torch.nn.Module],
                 loss_weights : list[float]|None):
        super().__init__()
        
        self.model = model
        
        assert len(loss_functions) == len(loss_weights), "Loss functions and weights must be the same length"
        
        self.loss_functions = loss_functions
        self.loss_weights = loss_weights
        
        self.optimizer_params = optimizer_params
        
        self.save_hyperparameters(ignore=['model'])

    def training_step(self, batch, batch_idx) -> torch.Tensor:

        outputs = self.model(batch)
        
        losses = self._get_losses(outputs, batch)
        self._log_losses(losses, "train")
        
        if batch_idx == 0:
            self._log_predictions(batch, outputs, "train")
            
        return torch.sum(losses)
    
    
    def validation_step(self, batch : tuple, batch_idx) -> torch.Tensor:

        outputs = self.model(...)

        losses = self._get_losses(outputs, batch)
        
        if batch_idx == 0:
            self._log_predictions(batch, outputs, "val")

        return torch.sum(losses)

    def _get_sarphiv_schedular(self, optimizer):
        """
        Sequential Annealing Restarting Periodical Hybrid Initialized Velocity schedular
        """

        # NOTE: Must instantiate cosine scheduler first,
        #  because super scheduler mutates the initial learning rate.
        lr_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=self.optimizer_params.lr_half_period,
            T_mult=self.optimizer_params.lr_mult_period,
            eta_min=self.optimizer_params.lr_min
        )
        lr_super = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=self.optimizer_params.lr_warmup_max,
            total_steps=self.optimizer_params.lr_warmup_period,
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[lr_super, lr_cosine], # type: ignore
            milestones=[self.optimizer_params.lr_warmup_period],
        )
        
        return lr_scheduler

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = create_optimizer(
            self.model, # type: ignore
            self.optimizer_params.optimizer,
            lr=self.optimizer_params.lr_max,
            weight_decay=self.optimizer_params.weight_decay,
            use_lookahead=True,
            use_gc=True,
            eps=1e-6
        )

        lr_scheduler = self._get_sarphiv_schedular(optimizer)

        return { # type: ignore
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "lr"
            }
        }
    
    def on_save_checkpoint(self, checkpoint: dict[str, Any]):
        super().on_save_checkpoint(checkpoint)
        
        checkpoint['name'] = self.model.NAME
    
    def _get_losses(self, outputs : list[torch.Tensor], batch : list[torch.Tensor]) -> list[torch.Tensor]:
        
        losses = []
        
        for weight, loss_func in zip(self.loss_weights, self.loss_functions):
            loss = loss_func(outputs, batch) * weight
            losses.append(loss)
            
        return losses
        
    def _log_losses(self, losses : dict[str, torch.Tensor], subdir : str):
        """
        Log the losses to the logger
        """
        for key, value in losses.items():
            self.log(f"{subdir}/{key}", value, on_epoch=True, on_step=False)

    def _log_predictions(self, batch, outputs, prefix : str):
        """
        Implement logging for predictions
        """
        
        return
