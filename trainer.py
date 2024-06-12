import os
import torch

from transformers import Trainer
from typing import Optional


class Phi3VTrainer(Trainer):

    def _save_checkpoint(self, model, trial, metrics=None):
        super(Phi3VTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
            super(Phi3VTrainer, self)._save(output_dir, state_dict)

    # def training_step(self, model, inputs):
    #     for name, param in model.named_parameters():
    #         if 'vision_model' in name:
    #             print(f"Training parameter {name}")
            
    #         elif 'img_projection' in name:
    #             print(f"Training parameter {name}")
    #     return super().training_step(model, inputs)
            