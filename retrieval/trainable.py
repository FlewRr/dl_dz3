from typing import Any

import torch
import torchmetrics
from torch import nn
from torch.nn import TripletMarginLoss, CosineEmbeddingLoss

from retrieval.config import RetrievalConfig, TripletLossConfig, ContrastiveLossConfig
from retrieval.trainer import Trainable


class RetrievalTrainable(Trainable):
    def __init__(self, config: RetrievalConfig):
        self._config = config
        self._loss_name, self._loss = self._setup_loss()


    def _setup_loss(self):
        match self._config.trainer.loss:
            case TripletLossConfig():
                return "TripletLoss", TripletMarginLoss(margin=self._config.trainer.loss.similarity_margin)
            case ContrastiveLossConfig():
                return "ContrastiveLoss", CosineEmbeddingLoss(margin=self._config.trainer.loss.similarity_margin)
            case _:
                raise ValueError(f'Optimizer {self._config.optimizer} is not supported')

    def forward_pass(self, model: nn.Module, model_inputs) -> tuple[torch.Tensor, Any]:
        if self._loss_name == "TripletLoss":
            positive = model_inputs['positive'].to(self._config.device)
            negative = model_inputs['negative'].to(self._config.device)
            anchor = model_inputs['anchor'].to(self._config.device)

            vectors_positive = model(positive['input_ids'], positive['attention_mask'])
            vectors_negative = model(negative['input_ids'], negative['attention_mask'])
            vectors_anchor = model(anchor['input_ids'], anchor['attention_mask'])

            loss = self._loss(vectors_positive, vectors_negative, vectors_anchor)

            return loss, {"loss": loss.item()}

        elif self._loss_name == "ContrastiveLoss":
            labels = model_inputs["labels"].to(self._config.device)
            x1 = model_inputs["x1"].to(self._config.device)
            x2 = model_inputs["x2"].to(self._config.device)

            x1 = model(x1["input_ids"], x1["attention_masks"])
            x2 = model(x2["input_ids"], x2["attention_masks"])

            loss = self._loss(x1, x2, labels)

            return loss, {"loss": loss.item()}

    def create_metrics(self) -> dict[str, torchmetrics.Metric]:
        return {
            'loss': torchmetrics.MeanMetric()
        }

    def update_metrics(self, model_outputs, metrics: dict[str, torchmetrics.Metric]):
        metrics['loss'].update(model_outputs['loss'])