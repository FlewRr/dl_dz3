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
        match self._config.loss:
            case TripletLossConfig():
                return "TripletLoss", TripletMarginLoss(margin=self._config.loss.similarity_margin)
            case ContrastiveLossConfig():
                return "ContrastiveLoss", CosineEmbeddingLoss(margin=self._config.loss.similarity_margin)
            case _:
                raise ValueError(f'Optimizer {self._config.loss} is not supported')

    def forward_pass(self, model: nn.Module, model_inputs) -> tuple[torch.Tensor, Any]:
        if self._loss_name == "TripletLoss":
            positive = model_inputs['positive']["input_ids"].to(self._config.device), model_inputs['positive']["attention_mask"].to(self._config.device)
            negative = model_inputs['negative']["input_ids"].to(self._config.device), model_inputs['negative']["attention_mask"].to(self._config.device)
            anchor = model_inputs['anchor']["input_ids"].to(self._config.device), model_inputs['anchor']["attention_mask"].to(self._config.device)

            vectors_positive = model(*positive)
            vectors_negative = model(*negative)
            vectors_anchor = model(*anchor)

            loss = self._loss(vectors_positive, vectors_negative, vectors_anchor)

            return loss, {"loss": loss.item()}

        elif self._loss_name == "ContrastiveLoss":
            labels = model_inputs["labels"].to(self._config.device)
            x1 = model_inputs["x1"]["input_ids"].to(self._config.device), model_inputs["x1"]["attention_masks"].to(self._config.device)
            x2 = model_inputs["x2"]["input_ids"].to(self._config.device), model_inputs["x2"]["attention_mask"].to(self._config.device)

            x1 = model(*x1)
            x2 = model(*x2)

            loss = self._loss(x1, x2, labels)

            return loss, {"loss": loss.item()}

    def create_metrics(self) -> dict[str, torchmetrics.Metric]:
        return {
            'loss': torchmetrics.MeanMetric()
        }

    def update_metrics(self, model_outputs, metrics: dict[str, torchmetrics.Metric]):
        metrics['loss'].update(model_outputs['loss'])