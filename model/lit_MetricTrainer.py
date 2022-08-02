from typing import List
import os

import pytorch_lightning as pl
import torch
from torch import nn
from pytorch_metric_learning import distances, losses, miners, reducers, regularizers
import umap
import matplotlib.pyplot as plt 

from .ResNet import resnet18, resnet34, resnet50, resnet101, resnet152


class MetricTrainer(pl.LightningModule):
    def __init__(
            self,
            params,
            backborn="resnet50",
            pretrained=False,
            loss="ArcFace",
            regularize_embedder=True,
            lr=1e-3,
            lr_decay_freq=10,
            weight_decay=1e-5,
            class_names=[
                "airplane",
                "automobile",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck",
            ],
    ) -> None:
        super(MetricTrainer, self).__init__()
        self.save_hyperparameters()
        self.embedding_size = params["out_features"]
        self.num_classes = len(class_names)
        self.embedder = self.prep_backborn(backborn, pretrained, params)
        self._configure_metric_loss()

    def lossfun_emb(self, embedding: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.hparams.loss == "Triplet":
            dist_mat = self.miner(embedding, t)
            return self.embedding_loss(embedding, t, dist_mat)

        return self.embedding_loss(embedding, t)

    def get_logits(self, embedding):
        return self.embedding_loss.get_logits(embedding)

    def configure_optimizers(self):
        """Get a optimizer for trunk and classifier.

        Returns:
            [Tuple]: Adam instance and scheduler for optimzer.
        """
        optimizer_emb = torch.optim.Adam(
            self.embedder.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        step_lr_emb = torch.optim.lr_scheduler.StepLR(
            optimizer_emb, step_size=self.hparams.lr_decay_freq, gamma=0.7
        )
        return [optimizer_emb], [step_lr_emb]

    def training_step(self, batch, batch_idx):
        """Training step, called twice per mini-btach.
        If the metric loss function needs to be optimzed, optimization is done here.

        Args:
            batch (Tensor): mini-batch.
            batch_idx (int): mini-batch index.
        Returns:
            Results: Dict of tensors
        """
        x, t = batch
        t = t.view(-1)  # (batch_size, 1) -> (batch_size)
        
        embedding = self.embedder(x)
        loss_emb = self.lossfun_emb(embedding, t)
        logits = self.get_logits(embedding)
        results = {"loss": loss_emb, "logits": logits.detach(), "target": t.detach()}

        return results

    def training_epoch_end(self, outputs):
        """Compute embedding of training set at every epoch end.
        """
        metric_loss = torch.stack([tmp["loss"] for tmp in outputs]).mean()
        self.log("train_metric_loss", metric_loss, on_step=False)

        logits, targets = self._aggregate_result(outputs, ["logits", "target"])
        preds = torch.argmax(logits, dim=1)
        acc = float(torch.sum(targets == preds)) / len(targets)
        self.log("train_acc", acc)

    def validation_step(self, batch, batch_idx):
        """Validation step, called once per mini-batch.

        Args:
            batch (Tensor): mini-batch.
            batch_idx (int): index of mini-batch.

        Returns:
            Dict: Dict of lists of metrics.
        """
        x, t = batch
        t = t.view(-1)
        embedding = self.embedder(x)
        loss_emb = self.lossfun_emb(embedding, t)
        logits = self.get_logits(embedding)
        results = {
            "val_metric_loss": loss_emb, 
            "embedding":embedding.detach(), 
            "logits": logits.detach(), 
            "target": t.detach()
        }

        return results

    def validation_epoch_end(self, outputs):
        """Log embedding and confusion matrix.

        Args:
            outputs (None): None.
        """
        metric_loss = torch.stack([tmp["val_metric_loss"] for tmp in outputs]).mean()
        self.log("validation_metric_loss", metric_loss.mean(), on_step=False)

        embedding, logits, targets = self._aggregate_result(outputs, ["embedding", "logits", "target"])
        preds = torch.argmax(logits, dim=1)
        acc = float(torch.sum(targets == preds)) / len(targets)
        self.log("val_acc", acc)

        if self.global_step:
            self.embedding_plot(embedding, targets)

    def test_step(self, batch, batch_idx):
        """Return outputs of forward path

        Args:
            batch (Tensor): mini-batch not from m-per-sampler
            batch_idx (int): batch index

        Returns:
            Dict: dictionary of outputs
        """
        x, t = batch
        t = t.view(-1)
        embedding = self.embedder(x)
        loss_emb = self.lossfun_emb(embedding, t)
        logits = self.get_logits(embedding)
        results = {
            "test_metric_loss": loss_emb, 
            "embedding":embedding.detach(), 
            "logits": logits.detach(), 
            "target": t.detach()
        }

        return results

    def test_epoch_end(self, outputs: dict) -> None:
        """At the end of test, this calculates mahalanobis/cos distance.

        Args:
            outputs (None): None
        """
        metric_loss = torch.stack([tmp["test_metric_loss"] for tmp in outputs]).mean()
        self.log("test_metric_loss", metric_loss.mean(), on_step=False)

        embedding, logits, targets = self._aggregate_result(outputs, ["embedding", "logits", "target"])
        preds = torch.argmax(logits, dim=1)
        acc = float(torch.sum(targets == preds)) / len(targets)
        self.log("test_acc", acc)

        if self.global_step:
            self.embedding_plot(embedding, targets)

    def _configure_metric_loss(self) -> None:
        """Configure metric regularizer and other tunable parameters for chosen loss function."""
        if self.hparams.loss == "Triplet":
            self.need_loss_optimize = False
            distance = distances.CosineSimilarity()
            self.reducer = reducers.ThresholdReducer(0)
            self.miner = miners.TripletMarginMiner(margin=0.2, distance=distance)
            if self.hparams.regularize_embedder:
                self.regularizer = regularizers.RegularFaceRegularizer()
                self.embedding_loss = losses.TripletMarginLoss(
                    margin=0.2,
                    reducer=self.reducer,
                    distance=distance,
                    embedding_regularizer=self.regularizer,
                )
            else:
                self.embedding_loss = losses.TripletMarginLoss(
                    margin=0.2,
                    reducer=self.reducer,
                    distance=distance,
                )

        elif self.hparams.loss == "SoftTriple":
            self.need_loss_optimize = True
            if self.hparams.regularize_embedder:
                self.regularizer = regularizers.LpRegularizer()
                self.embedding_loss = losses.SoftTripleLoss(
                    num_classes=self.num_classes,
                    embedding_size=self.embedding_size,
                    weight_regularizer=self.regularizer,
                )
            else:
                self.embedding_loss = losses.SoftTripleLoss(
                    num_classes=self.num_classes,
                    embedding_size=self.embedding_size,
                )

        elif self.hparams.loss == "ArcFace":
            self.need_loss_optimize = True
            if self.hparams.regularize_embedder:
                self.regularizer = regularizers.RegularFaceRegularizer
                self.embedding_loss = losses.ArcFaceLoss(
                    num_classes=self.num_classes,
                    embedding_size=self.embedding_size,
                    weight_regularizer=self.regularizer,
                )
            else:
                self.embedding_loss = losses.ArcFaceLoss(
                    num_classes=self.num_classes,
                    embedding_size=self.embedding_size,
                )

        else:
            raise Exception("Loss options: Triplet, SoftTriple, or ArcFace")

        if self.need_loss_optimize:
            self.optimizer_loss = torch.optim.Adam(
                self.embedding_loss.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
            )

    def _aggregate_result(self, out_list: dict, keys: List[str]) -> List[torch.Tensor]:
        rtn = []
        for key in keys:
            rtn.append(torch.cat([tmp[key] for tmp in out_list]))

        return rtn

    def embedding_plot(self, embedding, labels):
        mapper = umap.UMAP(random_state=0)
        embedding = mapper.fit_transform(embedding.cpu())

        os.makedirs("./embedding_plot", exist_ok=True)

        plt.figure(figsize=(13, 7))
        plt.scatter(embedding[:, 0], embedding[:, 1],
                    c=labels.cpu(), cmap='jet',
                    s=15, alpha=0.5)
        plt.axis('off')
        plt.colorbar()
        plt.savefig(f'embedding_plot/{self.current_epoch}.png',dpi=600)
        plt.close()
    
    def _backborn(self, backborn, pretrained, hparams):
        if backborn=="resnet18":
            return resnet18(pretrained, **hparams)
        elif backborn=="resnet34":
            return resnet34(pretrained, **hparams)
        elif backborn=="resnet50":
            return resnet50(pretrained, **hparams)
        elif backborn=="resnet101":
            return resnet101(pretrained, **hparams)
        elif backborn=="resnet152":
            return resnet152(pretrained, **hparams)
        else:
            raise Exception("Backborn option: resnet18, resnet34, resnet50, resnet101, or resnet152")
    
    def prep_backborn(self, backborn, pretrained, hparams):
        model = self._backborn(backborn, pretrained, hparams)
        if pretrained:
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features=in_features, out_features=self.embedding_size)
        return model