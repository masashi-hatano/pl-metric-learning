from typing import List

import pytorch_lightning as pl
import torch
from pytorch_metric_learning import distances, losses, miners, reducers, regularizers

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
        results = {"loss": loss_emb}

        return results

    def training_epoch_end(self, outputs):
        """Compute embedding of training set at every epoch end.
        """
        metric_loss = torch.stack([tmp["loss"] for tmp in outputs]).mean()
        self.log("train_metric_loss", metric_loss, on_step=False)

        # logits, targets = self._aggregate_result(outputs, ["logits", "target"])
        # preds = torch.argmax(logits, dim=1)
        # acc = float(torch.sum(targets == preds)) / len(targets)
        self.log("train_loss", metric_loss)

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
        results = {
            "val_metric_loss": loss_emb,
            "embedding": embedding.detach(),
            "target": t.detach(),
        }

        return results

    def validation_epoch_end(self, outputs):
        """Log embedding and confusion matrix.

        Args:
            outputs (None): None.
        """
        metric_loss = torch.stack([tmp["val_metric_loss"] for tmp in outputs]).mean()
        self.log("validation_metric_loss", metric_loss.mean(), on_step=False)

        # logits, targets = self._aggregate_result(outputs, ["logits", "target"])
        # preds = torch.argmax(logits, dim=1)
        # acc = float(torch.sum(targets == preds)) / len(targets)
        self.log("val_loss", metric_loss)

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
        results = {
            "embedding": embedding.detach(),
            "target": t.detach(),
        }

        return results

    # def test_epoch_end(self, outputs: dict) -> None:
    #     """At the end of test, this calculates mahalanobis/cos distance.

    #     Args:
    #         outputs (None): None
    #     """

    #     # Moving tensor by .to() is prohibited for lightning
    #     self.cluster_centre = self.cluster_centre.new_tensor(
    #         self.cluster_centre, device=self.device
    #     )
    #     self.cluster_cov = self.cluster_cov.new_tensor(
    #         self.cluster_cov, device=self.device
    #     )

    #     embeddings, logits, targets = self._aggregate_result(
    #         outputs, ["embedding", "logits", "target"]
    #     )

    #     preds = torch.argmax(logits, dim=1)
    #     acc = float(torch.sum(targets == preds)) / len(targets)

    #     target_centre = self.cluster_centre[self.object_cluster_idx]
    #     pre_act_target_centre = self.cluster_centre[self.object_cluster_idx - 1]

    #     if not self.use_ensemble_distance:
    #         cos_dis = log_helpers.log_cosine_distance(
    #             points=embeddings,
    #             centre=target_centre.unsqueeze(0),
    #             centres=self.cluster_centre,
    #             targets=targets,
    #             object_cluster_idx=self.object_cluster_idx,
    #             offset=5,
    #         )
    #     else:
    #         cos_dis = log_helpers.log_ensemble_cosine_distance(points=embeddings, centre=target_centre.unsqueeze(0),
    #                                                            centres=self.cluster_centre, targets=targets,
    #                                                            object_cluster_idx=self.object_cluster_idx,
    #                                                            pre_act_center=pre_act_target_centre.unsqueeze(0),
    #                                                            offset=5, logits=logits)

    #     mah_dis = log_helpers.log_mahalanovis(
    #         embeddings,
    #         self.cluster_centre,
    #         covariances=self.cluster_cov,
    #         targets=targets,
    #         object_cluster_idx=self.object_cluster_idx,
    #         offset=5,
    #     )

    #     fp_score, tp_score, ftp_score = self.future_metrics(torch.from_numpy(cos_dis).clone(), targets)

    #     log_helpers.log_embedding_plot(
    #         self,
    #         self.num_classes,
    #         self.hparams.class_names,
    #         outputs,
    #         title="Test",
    #         is_wandb=self.hparams.use_wandb,
    #         pca_obj=self.reductor,
    #     )

    #     self.log("test score", acc)
    #     self.log("fp_score", fp_score)
    #     self.log("tp_score", tp_score)
    #     self.log("ftp_score", ftp_score)


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

    # def _log_input(self, query: torch.Tensor) -> None:
    #     # rgb_imgs = query[:, 0, :3, :, :][:, [2, 1, 0]]
    #     rgb_imgs = torch.unsqueeze(query[:, 0, 0, :, :], 1)

    #     # renormlize the RGB frames again so that it can be displayed more properly
    #     rgb_imgs = (rgb_imgs - rgb_imgs.min()) / (rgb_imgs.max() - rgb_imgs.min())

    #     # rgb_imgs_numpy_sample = rgb_imgs[0, ...].squeeze().detach().cpu().numpy()
    #     # np.save("/home/kanhua/rgb_sample.npy", rgb_imgs_numpy_sample)

    #     if self.hparams.use_wandb:
    #         self.logger.experiment.log(
    #             {
    #                 "train_rgb_images": Image(rgb_imgs),
    #                 "global_step": self.global_step,
    #             }
    #         )
    #     else:
    #         self.logger.experiment.add_images(
    #             "train_rgb_images", rgb_imgs, self.global_step
    #         )
    
    def prep_backborn(self, backborn, pretrained, hparams):
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