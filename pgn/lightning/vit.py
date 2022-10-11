from typing import Optional, List

import pytorch_lightning as pl
import torch
import torchmetrics
import torchvision.models
from clip.simple_tokenizer import SimpleTokenizer
from torch.nn.functional import cross_entropy
from pgn.lr_schedulers.cosine_with_warmup import CosineWithWarmup
from pgn.pgn_models.iip import IIP

from pgn.pgn_models.tlpgn import TLPGN


class VisionIDP(pl.LightningModule):
    eot_token = SimpleTokenizer().encoder["<|endoftext|>"]

    def __init__(
            self,
            optimizer: str,
            warmup_epochs: int,
            nr_of_classes: int,
            init_lr: Optional[float] = 40,
            entropy_loss_coeff: Optional[float] = 0,
            lr_scheduler: Optional[str] = 'cosine',
            epochs: Optional[int] = 150,
            pgn_settings: Optional[dict] = None,
            random_classifier: Optional[bool] = False,
            **kwargs,
    ) -> None:

        super().__init__()
        self.save_hyperparameters()
        self.vit_model = self._build_vision_model()
        self._freeze_components()
        self._create_metrics()
        self._build_pgn_module(pgn_settings)

    def _build_vision_model(self):
        vit_model = torchvision.models.vit_b_32(
            weights=torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1
        )
        if self.hparams.random_classifier:
            vit_model.heads = torch.nn.Sequential(
                torch.nn.Linear(768, self.hparams.nr_of_classes)
            )
        return vit_model

    def _build_pgn_module(self, pgn_settings):
        if not pgn_settings:
            self.pgn_module = None
        elif pgn_settings['prompt_mode'] == 'pgn':
            self.pgn_module = TLPGN(
                **pgn_settings
            )
        else:
            self.pgn_module = IIP(
                **pgn_settings
            )

    def _create_metrics(self):
        self.top5_accuracy = torchmetrics.Accuracy(
            top_k=5
        )
        self.top1_accuracy = torchmetrics.Accuracy(
            top_k=1
        )

    def forward(self, images):
        logits_per_image = self._encode_image(images)
        return logits_per_image

    def loss_function(self, logits, labels):
        cls_loss = cross_entropy(logits, labels)
        return cls_loss

    def configure_optimizers(self):
        lr = self.hparams.init_lr
        if self.hparams.optimizer == 'adam':
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=lr
            )
        else:
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=lr,
                momentum=0.9
            )
        if self.hparams.lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=self.hparams.epochs,
                eta_min=(lr / 1e2),
                verbose=True
            )
        else:
            scheduler = CosineWithWarmup(
                optimizer=optimizer,
                warmup_epochs=self.hparams.warmup_epochs,
                T_max=self.hparams.epochs,
                eta_min=(lr / 1e2)
            )
        return {'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                }}

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits_per_image = self(images)
        loss = self.loss_function(
            logits_per_image,
            labels,
        )
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits_per_image = self(images)
        loss = self.loss_function(
            logits_per_image,
            labels,
        )
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=True)
        self.top1_accuracy(logits_per_image, labels)
        self.top5_accuracy(logits_per_image, labels)

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits_per_image = self(images)
        self.top1_accuracy(logits_per_image, labels)
        self.top5_accuracy(logits_per_image, labels)

    def validation_epoch_end(self, outputs) -> None:
        self.log('val_top1_accuracy', self.top1_accuracy)
        self.log('val_top5_accuracy', self.top5_accuracy)

    def test_epoch_end(self, outputs) -> None:
        self.log('test_top1_accuracy', self.top1_accuracy)
        self.log('test_top5_accuracy', self.top5_accuracy)

    def _encode_image(self, images):
        if images.dim == 3:
            images = images.unsqueeze(0)

        if self.input_dependent_prompt:
            visual_context, mixture_logits = self.input_dependent_prompt(images)

            video_features = self._modified_visual_encode(images,
                                                          visual_context)
        else:
            video_features = self._modified_visual_encode(images)
            mixture_logits = None

        return video_features, mixture_logits

    def _modified_visual_encode(self, x, context=None):
        # Reshape and permute the input tensor
        x = self.vit_model._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.vit_model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        if context is not None:
            torch._assert(x.dim() == 3,
                          f"Expected (batch_size, seq_length, hidden_dim) got {x.shape}")
            x = x + self.vit_model.encoder.pos_embedding
            x = torch.cat(
                [x, context],
                dim=1
            )
            x = self.vit_model.encoder.ln(
                self.vit_model.encoder.layers(
                    self.vit_model.encoder.dropout(x)
                )
            )
        else:
            x = self.vit_model.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.vit_model.heads(x)

        return x

    def _freeze_components(self) -> None:
        for param in self.vit_model.parameters():
            param.requires_grad = False
        for param in self.vit_model.heads.parameters():
            param.requires_grad = True

    def on_fit_start(self) -> None:
        self.hparams['nr_of_classes'] = self.trainer.datamodule.nr_of_classes

    def on_test_start(self) -> None:
        self.hparams['nr_of_classes'] = self.trainer.datamodule.nr_of_classes
