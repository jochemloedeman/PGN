import random
from typing import Optional, List

import pytorch_lightning as pl
import torch
import torchmetrics
from clip.simple_tokenizer import SimpleTokenizer
from torch.nn.functional import cross_entropy
from pgn.lr_schedulers.cosine_with_warmup import CosineWithWarmup
from pgn.pgn_models.iip import IIP
from pgn.pgn_models.tlpgn import TLPGN

from pgn.pgn_models.vision_transformer import DINOHead


class DinoPGN(pl.LightningModule):
    eot_token = SimpleTokenizer().encoder["<|endoftext|>"]

    def __init__(
            self,
            optimizer: str,
            warmup_epochs: int,
            nr_of_classes: int,
            ckpt_path: str,
            init_lr: Optional[float] = 40,
            entropy_loss_coeff: Optional[float] = 0,
            lr_scheduler: Optional[str] = 'cosine',
            epochs: Optional[int] = 150,
            pgn_settings: Optional[dict] = None,
            random_classifier: Optional[bool] = False
    ) -> None:

        super().__init__()
        self.save_hyperparameters()
        self.dino_vit, self.dino_head = self._build_vision_model()
        self._freeze_components()
        self._create_metrics()
        self._build_pgn_module(pgn_settings)

    def _build_vision_model(self):
        vits16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        state_dict = torch.load(self.hparams.ckpt_path)['student']
        state_dict = {k: v for k, v in state_dict.items() if k.startswith('module.head.')}
        state_dict = {k.replace("module.head.", ""): v for k, v in state_dict.items()}
        head = DINOHead(in_dim=384, out_dim=65536)
        head.load_state_dict(state_dict)
        new_ll = torch.nn.Linear(256, self.hparams.nr_of_classes, bias=False)
        if not self.hparams.random_classifier:
            output_indices = random.sample(list(range(65536)),
                                           k=self.hparams.nr_of_classes)
            new_ll.weight = torch.nn.Parameter(head.last_layer.weight[output_indices])
        head.last_layer = new_ll
        return vits16, head

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

        if self.pgn_module:
            visual_context = self.pgn_module(images)

            video_features = self._modified_visual_encode(images,
                                                          visual_context)
        else:
            video_features = self._modified_visual_encode(images)

        video_features = self.dino_head(video_features)
        return video_features

    def _modified_visual_encode(self, x, context=None):
        x = self.dino_vit.prepare_tokens(x)
        if context is not None:
            x = torch.cat(
                [x, context],
                dim=1
            )
        for blk in self.dino_vit.blocks:
            x = blk(x)
        x = self.dino_vit.norm(x)
        return x[:, 0]

    def _freeze_components(self) -> None:
        for param in self.dino_vit.parameters():
            param.requires_grad = False
        for param in self.dino_head.parameters():
            param.requires_grad = False

    def on_fit_start(self) -> None:
        self.hparams['nr_of_classes'] = self.trainer.datamodule.nr_of_classes

    def on_test_start(self) -> None:
        self.hparams['nr_of_classes'] = self.trainer.datamodule.nr_of_classes
