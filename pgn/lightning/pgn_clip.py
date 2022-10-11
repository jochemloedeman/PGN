from typing import Optional, List

import pytorch_lightning as pl
import torch
import torchmetrics
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer
from torch.nn.functional import cross_entropy
from pgn.lr_schedulers.cosine_with_warmup import CosineWithWarmup
from pgn.pgn_models.iip import IIP

from pgn.pgn_models.tlpgn import TLPGN


class PGNCLIP(pl.LightningModule):
    eot_token = SimpleTokenizer().encoder["<|endoftext|>"]

    def __init__(
            self,
            clip_architecture: Optional[str] = 'ViT-B/32',
            optimizer: Optional[str] = 'sgd',
            warmup_epochs: Optional[int] = 50,
            init_lr: Optional[float] = 40,
            entropy_loss_coeff: Optional[float] = 0,
            lr_scheduler: Optional[str] = 'cosine',
            epochs: Optional[int] = 150,
            pgn_settings: Optional[dict] = None,
            disable_loggers: Optional[bool] = False,
    ) -> None:

        super().__init__()
        self.save_hyperparameters()
        self.clip_model, _ = clip.load(clip_architecture, device='cpu')
        self._freeze_components()
        self._create_metrics()
        self._build_pgn_module(pgn_settings)

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
        image_features = self._encode_image(images)
        image_features = image_features / image_features.norm(dim=1,
                                                              keepdim=True)
        text_features = self._encode_text()
        text_features = text_features / text_features.norm(dim=1,
                                                           keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
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

    def _encode_text(self) -> torch.Tensor:
        eot_indices = (self.tokenized_prompts
                       == self.eot_token).nonzero(as_tuple=True)[1]

        x = self.clip_model.token_embedding(self.tokenized_prompts)

        x = self._modified_text_encode(x, eot_indices)

        return x

    def _modified_text_encode(self, x, eot_indices):
        x = x + self.clip_model.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), eot_indices]

        x = x @ self.clip_model.text_projection
        return x

    def _encode_image(self, images):
        if images.dim == 3:
            images = images.unsqueeze(0)

        if self.pgn_module:
            visual_context = self.pgn_module(images)

            video_features = self._modified_visual_encode(images,
                                                          visual_context)
        else:
            video_features = self._modified_visual_encode(images)

        return video_features

    def _modified_visual_encode(self, x, context=None):
        x = self.clip_model.visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.clip_model.visual.class_embedding.to(x.dtype)
             + torch.zeros(x.shape[0], 1, x.shape[-1],
                           dtype=x.dtype, device=x.device),
             x],
            dim=1
        )
        x = x + self.clip_model.visual.positional_embedding.to(x.dtype)
        if context is not None:
            x = torch.cat(
                [x, context],
                dim=1
            )
        x = self.clip_model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.clip_model.visual.ln_post(x[:, 0, :])

        if self.clip_model.visual.proj is not None:
            x = x @ self.clip_model.visual.proj

        return x

    def _tokenize_prompts(self) -> None:
        tokenized_prompts = clip.tokenize(
            self.trainer.datamodule.prompts
        )
        self.tokenized_prompts = tokenized_prompts.to(self.device)

    def _freeze_components(self) -> None:
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def on_test_start(self) -> None:
        self._tokenize_prompts()

    def on_fit_start(self) -> None:
        self._tokenize_prompts()
