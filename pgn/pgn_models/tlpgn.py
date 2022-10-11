import functools

import pytorch_lightning as pl
import torch
import torchvision

from pgn.pgn_models.pgn_switch import pgn_switch


def _get_act_fn(act_fn):
    if act_fn == 'softmax':
        return functools.partial(torch.softmax, dim=-1)
    elif act_fn == 'sigmoid':
        return torch.sigmoid
    else:
        raise ValueError("Invalid IDP activation function")


class TLPGN(pl.LightningModule):
    def __init__(
            self,
            nr_output_vectors,
            mixture_size,
            vector_dim,
            model_type,
            pgn_act_fn,
            pgn_resolution,
            **kwargs,
    ) -> None:
        super().__init__()
        self.nr_output_vectors = nr_output_vectors
        self.mixture_size = mixture_size
        self.vector_dim = vector_dim
        self.resolution = pgn_resolution
        self.model = pgn_switch(model_type,
                               pgn_resolution=pgn_resolution,
                               out_features=nr_output_vectors * mixture_size,
                               **kwargs)

        tl_vectors = torch.empty(
            mixture_size,
            vector_dim,
            dtype=self.dtype,
            device=self.device,
        )
        torch.nn.init.normal_(tl_vectors, std=0.02)
        self.tl_vectors = torch.nn.Parameter(tl_vectors)
        self.pgn_act_fn = pgn_act_fn
        self.act_fn = self._get_act_fn(pgn_act_fn)

    def forward(self, images):
        images = torchvision.transforms.functional.resize(images,
                                                          self.resolution)
        logits = self.model(images)
        split_logits = logits.reshape(
            len(logits),
            self.nr_output_vectors,
            self.mixture_size
        )
        mixture_coeffs = self.act_fn(
            split_logits
        )
        pgn_prompts = torch.einsum(
            'bom,mv->bov',
            [mixture_coeffs, self.tl_vectors]
        )
        return pgn_prompts

    @staticmethod
    def _get_act_fn(act_fn):
        if act_fn == 'softmax':
            return functools.partial(torch.softmax, dim=-1)
        elif act_fn == 'sigmoid':
            return torch.sigmoid
        else:
            raise ValueError("Invalid PGN activation function")
