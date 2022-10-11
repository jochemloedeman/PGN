import pytorch_lightning as pl
import torch


class IIP(pl.LightningModule):
    def __init__(
            self,
            nr_output_vectors,
            vector_dim,
            **kwargs,
    ) -> None:
        super().__init__()
        self.nr_output_vectors = nr_output_vectors
        self.vector_dim = vector_dim

        pgn_vectors = torch.empty(
            nr_output_vectors,
            vector_dim,
            dtype=self.dtype,
            device=self.device,
        )
        torch.nn.init.normal_(pgn_vectors, std=0.02)
        self.pgn_vectors = torch.nn.Parameter(pgn_vectors)

    def forward(self, images):
        batch_size = len(images)
        pgn_prompts = self.pgn_vectors.repeat(batch_size, 1, 1).type_as(images)
        return pgn_prompts
