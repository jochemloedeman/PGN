# Prompt Generation Networks for Efficient Adaptation of Frozen Vision Transformers

This repository is the official implementation of [Prompt Generation Networks for Efficient Adaptation of Frozen Vision Transformers](https://arxiv.org/abs/2030.12345). 

![](./figure/arch.png)

## Requirements

To install python dependencies, make sure that poetry is installed and execute the following in the project root directory:

```setup
poetry install
```
### Data
See `data/DATA.md`

## Training

To train/test with the CLIP backbone, run

```train
poetry run train_clip
poetry run test_clip
```

To train/test with either DINO or supervised ViT, specify the backbone with `--vision_model_type` and run

```train
poetry run train_visionmodel
poetry run test_visionmodel
```

For all available command line arguments, see `pgn/scripts`.

## Pre-trained PGNs
Pre-trained PGNs are supplied in `pretrained_pgns/`. To use them in the context of this repository, specify the desired model by setting the `--pgn_path` argument in the test scripts.
