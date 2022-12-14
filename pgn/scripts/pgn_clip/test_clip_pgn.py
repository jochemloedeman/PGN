import argparse

from pgn.lightning.pgn_clip import PGNCLIP
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from pgn.datamodules import \
    CIFAR100DataModule, DTDDataModule, SUN397DataModule, Food101DataModule, \
    Flowers102DataModule, EuroSATDataModule, UCF101DataModule, \
    OxfordPetsDataModule, CIFAR10DataModule, SVHNDataModule, RESISC45DataModule, \
    CLEVRCountDataModule

datamodules = {
    'cifar10': CIFAR10DataModule,
    'cifar100': CIFAR100DataModule,
    'dtd': DTDDataModule,
    'sun397': SUN397DataModule,
    'food101': Food101DataModule,
    'flowers102': Flowers102DataModule,
    'eurosat': EuroSATDataModule,
    'ucf101': UCF101DataModule,
    'oxford_pets': OxfordPetsDataModule,
    'svhn': SVHNDataModule,
    'resisc45': RESISC45DataModule,
    'clevr_count': CLEVRCountDataModule,
}

visual_embedding_dims = {
    'ViT-B/32': 768,
    'ViT-B/16': 768,
    'ViT-L/14': 1024,
}

def test(args):
    seed_everything(seed=args.seed, workers=True)

    datamodule = datamodules[args.dataset](
        data_root=args.data_root,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        scale_lower_bound=args.rrc_scale_lb,
        jitter_prob=args.jitter_prob,
        greyscale_prob=args.greyscale_prob,
        solarize_prob=args.solarize_prob,
    )

    if not args.disable_pgn:
        pgn_settings = {
            'prompt_mode': args.prompt_mode,
            'pgn_resolution': args.pgn_resolution,
            'nr_output_vectors': args.pgn_length,
            'vector_dim': visual_embedding_dims[args.architecture],
            'mixture_size': args.pgn_mixture_size,
            'pretrained_pgn': args.pretrained_pgn,
            'model_type': args.pgn_model_type,
            'proj_type': args.pgn_proj_type,
            'pgn_act_fn': args.pgn_act_fn,
            'nr_groups': args.nr_groups,
            'blocks_per_group': args.blocks_per_group,
            'initial_channels': args.initial_channels,
            'init_max_pool': args.init_max_pool

        }
    else:
        pgn_settings = None

    pgn_clip = PGNCLIP(
        clip_architecture=args.architecture,
        pgn_settings=pgn_settings,
        optimizer=args.optimizer,
        init_lr=args.init_lr,
        lr_scheduler=args.lr_scheduler,
        warmup_epochs=args.warmup_epochs,
        epochs=args.epochs,
        pgn_path=args.pgn_model_path,
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=-1,
        precision=args.precision,
        strategy=args.strategy,
    )

    trainer.test(
        model=pgn_clip,
        datamodule=datamodule,
    )


def main():
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--dataset', default='cifar100', type=str)
    parser.add_argument('--data_root',
                        default='/home/jochem/Documents/ai/scriptie/PGN/data',
                        type=str)
    parser.add_argument('--rrc_scale_lb', default=1., type=float)
    parser.add_argument('--jitter_prob', default=0.8, type=float)
    parser.add_argument('--greyscale_prob', default=0.2, type=float)
    parser.add_argument('--solarize_prob', default=0.2, type=float)

    # Model + Training
    parser.add_argument('--pgn_model_path', default='checkpoints/pgn_clip/cifar100/pgn_clip_cifar100.pt', type=str)
    parser.add_argument('--architecture', default='ViT-B/32', type=str)
    parser.add_argument('--train_batch_size', default=64, type=int)
    parser.add_argument('--val_batch_size', default=64, type=int)
    parser.add_argument('--precision', default=32, type=int)

    # PGN
    parser.add_argument('--pgn_length', default=16, type=int)
    parser.add_argument('--prompt_mode', default='pgn', type=str)
    parser.add_argument('--pgn_mixture_size', default=256, type=int)
    parser.add_argument('--pgn_act_fn', default='softmax', type=str)

    # PGN Model
    parser.add_argument('--pgn_model_type', default='resnet10', type=str)
    parser.add_argument('--pgn_proj_type', default='linear', type=str)
    parser.add_argument('--pgn_resolution', default=224, type=int)
    parser.add_argument('--nr_groups', default=4, type=int)
    parser.add_argument('--blocks_per_group', default=1, type=int)
    parser.add_argument('--initial_channels', default=16, type=int)
    parser.add_argument('--init_max_pool', action=argparse.BooleanOptionalAction,
                        default=True)

    # Optimizer
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--init_lr', default=0.1, type=float)
    parser.add_argument('--lr_scheduler', default='warmup', type=str)

    # Experiment
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--warmup_epochs', default=50, type=int)
    parser.add_argument('--strategy', default=None, type=str)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--seed', default=0, type=int)

    # Switches
    parser.add_argument('--dev_run', action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument('--profiler', action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument('--disable_pgn', action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument('--pretrained_pgn', action=argparse.BooleanOptionalAction,
                        default=False)

    args = parser.parse_args()

    test(args)

if __name__ == '__main__':
    main()