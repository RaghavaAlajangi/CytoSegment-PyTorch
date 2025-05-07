from pathlib import Path

import click
import torch
import yaml

from .ml_trainer import SetupTrainer


@click.command(help="Script to start the training")
@click.option("--params_path",
              type=click.Path(exists=True,
                              dir_okay=False,
                              resolve_path=True,
                              path_type=Path),
              help="Path to params file (.yaml)")
def main(params_path):
    params = yaml.safe_load(open(params_path))
    print("=" * 80)
    print("Cuda available:", torch.cuda.is_available())
    trainer = SetupTrainer.with_params(params)
    print("=" * 80)
    print("Model training has been started....")
    trainer.start_train()


if __name__ == "__main__":
    main()
