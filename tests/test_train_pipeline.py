import json
from pathlib import Path

import yaml

from semanticsegmentor.ml_criterions import get_criterion_with_params
from semanticsegmentor.ml_dataset import get_dataloaders_with_params
from semanticsegmentor.ml_metrics import get_metric_with_params
from semanticsegmentor.ml_models import get_model_with_params
from semanticsegmentor.ml_optimizers import get_optimizer_with_params
from semanticsegmentor.ml_schedulers import get_scheduler_with_params
from semanticsegmentor.ml_trainer import SetupTrainer
from .helper_methods import retrieve_train_data_path

path_in = retrieve_train_data_path("test_dataset.zip") / "dataset"
params_path = Path(__file__).parent / "data" / "test_params.yaml"
params = yaml.safe_load(open(params_path))

dataset_params = {
    "dataset": {
        "type": "PNG",
        "data_path": path_in,
        "augmentation": False,
        "valid_size": 0.2,
        "batch_size": 4,
        "mean": [0.67709],
        "std": [0.13369],
        "num_workers": 0,
        "num_samples": None
    }
}


def test_trainer_and_pipeline():
    path_out = path_in.with_name("temp_out")

    model = get_model_with_params(params)
    dataloaders = get_dataloaders_with_params(dataset_params)
    criterion = get_criterion_with_params(params)
    metric = get_metric_with_params(params)
    optimizer = get_optimizer_with_params(params, model)
    scheduler = get_scheduler_with_params(params, optimizer)

    max_epochs = params.get("max_epochs")
    use_cuda = params.get("use_cuda")
    min_ckp_acc = params.get("min_ckp_acc")
    early_stop_patience = params.get("early_stop_patience")
    tensorboard = params.get("tensorboard")

    trainer = SetupTrainer(model,
                           dataloaders,
                           criterion,
                           metric,
                           optimizer,
                           scheduler,
                           max_epochs=max_epochs,
                           use_cuda=use_cuda,
                           min_ckp_acc=min_ckp_acc,
                           early_stop_patience=early_stop_patience,
                           path_out=path_out,
                           tensorboard=tensorboard,
                           init_from_ckp=None)
    trainer.start_train()
    test_log_file = [p for p in path_out.glob("**/train_logs.json")][0]
    with open(test_log_file) as f:
        data = json.load(f)
    assert "epochs" in data.keys()
    assert len(data["epochs"]) == 2
    assert data["train_samples"] == 8
    assert data["valid_samples"] == 2
