import csv
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from .divided_group_inference import div_inference
from .early_stopping import EarlyStopping
from .ml_criterions import get_criterion_with_params
from .ml_dataset import get_dataloaders_with_params
from .ml_inferece import inference
from .ml_metrics import get_metric_with_params
from .ml_optimizers import get_optimizer_with_params
from .ml_schedulers import get_scheduler_with_params
from .models import convert_torch_to_onnx, get_model_with_params, summary

save_valid_results = False


def keep_file_delete_others(folder_path, file_to_keep):
    folder_path = Path(folder_path)
    for file in folder_path.iterdir():
        if file.is_file() and file.name != Path(file_to_keep).name:
            file.unlink()


def plot_valid_results(
    results_path, n, image_torch, target_torch, predict_torch
):
    results_path = results_path / "valid_results"
    results_path.mkdir(parents=True, exist_ok=True)

    img = image_torch.squeeze(1).detach().cpu().numpy()
    msk = target_torch.squeeze(1).detach().cpu().numpy()
    pred = torch.sigmoid(predict_torch).squeeze(1).detach().cpu().numpy()

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(20, 6))
    for r, row in enumerate(ax):
        for c, col in enumerate(row):
            if c == 0:
                col.imshow(img[r], "gray")
                col.set_title("img")
                col.axis("off")
            if c == 1:
                col.imshow(msk[r], "gray")
                col.set_title("msk")
                col.axis("off")
            if c == 2:
                col.imshow(pred[r], "gray")
                col.set_title("pred")
                col.axis("off")
    fig.savefig(results_path / f"valid_pred_at_epoch_{n}.png")
    plt.close(fig)


class SetupTrainer:
    def __init__(
        self,
        model,
        dataloaders,
        criterion,
        metric,
        optimizer,
        scheduler,
        max_epochs=100,
        use_cuda=False,
        min_ckp_acc=0.85,
        early_stop_patience=None,
        path_out="experiments",
        tensorboard=False,
        init_from_ckp=None,
    ):
        self.model = model
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.metric = metric
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_epochs = max_epochs
        self.use_cuda = use_cuda
        self.min_ckp_acc = min_ckp_acc
        self.tensorboard = tensorboard

        # Create EarlyStop instance only if it is specified
        if early_stop_patience is not None:
            self.early_stop = EarlyStopping(early_stop_patience)
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        if init_from_ckp is not None:
            self.restore_checkpoint(init_from_ckp)

        if Path(path_out).is_dir() and path_out != "experiments":
            self.exp_path = Path(path_out)
        else:
            # Create a folder to store experiment results
            time_now = datetime.now().strftime("%d_%b_%Y_%H%M%S%f")
            self.exp_path = Path(path_out) / time_now
            self.exp_path.mkdir(parents=True, exist_ok=True)

        # Create a folder to save model checkpoints
        self.ckp_path = self.exp_path / "checkpoints"
        self.ckp_path.mkdir(parents=True, exist_ok=True)

        self.dump_model_summary(model, self.exp_path)

        if self.use_cuda:
            self.model = model.cuda()
            self.criterion = criterion.cuda()
            # self.model = DataParallel(model)

        # Create a folder to save logs with tensorboard
        if self.tensorboard:
            tb_path = str(self.exp_path.parents[0] / "tensorboard")
            run_name = str(self.exp_path.name)
            self.writer = SummaryWriter(log_dir=tb_path + f"/{run_name}")

        print(f"Training samples: {len(self.dataloaders['train'].dataset)}")
        print(f"Validation samples: {len(self.dataloaders['valid'].dataset)}")

    @classmethod
    def with_params(cls, params):
        model = get_model_with_params(params)
        dataloaders = get_dataloaders_with_params(params)
        criterion = get_criterion_with_params(params)
        metric = get_metric_with_params(params)
        optimizer = get_optimizer_with_params(params, model)
        scheduler = get_scheduler_with_params(params, optimizer)

        other_params = params.get("others")
        max_epochs = other_params.get("max_epochs")
        use_cuda = other_params.get("use_cuda")
        min_ckp_acc = other_params.get("min_ckp_acc")
        early_stop_patience = other_params.get("early_stop_patience")
        path_out = other_params.get("path_out")
        init_from_ckp = other_params.get("init_from_ckp")
        tensorboard = other_params.get("tensorboard")

        # Create a folder to store experiment results based on current time
        time_now = datetime.now().strftime("%d_%b_%Y_%H%M%S%f")
        exp_path = Path(path_out) / time_now
        exp_path.mkdir(parents=True, exist_ok=True)

        # Save session parameters as a params.yaml file
        out_file_path = exp_path / "train_params.yaml"
        with open(out_file_path, "w") as file:
            yaml.dump(params, file, sort_keys=False)

        return cls(
            model,
            dataloaders,
            criterion,
            metric,
            optimizer,
            scheduler,
            max_epochs,
            use_cuda,
            min_ckp_acc,
            early_stop_patience,
            str(exp_path),
            tensorboard,
            init_from_ckp,
        )

    def restore_checkpoint(self, ckp_path):
        ckp = torch.load(ckp_path, map_location=self.device)
        self.model.load_state_dict(ckp["model_state_dict"])
        self.optimizer.load_state_dict(ckp["optimizer_state_dict"])
        for param in self.model.features.parameters():
            param.requires_grad = False

    def dump_model_summary(self, model, pathout):
        images, _ = next(iter(self.dataloaders["train"]))
        result, params_info = summary(model, tuple(images[0].shape))
        print(f"Total parameters in the model:{params_info[0]}")
        print(f"Trainable parameters in the model:{params_info[1]}")
        lines = result.split("\n")

        # Open the CSV file in write mode
        with open(pathout / "model_summary.csv", "w", newline="") as f:
            writer = csv.writer(f)
            for line in lines:
                writer.writerow([line])

    def epoch_runner(self, epoch, mode):
        if mode.lower() == "train":
            self.model.train()
        if mode.lower() == "valid":
            self.model.eval()
        loss_list = []
        # predict_list = []
        # mask_list = []

        bscore = []
        # Get batch of images and labels iteratively
        for n, (images, masks) in enumerate(self.dataloaders[mode]):
            # Pass the data to the device
            images = images.to(self.device, dtype=torch.float32)
            masks = masks.to(self.device, dtype=torch.float32)
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            # Track history only in train mode
            with torch.set_grad_enabled(mode.lower() == "train"):
                predicts = self.model(images)
                # [B, 1, H, W] --> [B, H, W] for loss calculation
                # predicts = predicts.squeeze(1)
                loss = self.criterion(predicts, masks)
                # Backward + optimize only in train mode
                if mode.lower() == "train":
                    loss.backward()
                    self.optimizer.step()
                loss_item = loss.item()
                del loss  # this may be the fix for my OOM error
                loss_list.append(loss_item)

                # predict_list.append(predicts)
                # mask_list.append(masks)

                bwise_scores = self.metric(predicts, masks)
                bscore.append(bwise_scores.cpu())

                if save_valid_results:
                    if mode.lower() == "valid" and n == 5:
                        plot_valid_results(
                            self.exp_path, epoch, images, masks, predicts
                        )

        # predict_tensor = torch.cat(predict_list, dim=0)
        # mask_tensor = torch.cat(mask_list, dim=0)
        # scores = self.metric(predict_tensor, mask_tensor)
        # acc_avg = float(scores.mean())

        loss_avg = float(np.stack(loss_list).mean())
        acc_avg = float(torch.mean(torch.stack(bscore)))

        return loss_avg, acc_avg

    def plot_logs(self, logs):
        epoch = np.arange(1, logs["epochs"] + 1)
        train_loss = logs["train_loss"]
        train_acc = logs["train_acc"]
        val_loss = logs["val_loss"]
        val_acc = logs["val_acc"]
        ckp_data = logs["ckp_flags"]
        early_stop_epoch = logs["early_stop"]
        if len(ckp_data) != 0:
            ckp_epochs = np.array(ckp_data)[:, 0]
            ckp_val_acc = np.array(ckp_data)[:, 1]
            ckp_val_loss = np.array(ckp_data)[:, 2]
        else:
            ckp_epochs, ckp_val_acc, ckp_val_loss = [], [], []

        plt.figure(figsize=(14, 7), facecolor=(1, 1, 1))
        plt.rc("grid", linestyle="--", color="lightgrey")
        plt.subplot(121)
        plt.plot(
            epoch, train_acc, color="red", label="Train accuracy", zorder=1
        )
        plt.plot(
            epoch, val_acc, color="green", label="Valid accuracy", zorder=2
        )
        if len(ckp_epochs) > 1:
            plt.scatter(
                ckp_epochs,
                ckp_val_acc,
                c="blue",
                s=20,
                marker="x",
                label=f"Saved_ckp(>={self.min_ckp_acc:.2f})",
                zorder=3,
            )
        plt.legend(loc="lower right")
        plt.title("Accuracy plot")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.ylim(0.3, 1.0)
        plt.grid()
        plt.subplot(122)
        plt.plot(epoch, train_loss, color="red", label="Train loss", zorder=1)
        plt.plot(epoch, val_loss, color="green", label="Valid loss", zorder=2)
        if len(early_stop_epoch) == 1:
            plt.scatter(
                early_stop_epoch,
                ckp_val_loss,
                c="blue",
                s=20,
                marker="x",
                label="Early_stopping",
                zorder=3,
            )
        plt.legend(loc="upper right")
        plt.title("Loss plot")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.ylim(-0.01, 0.5)
        plt.grid()
        plt.savefig(self.exp_path / "train_plot.png")

    def print_epoch_logs(self, epoch_log):
        epoch, dynamic_lr, train_loss, train_acc, val_loss, val_acc = epoch_log
        print(f"[Epochs-{epoch}/{self.max_epochs} | lr:{dynamic_lr}]:")
        print(
            f"[Train_loss:{train_loss:.4f} | Train_acc:{train_acc:.4f} | "
            f"Val_loss:{val_loss:.4f} | Val_acc:{val_acc:.4f}]"
        )

    def save_checkpoint(self, new_ckp_name, mode="jit"):
        # Model meta data
        params_dict = {
            "image_shape": self.dataloaders["train"].dataset.target_shape,
            "mean": self.dataloaders["train"].dataset.mean,
            "std": self.dataloaders["train"].dataset.std,
            "padding_ufunc": "np.mean",
        }

        # make sure model is in eval mode
        model = self.model.eval()
        if mode == "jit":
            jit_dir = self.ckp_path / "torch_jit"
            jit_dir.mkdir(parents=True, exist_ok=True)
            jit_path = str(jit_dir) + f"/{new_ckp_name}_jit.ckp"
            model_scripted = torch.jit.script(model)
            model_scripted.save(jit_path)

            extra_files = {"meta": str(params_dict)}
            torch.jit.save(model_scripted, jit_path, _extra_files=extra_files)
        else:
            torch_dir = self.ckp_path / "torch_original"
            torch_dir.mkdir(parents=True, exist_ok=True)
            org_path = str(torch_dir) + f"/{new_ckp_name}_org.ckp"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "model_instance": model,
                },
                org_path,
            )

    def add_graph_tb(self):
        dataiter = iter(self.dataloaders["train"])
        images, masks = next(dataiter)
        images = images.to(self.device, dtype=torch.float32)
        self.writer.add_graph(self.model, images)

    def dump_train_logs(self, logs):
        with open(self.exp_path / "train_logs.yaml", "w") as fp:
            yaml.dump(logs, fp, sort_keys=False, default_flow_style=None)

    def dump_test_scores(self, scores):
        score_dict = {
            "img_path": scores[-1],
            "iou_scores": scores[1],
            "dice_scores": scores[2],
        }
        with open(self.exp_path / "test_scores.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(score_dict.keys())
            writer.writerows(zip(*score_dict.values()))

    def start_train(self):
        val_avg_acc_list = []
        train_logs = {
            "train_samples": len(self.dataloaders["train"].dataset),
            "valid_samples": len(self.dataloaders["valid"].dataset),
            "test_samples": len(self.dataloaders["test"].dataset),
            "training_time": 0,
            "inference_gpu": 0,
            "inference_cpu": 0,
            "test_iou_mean": 0,
            "test_dice_mean": 0,
            "epochs": 0,
            "ckp_flags": [],
            "val_acc": [],
            "train_acc": [],
            "val_loss": [],
            "train_loss": [],
            "dynamicLR": [],
            "early_stop": [],
        }
        start_time = time.time()
        for epoch in range(1, self.max_epochs + 1):
            train_avg_loss, train_avg_acc = self.epoch_runner(
                epoch, mode="train"
            )
            val_avg_loss, val_avg_acc = self.epoch_runner(epoch, mode="valid")
            val_avg_acc_list.append(val_avg_acc)
            dynamic_lr = [
                group["lr"] for group in self.optimizer.param_groups
            ][0]
            train_logs["epochs"] = self.max_epochs
            train_logs["dynamicLR"].append(dynamic_lr)
            train_logs["train_loss"].append(train_avg_loss)
            train_logs["train_acc"].append(train_avg_acc)
            train_logs["val_loss"].append(val_avg_loss)
            train_logs["val_acc"].append(val_avg_acc)
            epoch_log = [
                epoch,
                dynamic_lr,
                train_avg_loss,
                train_avg_acc,
                val_avg_loss,
                val_avg_acc,
            ]
            self.print_epoch_logs(epoch_log)

            # Tensorboard tracking loss and accuracies
            if self.tensorboard:
                self.writer.add_scalars(
                    "Loss Plot (Train vs Valid)",
                    {"Train": train_avg_loss, "Valid": val_avg_loss},
                    epoch,
                )
                self.writer.add_scalars(
                    "Accuracy Plot (Train vs Valid)",
                    {"Train": train_avg_acc, "Valid": val_avg_acc},
                    epoch,
                )

            if val_avg_acc > self.min_ckp_acc and val_avg_acc == max(
                val_avg_acc_list
            ):
                new_ckp_name = (
                    f"/E{epoch}_trainAcc_"
                    f"{int(train_avg_acc * 1e4)}_validAcc_"
                    f"{int(val_avg_acc * 1e4)}"
                )
                self.save_checkpoint(new_ckp_name)
                self.save_checkpoint(new_ckp_name, mode="original")
                train_logs["ckp_flags"].append(
                    [epoch, val_avg_acc, val_avg_loss]
                )

            # Reduce the learning rate, if validation accuracy is not improving
            if self.scheduler and isinstance(self.scheduler, StepLR):
                self.scheduler.step()
            else:
                self.scheduler.step(val_avg_loss)

            # Track early stopping patience if it is specified
            if hasattr(self, "early_stop"):
                # Stop the training, if validation loss is not improving
                self.early_stop(val_avg_loss)
                if self.early_stop.should_stop:
                    train_logs["early_stop"].append(epoch)
                    print("Early stopping!")
                    break
        # Calculate training time and save it in results logs
        end_time = time.time() - start_time
        train_time = str(timedelta(seconds=end_time)).split(".")[0]
        # Reset the memory by deleting model and cache
        del self.model
        torch.cuda.empty_cache()
        print("=" * 80)
        print(f"Total training time: {train_time}")
        train_logs["training_time"] = train_time

        ckp_flag_arr = np.array(train_logs["ckp_flags"])

        if len(ckp_flag_arr) > 0:
            req_flag = int(max(ckp_flag_arr[:, 0]))
            org_dir = self.ckp_path / "torch_original"
            org_ckp_paths = [p for p in Path(org_dir).rglob("*.ckp")]
            org_paths = [p for p in org_ckp_paths if f"E{req_flag}" in str(p)]
            if len(org_paths) > 0:
                final_org_path = org_paths[0]
                keep_file_delete_others(org_dir, final_org_path)
                convert_torch_to_onnx(
                    final_org_path,
                    img_size=self.dataloaders["train"].dataset.target_shape,
                )

        if len(ckp_flag_arr) > 0:
            req_ckp_flag = int(max(ckp_flag_arr[:, 0]))
            jit_dir = self.ckp_path / "torch_jit"
            ckp_paths = [p for p in Path(jit_dir).rglob("*.ckp")]
            ckp_path = [p for p in ckp_paths if f"E{req_ckp_flag}" in str(p)]
            if len(ckp_path) > 0:
                final_ckp_path = ckp_path[0]
                keep_file_delete_others(jit_dir, final_ckp_path)
                test_results = inference(
                    self.dataloaders["test"],
                    final_ckp_path,
                    self.exp_path,
                    use_cuda=False,
                    save_results=False,
                )
                train_logs["inference_cpu"] = test_results[0]
                train_logs["test_iou_mean"] = float(test_results[1].mean())
                train_logs["test_dice_mean"] = float(test_results[2].mean())
                self.dump_test_scores(test_results)

                # Testing divided groups with CPU device
                div_inference(final_ckp_path, self.exp_path, use_cuda=False)

                # Run inference using gpu only it is available
                if torch.cuda.is_available():
                    test_results = inference(
                        self.dataloaders["test"],
                        final_ckp_path,
                        self.exp_path,
                        use_cuda=True,
                        save_results=True,
                    )
                    train_logs["inference_gpu"] = test_results[0]

                    # Testing divided groups with CUDA device
                    div_inference(final_ckp_path, self.exp_path, use_cuda=True)

        # Plot and save results logs
        self.plot_logs(train_logs)
        self.dump_train_logs(train_logs)
        # self.add_graph_tb()
        if self.tensorboard:
            self.close()

    def close(self):
        self.writer.flush()
        self.writer.close()
