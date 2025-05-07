import csv
import time
from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from skimage.measure import find_contours

from .ml_dataset import UNetDataset, create_dataloaders, read_data
from .ml_inferece import load_model
from .ml_metrics import DiceCoeff, IoUCoeff


def dump_test_scores(scores, path):
    score_dict = {
        "img_path": scores[0],
        "iou_scores": scores[1],
        "dice_scores": scores[2],
    }
    with open(path / "test_scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(score_dict.keys())
        writer.writerows(zip(*score_dict.values()))


def bestmodel(path, data, mode):
    with open(path / f"model_{mode}_time.txt", "w", newline="") as f:
        f.write(str(data[0]) + "\n")
        f.write("\n")
        f.write("\n")
        f.write("\n")
        f.write("\n")
        f.write(f"{mode} time per image:" + str(data[1]) + "\n")


def div_inference(model_path, results_path, use_cuda=True):
    params_path = yaml.safe_load(open(results_path / "train_params.yaml"))
    data_path = Path(params_path["dataset"]["data_path"]).with_suffix("")
    div_fol_path = data_path / "testing_divided_groups"
    if div_fol_path.is_dir():
        target_shape = params_path["dataset"]["img_size"]
        mean = params_path["dataset"]["mean"]
        std = params_path["dataset"]["std"]

        unet, unet_meta = load_model(model_path, use_cuda=use_cuda)
        device = torch.device("cuda" if use_cuda else "cpu")

        for fname in div_fol_path.iterdir():
            if (fname / "images").is_dir():
                img_files = sorted(
                    [
                        p
                        for p in (fname / "images").rglob("*.png")
                        if p.is_file()
                    ]
                )

                im_file_names = [p.stem for p in img_files]
                im_file_names.append("Average Scores")

                outpath = results_path / fname.stem
                outpath.mkdir(parents=True, exist_ok=True)

                images, masks = read_data(fname, seed=42, shuffle=True)

                test_dataset = UNetDataset(
                    images, masks, target_shape, mean=mean, std=std
                )

                data_dict = {"test": test_dataset}
                test_dataloader = create_dataloaders(
                    data_dict, batch_size=8, num_workers=0
                )["test"]

                ioumetric = IoUCoeff()
                dicemetric = DiceCoeff(sample_wise=True)

                image_list = []
                predict_list = []
                target_list = []

                tik = time.time()

                for img_batch, lbl_batch in test_dataloader:
                    img_batch = img_batch.to(device, dtype=torch.float32)
                    lbl_batch = lbl_batch.to(device, dtype=torch.float32)

                    predicts = unet(img_batch)

                    predict_list.append(predicts)
                    target_list.append(lbl_batch)
                    image_list.append(img_batch)

                inference_time = time.time() - tik
                inf_time_per_img = inference_time / len(
                    test_dataloader.dataset
                )
                if use_cuda:
                    bestmodel(outpath, [model_path, inf_time_per_img], "gpu")
                    print("Inference time (gpu)/image:", inf_time_per_img)
                else:
                    bestmodel(outpath, [model_path, inf_time_per_img], "cpu")
                    print("Inference time (cpu)/image:", inf_time_per_img)

                image_torch = torch.cat(image_list, dim=0)
                predict_torch = torch.cat(predict_list, dim=0)
                target_torch = torch.cat(target_list, dim=0)

                iou_scores = ioumetric(predict_torch, target_torch)
                dice_scores = dicemetric(predict_torch, target_torch)
                iou_scores = iou_scores.detach().cpu().numpy()
                dice_scores = dice_scores.detach().cpu().numpy()

                iou_scores = np.append(iou_scores, [np.mean(iou_scores)])
                dice_scores = np.append(dice_scores, [np.mean(dice_scores)])

                if use_cuda:
                    dump_test_scores(
                        [im_file_names, iou_scores, dice_scores], outpath
                    )

                if use_cuda:
                    image_numpy = image_torch.squeeze(1).detach().cpu().numpy()
                    mask_numpy = target_torch.squeeze(1).detach().cpu().numpy()
                    predict_numpy = (
                        predict_torch.squeeze(1).detach().cpu().numpy()
                    )

                    for n in range(len(image_numpy)):
                        img = image_numpy[n]
                        msk = mask_numpy[n]
                        pred = predict_numpy[n]
                        iou = iou_scores[n]
                        dice = dice_scores[n]
                        im_name = im_file_names[n]

                        pred_cnt = find_contours(pred, 0.8)
                        msk_cnt = find_contours(msk, 0.8)

                        fig, axs = plt.subplots(
                            nrows=2, ncols=3, figsize=(8, 4)
                        )

                        axs[0, 0].imshow(img, "gray")
                        axs[0, 0].axis("off")
                        axs[0, 1].imshow(msk, "gray")
                        axs[0, 1].set_title("Ground truth")
                        axs[0, 1].axis("off")

                        axs[0, 2].imshow(pred, "gray")
                        axs[0, 2].set_title(
                            f"Pred (IoU: {iou:.3f}, Dice: {dice:.3f})"
                        )
                        axs[0, 2].axis("off")
                        gs = axs[1, 1].get_gridspec()

                        # remove the underlying axes
                        for ax in axs[1, :]:
                            ax.remove()
                        axbig = fig.add_subplot(gs[1, :])
                        axbig.imshow(img, "gray")
                        axbig.axis("off")
                        for pc in pred_cnt:
                            axbig.plot(pc[:, 1], pc[:, 0], "r", linewidth=2)
                        for mc in msk_cnt:
                            axbig.plot(mc[:, 1], mc[:, 0], "g", linewidth=2)

                        red_patch = mpatches.Patch(
                            color="red", label="Prediction"
                        )
                        green_patch = mpatches.Patch(
                            color="green", label="Ground Truth"
                        )
                        plt.legend(handles=[red_patch, green_patch])

                        fig.tight_layout()
                        fig.savefig(
                            outpath / f"{im_name}_iou_{iou:.1f}_dice_"
                            f"{dice:.1f}.png"
                        )
                        plt.close()
