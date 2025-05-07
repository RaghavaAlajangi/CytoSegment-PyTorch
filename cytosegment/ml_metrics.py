import torch
import torch.nn as nn


def get_metric_with_params(params):
    assert {"metric"}.issubset(params)
    metric_params = params.get("metric")
    assert {"type"}.issubset(metric_params)
    metric_type = metric_params.get("type")

    if metric_type.lower() == "ioucoeff":
        return IoUCoeff()

    if metric_type.lower() == "dicecoeff":
        return DiceCoeff()


class DiceCoeff(nn.Module):
    def __init__(self, smooth=1e-7, thresh=0.5, sample_wise=False):
        super(DiceCoeff, self).__init__()
        self.smooth = smooth
        self.thresh = thresh
        self.sample_wise = sample_wise

    def forward(self, predicts, targets):
        # Note: Sample wise metric evaluation gives different result than
        # batch wise nearly 1.2% difference
        if self.sample_wise:
            # Reshape predicts and targets [B, C, W, H] --> [B, C*W*H]
            predicts = predicts.view(predicts.shape[0], -1)
            targets = targets.view(targets.shape[0], -1)

            # Apply thresholding
            predicts = torch.where(predicts >= self.thresh, 1.0, 0.0)
            targets = torch.where(targets >= self.thresh, 1.0, 0.0)

            # Compute numerator and denominator
            numerator = 2 * (predicts * targets).sum(dim=1)
            denominator = predicts.sum(dim=1) + targets.sum(dim=1)

            dice_coeff = numerator / (denominator + self.smooth)
            return dice_coeff
        else:
            # Reshape predicts and targets [B, C, W, H] --> [B*C*W*H]
            predicts = predicts.flatten()
            targets = targets.flatten()

            # Apply thresholding
            predicts = torch.where(predicts >= self.thresh, 1.0, 0.0)
            targets = torch.where(targets >= self.thresh, 1.0, 0.0)

            numerator = 2 * (predicts * targets).sum()
            denominator = predicts.sum() + targets.sum()

            dice_coeff = numerator / (denominator + self.smooth)
            return dice_coeff


class IoUCoeff(nn.Module):
    def __init__(self, smooth=1e-6, thresh=0.5):
        super(IoUCoeff, self).__init__()
        self.smooth = smooth
        self.thresh = thresh

    def forward(self, predicts, targets):
        # Flatten and binarize the predictions and targets
        predicts = predicts.view(predicts.shape[0], -1) >= self.thresh
        targets = targets.view(targets.shape[0], -1) >= self.thresh

        # Compute intersection and union
        intersection = torch.logical_and(predicts, targets).sum(dim=1)
        union = torch.logical_or(predicts, targets).sum(dim=1)
        # Add smooth to avoid 0/0
        iou = (intersection + self.smooth) / (union + self.smooth)
        return iou


class PixelHit(nn.Module):
    def __init__(self, smooth=1e-6, thresh=0.5):
        super(PixelHit, self).__init__()
        self.smooth = smooth
        self.thresh = thresh

    def forward(self, predicts, targets):
        # Squeeze channel dim - (BATCH x 1 x H x W) --> (BATCH x H x W)
        if len(predicts.shape) == 4:
            predicts = predicts.squeeze(1)
        if len(targets.shape) == 4:
            targets = targets.squeeze(1)
        # Binarize masks and predictions
        predicts = predicts > self.thresh
        targets = targets > self.thresh

        pixel_diff = torch.logical_xor(targets, predicts).type(torch.int)
        # pixel_diff[pixel_diff == -1] =

        hit_px_list = []
        for m, d in zip(targets, pixel_diff):
            white_px_tar = len(torch.where(m == 1)[0])
            white_px_diff = len(torch.where(d == 1)[0])
            hit_pixel = white_px_tar - white_px_diff
            hit_pixel_ratio = hit_pixel / white_px_tar
            hit_px_list.append(hit_pixel_ratio)
        return hit_px_list
