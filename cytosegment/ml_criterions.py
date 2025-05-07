"""
Some observations that may be useful to experimenting with different loss
functions.

Tversky and Focal-Tversky loss benefit from very low learning rates, of the
order 5e-5 to 1e-4. They would not see much improvement in my kernels until
around 7-10 epochs, upon which performance would improve significantly.

In general, if a loss function does not appear to be working well (or at
all), experiment with modifying the learning rate before moving on to other
options.

You can easily create your own loss functions by combining any of the above
with Binary Cross-Entropy or any combination of other losses. Bear in mind
that loss is calculated for every batch, so more complex losses will increase
runtime.

Care must be taken when writing loss functions for PyTorch. If you call a
function to modify the inputs that doesn't entirely use PyTorch's numerical
methods, the tensor will 'detach' from the graph that maps it back
through the neural network for the purposes of backpropagation, making the
loss function unusable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as fuc


def get_criterion_with_params(params):
    assert {"criterion"}.issubset(params)
    criterion_params = params.get("criterion")
    assert {"type"}.issubset(criterion_params)
    criterion_type = criterion_params.get("type")

    if criterion_type.lower() == "diceloss":
        return DiceLoss()

    if criterion_type.lower() == "dicebceloss":
        return DiceBCELoss()

    if criterion_type.lower() == "iouloss":
        return IoULoss()

    if criterion_type.lower() == "focalloss":
        assert {"alpha", "gamma"}.issubset(criterion_params)
        alpha = criterion_params.get("alpha")
        gamma = criterion_params.get("gamma")
        return FocalLoss(alpha, gamma)

    if criterion_type.lower() == "tverskytoss":
        assert {"alpha", "beta"}.issubset(criterion_params)
        alpha = criterion_params.get("alpha")
        beta = criterion_params.get("beta")
        return TverskyLoss(alpha, beta)

    if criterion_type.lower() == "focaltverskyloss":
        assert {"alpha", "gamma"}.issubset(criterion_params)
        alpha = criterion_params.get("alpha")
        gamma = criterion_params.get("gamma")
        return FocalTverskyLoss(alpha, gamma)


class DiceLoss(nn.Module):
    """
    Dice Loss:
    The Dice coefficient, or Dice-Sørensen coefficient, is a common metric for
    pixel segmentation that can also be modified to act as a loss function
    """

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, predicts, targets):
        # flatten label and prediction tensors
        predicts = predicts.view(-1)
        targets = targets.view(-1)

        intersection = (predicts * targets).sum()
        dice = (2.0 * intersection + self.eps) / (
            predicts.sum() + targets.sum() + self.eps
        )

        return 1 - dice


class DiceBCELoss(nn.Module):
    """
    BCE-Dice Loss:
    This loss combines Dice loss with the standard binary cross-entropy (BCE)
    loss that is generally the default for segmentation models. Combining the
    two methods allows for some diversity in the loss, while benefitting from
    the stability of BCE. The equation for multi-class BCE by itself will
    be familiar to anyone who has studied logistic regression.
    """

    def __init__(self):
        super(DiceBCELoss, self).__init__()
        self.eps = 1e-6

    def forward(self, predicts, targets):
        # flatten label and prediction tensors
        predicts = predicts.view(-1)
        targets = targets.view(-1)

        intersection = (predicts * targets).sum()
        dice_loss = 1 - (2.0 * intersection + self.eps) / (
            predicts.sum() + targets.sum() + self.eps
        )
        bce = fuc.binary_cross_entropy(predicts, targets, reduction="mean")
        return bce + dice_loss


class IoULoss(nn.Module):
    """
    Jaccard/Intersection over Union (IoU) Loss:
    The IoU metric, or Jaccard Index, is similar to the Dice metric and is
    calculated as the ratio between the overlap of the positive instances
    between two sets, and their mutual combined values. Like the Dice metric,
    it is a common means of evaluating the performance of pixel segmentation
    models.
    """

    def __init__(self):
        super(IoULoss, self).__init__()
        self.eps = 1e-6

    def forward(self, predicts, targets):
        # flatten label and prediction tensors
        predicts = predicts.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (predicts * targets).sum()
        total = (predicts + targets).sum()
        union = total - intersection
        return 1 - (intersection + self.eps) / (union + self.eps)


class FocalLoss(nn.Module):
    """
    Focal Loss:
    Focal Loss was introduced by Lin et al. of Facebook AI Research
    in 2017 as a means of combatting extremely imbalanced datasets where
    positive cases were relatively rare. Their paper "Focal Loss for Dense
    Object Detection" is retrievable here: https://arxiv.org/abs/1708.02002.
    In practice, the researchers used an alpha-modified version of the
    function, so it has been included in this implementation.
    """

    def __init__(self, alpha=0.8, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = 1e-10

    def forward(self, predicts, targets):
        # flatten label and prediction tensors
        predicts = predicts.view(-1)
        targets = targets.view(-1)

        # first compute binary cross-entropy
        bce = fuc.binary_cross_entropy(predicts, targets, reduction="mean")
        bce_exp = torch.exp(-bce)
        focal_loss = self.alpha * (1 - bce_exp) ** self.gamma * bce

        return focal_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss This loss was introduced in "Tversky loss function for image
    segmentationusing 3D fully convolutional deep networks", retrievable
    here: https://arxiv.org/abs/1706.05721. It was designed to optimise
    segmentation on imbalanced medical datasets by utilising constants that
    can adjust how harshly different types of error are penalised in the loss
    function. From the paper:

    ... in the case of α=β=0.5 the Tversky index simplifies to be the same as
    the Dice coefficient, which is also equal to the F1 score. With α=β=1,
    Equation 2 produces Tanimoto coefficient, and setting α+β=1 produces the
    set of Fβ scores. Larger βs weigh recall higher than precision (by placing
    more emphasis on false negatives).

    To summarise, this loss function is weighted by the constants 'alpha' and
    'beta' that penalise false positives and false negatives respectively to a
    higher degree in the loss function as their value is increased. The beta
    constant in particular has applications in situations where models can
    obtain misleadingly positive performance via highly conservative
    prediction. You may want to experiment with different values to find the
    optimum. With alpha==beta==0.5, this loss becomes equivalent to Dice Loss.
    """

    def __init__(self, alpha=0.5, beta=0.5):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-6

    def forward(self, predicts, targets):
        # flatten label and prediction tensors
        predicts = predicts.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        tp = (predicts * targets).sum()
        fp = ((1 - targets) * predicts).sum()
        fn = (targets * (1 - predicts)).sum()

        tversky = (tp + self.eps) / (
            tp + self.alpha * fp + self.beta * fn + self.eps
        )
        return 1 - tversky


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss:
    A variant on the Tversky loss that also includes the
    gamma modifier from Focal Loss.
    """

    def __init__(self, alpha=0.3, gamma=2):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.gamma = 1 / gamma
        self.eps = 1e-7

    def forward(self, predicts, targets):
        # flatten label and prediction tensors
        # predicts = predicts.view(-1)
        # targets = targets.view(-1)

        predicts = predicts.flatten()
        targets = targets.flatten()

        # True Positives, False Positives & False Negatives
        tp = (predicts * targets).sum()
        fn = ((1 - targets) * predicts).sum()
        fp = (targets * (1 - predicts)).sum()
        tversky = (tp + self.eps) / (
            self.eps + tp + self.alpha * fp + (1 - self.alpha) * fn
        )
        focal_tversky = (1 - tversky) ** self.gamma
        return focal_tversky
