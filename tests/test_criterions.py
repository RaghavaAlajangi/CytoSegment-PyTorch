import numpy as np

from semanticsegmentor.ml_criterions import FocalTverskyLoss

from .helper_methods import get_test_tensors


def test_focaltverskyloss():
    alpha = 0.3
    gamma = 0.75
    criterion = FocalTverskyLoss(alpha, gamma)
    predict, target = get_test_tensors()
    test_loss = criterion(predict, target)
    assert np.allclose(test_loss, 0.2543, atol=0.3, rtol=0)
