import numpy as np

from semanticsegmentor.ml_metrics import IoUCoeff

from .helper_methods import get_test_tensors


def test_IoUCoeff():
    metric = IoUCoeff()
    predict, target = get_test_tensors()
    score = metric(predict, target).mean()
    assert np.allclose(score, 0.733, atol=0.003, rtol=0)
