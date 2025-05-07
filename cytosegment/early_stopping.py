import numpy as np


class EarlyStopping:
    """
    EarlyStopping stops the training if validation loss does not improve
    after a given patience.
    """

    def __init__(self, patience=10):
        """
        Parameters
        ----------
        patience: int
            Epochs to wait until last validation loss improved
            Default: 10
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            print(
                f"EarlyStopping counter: {self.counter} "
                f"out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0
