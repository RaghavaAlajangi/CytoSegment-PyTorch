import pathlib
import tempfile
import zipfile

import torch


def get_data_dir():
    return pathlib.Path(__file__).parent / "data"


def get_test_tensors():
    # Create a test predict tensor
    predict = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                            1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
                            1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7])
    # Create a test target tensor
    target = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                           1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
                           1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7])
    # Convert test tensors into 2D tensors as if images and masks
    predict = predict.view(3, 1, 3, 3)
    target = target.view(3, 1, 3, 3)
    return predict, target


def find_data(path):
    """Find.rtdc data files in a directory"""
    path = pathlib.Path(path)
    rtdcfiles = [r for r in path.rglob("*.rtdc") if r.is_file()]
    files = [pathlib.Path(ff) for ff in rtdcfiles]
    return files


def retrieve_data(zip_file):
    """Extract contents of data zip file and return data files
    """
    zpath = pathlib.Path(__file__).resolve().parent / "data" / zip_file
    # unpack
    arc = zipfile.ZipFile(str(zpath))

    # extract all files to a temporary directory
    edest = tempfile.mkdtemp(prefix=zpath.name)
    arc.extractall(edest)

    # Load RT-DC dataset
    # find tdms files
    datafiles = find_data(edest)

    if len(datafiles) == 1:
        datafiles = datafiles[0]

    return datafiles


def retrieve_train_data_path(zip_file):
    zpath = pathlib.Path(__file__).resolve().parent / "data" / zip_file
    # unpack
    arc = zipfile.ZipFile(str(zpath))
    # extract all files to a temporary directory
    edest = tempfile.mkdtemp(prefix=zpath.name)
    arc.extractall(edest)
    path = pathlib.Path(edest)
    return path
