from pathlib import Path
import random
import zipfile

import numpy as np
from PIL import Image
import torch
from torch import mean as tmean
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tt
import torchvision.transforms.functional as tf


def get_dataloaders_with_params(params):
    assert {"dataset"}.issubset(params)
    dataset_params = params.get("dataset")
    assert {"type"}.issubset(dataset_params)
    # data_type = dataset_params.get("type")

    assert {"data_path", "augmentation"}.issubset(dataset_params)
    assert {"valid_size", "batch_size"}.issubset(dataset_params)
    assert {"mean", "std", "num_workers"}.issubset(dataset_params)
    assert {"random_seed"}.issubset(dataset_params)

    data_path = dataset_params.get("data_path")
    augmentation = dataset_params.get("augmentation")
    valid_size = dataset_params.get("valid_size")
    batch_size = dataset_params.get("batch_size")
    img_size = dataset_params.get("img_size")
    mean = dataset_params.get("mean")
    std = dataset_params.get("std")
    num_workers = dataset_params.get("num_workers")
    random_seed = dataset_params.get("random_seed")

    train_data_path, test_data_path = unzip_data(data_path)

    train_images, train_masks = read_data(train_data_path, seed=random_seed,
                                          shuffle=True)
    test_images, test_masks = read_data(test_data_path, seed=42, shuffle=False)

    train_imgs, valid_imgs, train_msks, valid_msks = split_data(train_images,
                                                                train_masks,
                                                                valid_size)

    # Create training dataset instance
    train_dataset = UNetDataset(train_imgs, train_msks, target_shape=img_size,
                                augment=augmentation, mean=mean, std=std)
    # Create validation dataset instance and make sure augmentation is False
    valid_dataset = UNetDataset(valid_imgs, valid_msks, target_shape=img_size,
                                augment=False, mean=mean, std=std)
    # Create testing dataset instance
    test_dataset = UNetDataset(test_images, test_masks, target_shape=img_size,
                               augment=False, mean=mean, std=std)

    data_dict = {
        "train": train_dataset,
        "valid": valid_dataset,
        "test": test_dataset,
    }
    # Training data will be shuffled
    dataloader_dict = create_dataloaders(data_dict, batch_size, num_workers)
    return dataloader_dict


def unzip_data(zipped_data_path):
    """Unzip data path and return train and test data paths"""

    # Create output path from input path
    pathout = Path(zipped_data_path).with_suffix("")

    # Create train and test datasets output paths
    train_data_path = pathout / "training"
    test_data_path = pathout / "testing"

    # Extract zipped file, if train and test dirs are not existed.
    if not train_data_path.exists() or not test_data_path.exists():
        with zipfile.ZipFile(zipped_data_path, "r") as zip_ref:
            zip_ref.extractall(pathout.parents[0])

    return train_data_path, test_data_path


def verify_image_file(file_path):
    """Verify if the image is valid."""
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except (IOError, SyntaxError) as e:
        print(f"Excluding corrupted file: ({file_path})")
        return False


def intersection_of_images_and_masks(image_paths, mask_paths):
    # Convert paths to sets of filenames for intersection
    image_filenames = {img.stem for img in
                       image_paths}  # .stem gets the filename without extension
    mask_filenames = {mask.stem for mask in mask_paths}

    # Find the common files (intersection)
    common_filenames = image_filenames & mask_filenames

    # Filter the images and masks based on the intersection
    filtered_images = [img for img in image_paths if
                       img.stem in common_filenames]
    filtered_masks = [mask for mask in mask_paths if
                      mask.stem in common_filenames]

    return filtered_images, filtered_masks


def read_data(data_path, seed=42, shuffle=False):
    """Read images and masks, verify, and return valid pairs."""
    img_path = Path(data_path) / "images"
    msk_path = Path(data_path) / "masks"

    # Get the list of all PNG files
    img_list = sorted([p for p in img_path.rglob("*.png") if p.is_file()])
    msk_list = sorted([p for p in msk_path.rglob("*.png") if p.is_file()])

    # Verify and filter corrupted files
    valid_img_list = [img for img in img_list if verify_image_file(img)]
    valid_msk_list = [msk for msk in msk_list if verify_image_file(msk)]

    if len(valid_img_list) != len(valid_msk_list):
        print(f"Warning: After verification, the number of valid images "
              f"({len(valid_img_list)}) and masks ({len(valid_msk_list)}) is "
              f"different.")
        valid_img_list, valid_msk_list = intersection_of_images_and_masks(
            valid_img_list, valid_msk_list)
        print(f"Using {len(valid_img_list)} common valid images and masks.")

    if shuffle:
        # Shuffle the valid image and mask lists
        random.seed(seed)
        random.shuffle(valid_img_list)
        random.seed(seed)
        random.shuffle(valid_msk_list)

    images = []
    masks = []

    for img_path, msk_path in zip(valid_img_list, valid_msk_list):
        img = np.array(Image.open(img_path))
        msk = np.array(Image.open(msk_path))
        images.append(img)
        masks.append(msk)

    return images, masks


def split_data(images, masks, valid_size=0.2):
    assert len(images) == len(masks)
    # Get the length of the dataset
    len_dataset = len(images)
    # Compute the train samples
    train_img_samples = int((1 - valid_size) * len_dataset)
    # Slicing train and valid samples
    train_imgs = images[:train_img_samples]
    test_imgs = images[train_img_samples:]

    train_msks = masks[:train_img_samples]
    test_msks = masks[train_img_samples:]

    return train_imgs, test_imgs, train_msks, test_msks


def create_dataloaders(data_dict, batch_size, num_workers=0):
    dataloader_dict = dict()
    for k in data_dict.keys():
        dataloader_dict[k] = DataLoader(
            data_dict[k],
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True if k == "train" else False
        )
    return dataloader_dict


def compute_mean_std(data_path, img_size):
    """ Computes the mean and standard deviation of a dataset.
    Parameters
    ----------
    data_path: str or Path
        Data directory path that has images and masks directories
    img_size: tuple
        Desired image size. Image samples are padded or cropped according
        to the img_size automatically

    Returns
    -------
    The mean and standard deviation of the training data
    """
    images, masks = read_data(data_path, seed=42, shuffle=False)
    data_dict = {"data": UNetDataset(images, masks, target_shape=img_size,
                                     augment=False)
                 }

    # Training data will be shuffled
    dataloader_dict = create_dataloaders(data_dict, batch_size=8)

    channel_sum, channel_square_sum, batch_counter = 0, 0, 0

    for imgs, _ in dataloader_dict["data"]:
        channel_sum += tmean(imgs, dim=[0, 2, 3])
        channel_square_sum += tmean(imgs ** 2, dim=[0, 2, 3])

        batch_counter += 1

    mean = float(channel_sum / batch_counter)
    std = float((channel_square_sum / batch_counter - mean ** 2) ** 0.5)
    return mean, std


class UNetDataset(Dataset):
    """ Create torch dataset instance for training"""

    def __init__(self, images, masks, target_shape, augment=False,
                 min_max=False, mean=None, std=None):
        self.images = images
        self.masks = masks
        self.target_shape = target_shape
        self.augment = augment
        self.min_max = min_max
        self.mean = 0. if mean is None else mean
        self.std = 1. if std is None else std

    @staticmethod
    def min_max_norm(img):
        norm_np_img = (img - img.min()) / (img.max() - img.min())
        norm_ten_img = torch.tensor(norm_np_img, dtype=torch.float32)
        return norm_ten_img.unsqueeze(0)

    def crop_pad_sample(self, image, mask, pad_value=0):
        height, width = image.shape
        target_height, target_width = self.target_shape

        # don't do cropping and padding if actual shape equal to target shape
        if (height, width) == (target_height, target_width):
            return image, mask

        # Calculate the difference in height and width
        height_diff = height - target_height
        width_diff = width - target_width

        # Adjust image height (crop or pad according to the target height)
        if height_diff > 0:
            # Cropping
            hcorr = abs(height_diff) // 2
            hcorr_img = image[hcorr: height - hcorr, :]
            hcorr_msk = mask[hcorr: height - hcorr, :]

        else:
            # Padding
            hpad = abs(height_diff) // 2
            hcorr_img = np.full((target_height, width), pad_value,
                                dtype=np.float32)
            hcorr_msk = np.zeros((target_height, width), dtype=np.float32)
            hcorr_img[hpad:hpad + height, :] = hcorr_img
            hcorr_msk[hpad:hpad + height, :] = hcorr_msk

        # Adjust image width (crop or pad according to the target width)
        if width_diff > 0:
            # Cropping
            wcorr = abs(width_diff) // 2
            wcorr_img = hcorr_img[:, wcorr: width - wcorr]
            wcorr_msk = hcorr_msk[:, wcorr: width - wcorr]
        else:
            # Padding
            wpad = abs(width_diff) // 2
            wcorr_img = np.full((target_height, target_width), pad_value,
                                dtype=np.float32)
            wcorr_msk = np.zeros((target_height, target_width),
                                 dtype=np.float32)
            wcorr_img[:, wpad:wpad + width] = hcorr_img
            wcorr_msk[:, wpad:wpad + width] = hcorr_msk

        return wcorr_img, wcorr_msk

    def custom_transform(self, image, mask):
        # Instantiate normalize and to_tensor functions
        normalize = tt.Normalize([self.mean], [self.std])
        to_tensor = tt.ToTensor()

        # Make sure mask is binary and tensor
        mask = mask / mask.max()
        mask = torch.tensor(mask, dtype=torch.float32)

        if self.min_max:
            image = self.min_max_norm(image)
        else:
            # Normalize image (divides the image with 255)
            image = to_tensor(image.astype("uint8"))

        # Standardize image with mean and std values
        image = normalize(image)

        if self.augment:
            # Random horizontal flipping
            if random.random() >= 0.5:
                image = tf.hflip(image)
                mask = tf.hflip(mask)
            # Random vertical flipping
            if random.random() >= 0.5:
                image = tf.vflip(image)
                mask = tf.vflip(mask)

            # Apply brightness (add/subtract a random number from the image)
            if random.random() >= 0.5:
                brightness_factor = random.uniform(-1, 1)
                image = image + brightness_factor

        return image, mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]

        # Compute image mean
        pad_value = image.mean()

        # Resize the image and mask sample according to the target shape
        resized_img, resized_msk = self.crop_pad_sample(image, mask,
                                                        pad_value=pad_value)
        # Augmentation
        aug_img, aug_msk = self.custom_transform(resized_img, resized_msk)

        return aug_img, aug_msk
