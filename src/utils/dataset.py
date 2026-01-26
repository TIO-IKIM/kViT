# -*- coding: utf-8 -*-

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

from logging import Logger
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torchvision import transforms
from torchvision.transforms import v2
import nibabel as nib
import numpy as np
from utils.utilities import ifft, fft
import utils.subsample as subsample
from utils.transforms import apply_mask
from pathlib import Path
from natsort import natsorted


def random_complex_cutout(
    x: torch.Tensor, cutout_size: int = 32, min_cutouts: int = 1, max_cutouts: int = 8
) -> torch.Tensor:
    """
    Applies random rectangular cutouts to a complex-valued tensor.
    For each cutout, a random position is selected and a square region of size `cutout_size` is set to zero
    across all channels. The number of cutouts is randomly chosen between `min_cutouts` and `max_cutouts`.
    Args:
        x (torch.Tensor): Input tensor of shape (C, H, W), where C is the number of channels.
        cutout_size (int, optional): Size of each square cutout. Default is 32.
        min_cutouts (int, optional): Minimum number of cutouts to apply. Default is 1.
        max_cutouts (int, optional): Maximum number of cutouts to apply. Default is 8.
    Returns:
        torch.Tensor: Tensor with random cutouts applied, same shape as input.
    """
    C, H, W = x.shape
    num_cutouts = np.random.randint(min_cutouts, max_cutouts + 1)
    mask = torch.ones((H, W), dtype=torch.bool)
    for _ in range(num_cutouts):
        cy = np.random.randint(0, H - cutout_size + 1)
        cx = np.random.randint(0, W - cutout_size + 1)
        mask[cy : cy + cutout_size, cx : cx + cutout_size] = False

    for c in range(C):
        x[c][~mask] = 0

    return x


def undersample(
    x: torch.Tensor, center_fractions: list[float], accelerations: list[int]
) -> torch.Tensor:
    """
    Applies random undersampling to a k-space tensor using specified center fractions and accelerations.
    Based on https://github.com/facebookresearch/fastMRI.

    Args:
        x (torch.Tensor): Input k-space data tensor. Should be complex-valued.
        center_fractions (list[float]): List of fractions of low-frequency k-space to keep.
        accelerations (list[int]): List of acceleration factors for undersampling.

    Returns:
        torch.Tensor: Undersampled k-space tensor with the same shape as the input.
    """

    mask_func = subsample.RandomMaskFunc(
        center_fractions=center_fractions,
        accelerations=accelerations,
        allow_any_combination=True,
    )

    # Change data dimensions for undersampling: last dimension is 2 (real and imaginary)
    x = torch.stack([x.real, x.imag], dim=-1)

    kspace_masked = apply_mask(x, mask_func)

    x = kspace_masked[0]

    # Change data dimensions back to original shape
    x = x[..., 0] + 1j * x[..., 1]

    return x


def complex_augmentation(complex_image: torch.Tensor) -> torch.Tensor:
    """
    Apply data augmentation to the input image tensor.

    Parameters:
        image (torch.Tensor): Input image tensor.

    Returns:
        torch.Tensor: Augmented image tensor.
    """
    transforms_simple = v2.Compose(
        [
            v2.RandomHorizontalFlip(p=0.5),
        ]
    )
    transforms_complex = v2.Compose(
        [
            v2.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ]
    )

    complex_image = transforms_simple(complex_image)
    complex_image = transforms_complex(complex_image.abs()) * torch.exp(
        1j * transforms_complex(complex_image.angle())
    )

    return complex_image


class kSpaceDataset:
    """
    A dataset class for handling k-space MRI data organized by patient, supporting augmentation, cropping, and undersampling.
    Each item in the dataset corresponds to a patient and contains a bag of image slices, the patient label,
    patient ID, and a list of sequence identifiers.

    Args:
        data_csv (pd.DataFrame): DataFrame containing metadata and file paths for MRI slices.
        augmentation (bool, optional): If True, apply data augmentation in the image domain. Default is False.
        crop_size (int, optional): Size to center crop each slice to. Default is 256.
        undersampling (tuple or None, optional): Tuple specifying (center_fraction, acceleration) for k-space undersampling. Default is None.
        **kwargs: Additional keyword arguments.

    Attributes:
        csv (pd.DataFrame): Input DataFrame.
        crop_size (int): Crop size for slices.
        bags_df (pd.DataFrame): DataFrame grouped by patient, aggregating slice paths and labels.
        patient_ids (list): List of patient IDs.
        aug (bool): Whether to apply augmentation.
        undersample_factor (tuple or None): Undersampling parameters.

    Methods:
        __len__(): Returns the number of patients in the dataset.
        __getitem__(idx): Retrieves all slices for a given patient, applies preprocessing, and returns a tensor bag, label, patient ID, and sequence list.
        bag_tensor (torch.Tensor): Tensor containing all slices for the patient (bag).
        label (torch.Tensor): Target label for the patient.
        patient_id (str): Patient identifier.
        sequence_list (list): List of sequence names for each slice.
    """

    def __init__(
        self, data_csv, augmentation=False, crop_size=256, undersampling=None, **kwargs
    ):
        self.csv = data_csv
        self.crop_size = crop_size
        self.bags_df = self.csv.groupby("Patient_id").agg(list)
        self.patient_ids = self.bags_df.index.tolist()
        self.aug = augmentation
        self.undersample_factor = undersampling

    def __len__(self):
        """
        Get the number of examples in the dataset.
        """
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> tuple:
        """
        Retrieves a bag of slices and associated metadata for a given patient index.
            idx (int): Index of the patient in the dataset.
            tuple:
                - bag_tensor (torch.Tensor): Stacked tensor of slices for the patient, possibly augmented and cropped.
                - label (torch.Tensor): Target label tensor for the patient.
                - patient_id (Any): Identifier for the patient.
                - sequence_list (List[str]): List of sequence names for each slice in the bag.

        Parameters:
            idx (int): Index of the item.

        Returns:
            tuple: A tuple containing the input tensor (self.x) and the target tensor (self.y).
        """
        patient_id = self.patient_ids[idx]
        sequence_list = []

        # Retrieve the list of all slice paths for this patient
        instance_paths = natsorted(self.bags_df.loc[patient_id, "filename"])

        # The label is the same for all instances in a bag, so we just take the first one
        label = self.bags_df.loc[patient_id, "class"][0]

        # Combine label 1.0 and 2.0
        if label == 2.0:
            label = 1.0
        if label == 3.0:
            label = 2.0

        bag_of_slices = []

        for path in instance_paths:
            sequence_list.append("_".join(Path(path).name.split("_")[:-3]))

            x = torch.tensor(np.load(path), dtype=torch.cfloat)

            x = x.unsqueeze(0)  # Add channel dimension

            if self.aug:
                x = ifft(x)  # Convert to image domain
                x = complex_augmentation(x)  # Apply data augmentation
                x = fft(x)  # Convert back to k-space domain
                x = transforms.RandomApply(
                    [lambda x: x + torch.randn_like(x) * 0.01], p=0.3
                )(x)
                x = transforms.RandomApply(
                    [lambda x: random_complex_cutout(x, cutout_size=32)], p=0.5
                )(x)
                x = undersample(
                    x, center_fractions=[0.04, 0.08], accelerations=[0, 2, 4]
                )

            if x.shape != (self.crop_size, self.crop_size):
                x = transforms.CenterCrop((self.crop_size, self.crop_size))(x)

            if self.undersample_factor:
                x = undersample(
                    x,
                    center_fractions=[self.undersample_factor[0]],
                    accelerations=[self.undersample_factor[1]],
                )

            # Normalize the image tensor
            x = (x - x.mean()) / (x.std())

            bag_of_slices.append(x)

        bag_tensor = torch.stack(bag_of_slices, dim=0)

        return (
            bag_tensor,
            torch.tensor(label, dtype=torch.long),
            patient_id,
            sequence_list,
        )


class ImageDataset(kSpaceDataset):
    """
    A PyTorch Dataset for loading and processing multi-instance medical images grouped by patient.
    Each item in the dataset corresponds to a patient and contains a bag of image slices, the patient label,
    patient ID, and a list of sequence identifiers.

    Args:
        data_csv (pd.DataFrame): DataFrame containing image file paths and labels, grouped by 'Patient_id'.
        augmentation (bool, optional): Whether to apply data augmentation to the images. Default is False.
        crop_size (int, optional): Size to which images are center-cropped. Default is 224.
        undersampling (tuple or None, optional): Tuple specifying undersampling parameters (center_fraction, acceleration).
        **kwargs: Additional keyword arguments.

    Attributes:
        csv (pd.DataFrame): The input DataFrame.
        crop_size (int): Crop size for images.
        bags_df (pd.DataFrame): DataFrame grouped by patient, aggregating file paths and labels.
        patient_ids (list): List of patient IDs in the dataset.
        aug (bool): Flag for augmentation.
        undersample_factor (tuple or None): Undersampling parameters.

    Methods:
        __len__(): Returns the number of patients in the dataset.
        __getitem__(idx): Loads and processes all slices for a given patient, applies augmentation and undersampling
                          if specified, normalizes the images, and returns a stacked tensor of slices, the label,
                          patient ID, and sequence list.
        tuple: (bag_tensor, label_tensor, patient_id, sequence_list)
            bag_tensor (torch.Tensor): Stacked tensor of processed image slices for the patient.
            label_tensor (torch.Tensor): Tensor containing the patient's label.
            patient_id (str): The patient identifier.
            sequence_list (list): List of sequence identifiers for each slice.
    """

    def __init__(
        self, data_csv, augmentation=False, crop_size=224, undersampling=None, **kwargs
    ):
        self.csv = data_csv
        self.crop_size = crop_size
        self.bags_df = self.csv.groupby("Patient_id").agg(list)
        self.patient_ids = self.bags_df.index.tolist()
        self.aug = augmentation
        self.undersample_factor = undersampling

    def __len__(self):
        """
        Get the number of examples in the dataset.
        """
        return len(self.patient_ids)

    def __getitem__(self, idx):
        """
        Retrieves a bag of slices and associated metadata for a given patient index.
            idx (int): Index of the patient in the dataset.
            tuple:
                - bag_tensor (torch.Tensor): Stacked tensor of slices for the patient, possibly augmented and cropped.
                - label (torch.Tensor): Target label tensor for the patient.
                - patient_id (Any): Identifier for the patient.
                - sequence_list (List[str]): List of sequence names for each slice in the bag.

        Parameters:
            idx (int): Index of the item.

        Returns:
            tuple: A tuple containing the input tensor (self.x) and the target tensor (self.y).
        """
        patient_id = self.patient_ids[idx]
        sequence_list = []

        instance_paths = natsorted(self.bags_df.loc[patient_id, "filename"])

        # Retrieve the list of all slice paths for this patient
        instance_paths = self.bags_df.loc[patient_id, "filename"]

        # The label is the same for all instances in a bag, so we just take the first one
        label = self.bags_df.loc[patient_id, "class"][0]

        # Combine label 1.0 and 2.0
        if label == 2.0:
            label = 1.0
        if label == 3.0:
            label = 2.0

        bag_of_slices = []

        for path in instance_paths:
            sequence_list.append("_".join(Path(path).name.split("_")[:-3]))

            x = torch.tensor(nib.load(path).get_fdata(), dtype=torch.float32)
            x = x.unsqueeze(0)  # Add channel dimension

            if self.aug:
                x = complex_augmentation(x)  # Apply data augmentation
                x = fft(x)  # Convert back to k-space domain
                x = transforms.RandomApply(
                    [lambda x: x + torch.randn_like(x) * 0.01], p=0.3
                )(x)
                x = transforms.RandomApply(
                    [lambda x: random_complex_cutout(x, cutout_size=32)], p=0.5
                )(x)
                x = ifft(x).abs()

            if x.shape != (self.crop_size, self.crop_size):
                x = fft(x)
                x = transforms.CenterCrop((self.crop_size, self.crop_size))(x)
                x = ifft(x).abs()

            if self.undersample_factor:
                x = fft(x)
                x = undersample(
                    x,
                    center_fractions=[self.undersample_factor[0]],
                    accelerations=[self.undersample_factor[1]],
                )
                x = ifft(x).abs()

            # Normalize the image tensor
            x = (x - x.mean()) / (x.std() + 1e-13)
            bag_of_slices.append(x)

        bag_tensor = torch.stack(bag_of_slices, dim=0)

        return (
            bag_tensor,
            torch.tensor(label, dtype=torch.long),
            patient_id,
            sequence_list,
        )


def collate_bags(batch):
    """
    Custom collate function to handle bags of varying sizes.
    It returns a list of tensors (each tensor is a bag) and a tensor of labels.
    """
    bags = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    id = [item[2] for item in batch]
    seq_list = [item[3] for item in batch]

    if len(batch[0]) == 6:
        slice_labels = [item[4] for item in batch]
        modality_ids = [item[5] for item in batch]
        return (
            bags,
            torch.stack(labels),
            id,
            seq_list,
            torch.tensor(slice_labels),
            modality_ids,
        )
    else:
        return bags, torch.stack(labels), id, seq_list


class FastMriDataset2D(Dataset):
    """
    PyTorch Dataset for loading and preprocessing 2D FastMRI data from a CSV file.

    Args:
        data_csv (pd.DataFrame): DataFrame containing file paths and labels.
        augmentation (bool, optional): Whether to apply data augmentation. Default is False.
        crop_size (int, optional): Size to center crop the images/k-space data. Default is 256.
        kspace (bool, optional): If True, returns data in k-space domain; otherwise, returns image domain. Default is True.
        hybrid (bool, optional): If True, returns both k-space and image domain data stacked. Default is False.
        undersample (tuple or None, optional): Undersampling parameters as (center_fraction, acceleration). Default is None.
        type (str, optional): Dataset type, either "knee" or "prostate". Default is "knee".
        **kwargs: Additional keyword arguments.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Loads and preprocesses the sample at the given index.

    Returns:
        tuple: (processed_data, label)
            processed_data (torch.Tensor): Preprocessed k-space or image domain data.
            label (torch.Tensor): Corresponding label for the sample.

    Notes:
        - Applies inverse FFT and FFT for domain conversion.
        - Supports complex-valued augmentation and random cutout.
        - Normalizes each channel of the output.
        - For "prostate" type, binarizes labels (<=2 as 0, >2 as 1).
    """

    def __init__(
        self,
        data_csv,
        augmentation=False,
        crop_size=256,
        kspace=True,
        hybrid=False,
        undersample=None,
        type: str = "knee",
        **kwargs,
    ):
        assert type in {"knee", "prostate"}, "type must be 'knee' or 'prostate'"
        self.csv = data_csv
        self.crop_size = crop_size
        self.aug = augmentation
        self.kspace = kspace
        self.hybrid = hybrid
        self.undersample_factor = undersample
        self.type = type

    def __len__(self):
        """
        Get the number of examples in the dataset.
        """
        return len(self.csv)

    def __getitem__(self, idx):
        """
        Get a specific item from the dataset.

        Parameters:
            idx (int): Index of the item.
        """
        x = torch.load(self.csv.loc[idx, "filename"], weights_only=False)
        if isinstance(x, dict):
            x = x["kspace"]

        x = x.cfloat().unsqueeze(0)

        label = torch.tensor(self.csv.loc[idx, "label"], dtype=torch.long)

        if self.type == "prostate":
            if label <= 2:
                label = 0
            else:
                label = 1

        x = ifft(x)  # Convert to image domain

        if self.aug:
            x = complex_augmentation(x)

        x = fft(x)  # Convert back to k-space domain

        if x.shape != (self.crop_size, self.crop_size):
            x = transforms.CenterCrop((self.crop_size, self.crop_size))(x)

        if self.aug:
            # Apply random cutout for complex-valued data
            x = transforms.RandomApply(
                [lambda x: x + torch.randn_like(x) * 0.01], p=0.3
            )(x)
            x = random_complex_cutout(x, cutout_size=32)

        if self.undersample_factor:
            x = undersample(
                x,
                center_fractions=[self.undersample_factor[0]],
                accelerations=[self.undersample_factor[1]],
            )

        if not self.kspace:
            x = ifft(x)
            x = x.abs()

        if self.hybrid:
            x = torch.stack([x, ifft(x)]).squeeze()

        for channel in range(x.shape[0]):
            x[channel] = (x[channel] - x[channel].mean()) / (x[channel].std() + 1e-13)

        return x, label


def get_loaders(
    csv_path: str,
    train_idx: list,
    val_idx: list,
    kspace: bool,
    crop_size: int = 256,
    batch_size: int = 16,
    num_workers: int = 5,
    pin_memory: bool = True,
    log: Logger | None = None,
    twod: bool = False,
    hybrid: bool = False,
    dataset_name: str = "FastMriDataset2D",
) -> tuple[DataLoader, DataLoader, int]:
    """
    Creates and returns DataLoader instances for training and validation datasets.

    Args:
        csv_path (str): Path to the CSV file containing dataset information.
        train_idx (list): List of indices for the training set.
        val_idx (list): List of indices for the validation set.
        kspace (bool): Whether to use k-space data.
        crop_size (int, optional): Size to crop images to. Defaults to 256.
        batch_size (int, optional): Number of samples per batch. Defaults to 16.
        num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to 5.
        pin_memory (bool, optional): If True, the data loader will copy Tensors into CUDA pinned memory before returning them. Defaults to True.
        log (Logger, optional): Logger instance for logging information. Defaults to None.
        twod (bool, optional): Whether to use 2D dataset mode. Defaults to False.
        hybrid (bool, optional): Whether to use hybrid mode. Defaults to False.
        dataset_name (str, optional): Name of the dataset class to use. Options: "FastMriDataset2D", "FastMriDataset",
                                     "ImageDataset", "kSpaceDataset". Defaults to "FastMriDataset2D".

    Returns:
        tuple[DataLoader, DataLoader, int]: A tuple containing the training DataLoader, validation DataLoader, and the total number of examples.
    """

    # Map dataset names to classes
    dataset_map = {
        "FastMriDataset_Knee": FastMriDataset2D,
        "FastMriDataset_Prostate": FastMriDataset2D,
        "ImageDataset": ImageDataset,
        "kSpaceDataset": kSpaceDataset,
    }

    if dataset_name not in dataset_map:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available options: {list(dataset_map.keys())}"
        )
    DatasetClass = dataset_map[dataset_name]

    csv = pd.read_csv(csv_path)
    train_files = csv.iloc[train_idx].reset_index()
    val_files = csv.iloc[val_idx].reset_index()

    # Create train and validation datasets
    if DatasetClass == FastMriDataset2D:
        if "knee" in dataset_name.lower():
            dataset_type = "knee"
        elif "prostate" in dataset_name.lower():
            dataset_type = "prostate"
        train_ds = DatasetClass(
            data_csv=train_files,
            augmentation=True,
            crop_size=crop_size,
            kspace=kspace,
            train=True,
            hybrid=hybrid,
            type=dataset_type,
        )
        val_ds = DatasetClass(
            data_csv=val_files,
            augmentation=False,
            crop_size=crop_size,
            kspace=kspace,
            train=False,
            hybrid=hybrid,
            type=dataset_type,
        )
    else:
        # For FastMriDataset, ImageDataset, kSpaceDataset
        train_ds = DatasetClass(
            data_csv=train_files, augmentation=True, crop_size=crop_size, kspace=kspace
        )
        val_ds = DatasetClass(
            data_csv=val_files, augmentation=False, crop_size=crop_size, kspace=kspace
        )

    # Create DataLoader instances
    loader_params = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "shuffle": True,
        "prefetch_factor": 1,
        "persistent_workers": True,
    }
    if twod:
        loader_params["collate_fn"] = None
        loader_params["drop_last"] = True
    else:
        loader_params["collate_fn"] = collate_bags

    train_loader = DataLoader(train_ds, **loader_params)
    val_loader = DataLoader(val_ds, **loader_params)

    if log is not None:
        log.info(
            f"Found {len(train_ds)} examples in the training-set; {len(val_ds)} examples in the validation-set ..."
        )
        log.info(f"Using dataset: {dataset_name}")

    return train_loader, val_loader, (len(train_ds) + len(val_ds))


def get_test_loader(
    csv_path: str,
    batch_size: int = 4,
    num_workers: int = 5,
    crop_size: int = 256,
    pin_memory: bool = True,
    kspace: bool = False,
    log: Logger | None = None,
    dataset_type: str = "",
    hybrid: bool = False,
    dataset_name: str = "FastMriDataset",
    undersample_list: list[int, float] = [0, 1.0],
) -> DataLoader:
    """
    Creates and returns a PyTorch DataLoader for testing, based on the specified dataset and configuration.

    Args:
        csv_path (str): Path to the CSV file containing dataset information.
        batch_size (int, optional): Number of samples per batch. Defaults to 4.
        num_workers (int, optional): Number of subprocesses for data loading. Defaults to 5.
        crop_size (int, optional): Size to crop the images. Defaults to 256.
        pin_memory (bool, optional): Whether to use pinned memory in DataLoader. Defaults to True.
        kspace (bool, optional): If True, loads k-space data. Defaults to False.
        log (Logger, optional): Logger for logging dataset information. Defaults to None.
        undersample (list, optional): Undersampling configuration. Defaults to None.
        dataset_type (str, optional): Type of dataset, e.g., "2D" or "MIL". Defaults to "".
        hybrid (bool, optional): If True, enables hybrid data loading. Defaults to False.
        dataset_name (str, optional): Name of the dataset class to use. Defaults to "FastMriDataset2D".
        undersample_list (list[int, float], optional): List containing undersampling rate and center fraction. Defaults to [0, 1.0].

    Returns:
        DataLoader: PyTorch DataLoader for the test dataset.

    Raises:
        ValueError: If an unknown dataset_name is provided.
    """

    dataset_map = {
        "FastMriDataset_Knee": FastMriDataset2D,
        "FastMriDataset_Prostate": FastMriDataset2D,
        "ImageDataset": ImageDataset,
        "kSpaceDataset": kSpaceDataset,
    }
    if dataset_name not in dataset_map:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available options: {list(dataset_map.keys())}"
        )

    DatasetClass = dataset_map[dataset_name]
    csv = pd.read_csv(csv_path)

    if DatasetClass == FastMriDataset2D:
        dataset_type = "knee" if "knee" in dataset_name.lower() else "prostate"
        test_ds = DatasetClass(
            data_csv=csv.reset_index(),
            augmentation=False,
            crop_size=crop_size,
            kspace=kspace,
            hybrid=hybrid,
            undersample=undersample_list,
            type=dataset_type,
        )
    else:
        test_ds = DatasetClass(
            data_csv=csv,
            augmentation=False,
            crop_size=crop_size,
            kspace=kspace,
            undersample=undersample_list,
        )

    loader_params = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "shuffle": False,
        "prefetch_factor": 1,
        "persistent_workers": True,
    }
    if dataset_type == "2D":
        loader_params["collate_fn"] = None
        loader_params["drop_last"] = False
    elif dataset_type == "MIL":
        loader_params["collate_fn"] = collate_bags

    test_loader = DataLoader(test_ds, **loader_params)

    if log is not None:
        log.info(
            f"Found {len(test_ds)} examples in the test-set using dataset: {dataset_name}"
        )

    return test_loader


class Folds:
    """
    Folds class for creating cross-validation folds from a CSV file.

    Attributes:
        folds (list): List of train/test indices for each fold.
    """

    def __init__(self, csv_path, n_folds):
        """
        Initializes the Folds class with the given CSV path and number of folds.

        Args:
            csv_path (str): Path to the CSV file containing the data.
            n_folds (int): Number of folds for cross-validation.
        """
        df = pd.read_csv(csv_path)
        x = df["filename"]
        y = df["label"]
        group = df["Patient_id"]

        le = LabelEncoder()
        group = le.fit_transform(group)

        group_kfold = StratifiedGroupKFold(
            n_splits=n_folds, random_state=42, shuffle=True
        )

        self.folds = list(group_kfold.split(x, y, group))

    def __len__(self):
        """
        Returns the number of folds.

        Returns:
            int: Number of folds.
        """
        return len(self.folds)

    def __getitem__(self, idx):
        """
        Retrieves the train/test indices for the given fold index.

        Args:
            idx (int): Index of the fold.

        Returns:
            tuple: Train/test indices for the given fold.
        """
        return self.folds[idx]
