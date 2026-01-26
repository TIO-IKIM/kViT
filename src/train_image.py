# -*- coding: utf-8 -*-

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import os
import random
import shutil
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import yaml
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit

from utils.dataset import get_loaders, Folds
from utils.metrics import ValidationMetrics
from utils.IKIMLogger import IKIMLogger
import utils.utilities as utilities
from model import image_classification as model
from model.image_classification import ClassificationModel2D


def parse_args():
    """
    Parse command-line arguments for training configuration.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(prog="Training")
    parser.add_argument("-e", type=int, default=50, help="Number of epochs")
    parser.add_argument("--tqdm", action="store_false", help="Disable tqdm logging")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id")
    parser.add_argument(
        "--config",
        type=str,
        default="/home/jovyan/k-radiomics-storage/kspace-radiomics/src/configs/train_unified_2D.yaml",
        help="Config file",
    )
    parser.add_argument("-c", type=str, default=None, help="Checkpoint folder")
    parser.add_argument(
        "-s",
        "--single_fold",
        action="store_false",
        help="Use single fold (default: cross-validation)",
    )
    return parser.parse_args()


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across numpy, random, torch, and CUDA.
    Args:
        seed (int): The random seed to use.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)


class TrainNetwork:
    """
    Main class for training and validating a 2D or MIL image classification model.
    Handles cross-validation or single split, checkpointing, and logging.
    """

    def __init__(self, args, config):
        """
        Initialize training configuration, model, and training parameters.
        Args:
            args (argparse.Namespace): Parsed command-line arguments.
            config (dict): Configuration loaded from YAML file.
        """
        self.config = config
        self.epochs = args.e
        self.checkpoint = args.c
        self.device = torch.device(
            f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
        )
        self.num_workers = config["num_workers"]
        self.pin_memory = config["pin_memory"]
        self.csv_path = config["csv_path"]
        self.optim = config["optimizer"]
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.crop_size = config["crop_size"]
        self.val_num = min(4, self.batch_size)
        self.dataset_type = config.get("dataset_type", "2D")
        self.kspace = config.get("kspace", False)
        self.hybrid = config.get("hybrid", False)
        self.dataset_name = config.get("dataset_name", "FastMriDataset2D")
        self.num_classes = config["num_classes"]
        self.model_name = f"{config['lr']}_{config['comment']}"
        self.save_base_folder = Path(config["base_output"]) / f"train_{self.model_name}"
        self.cross_fold = args.single_fold
        self.loss = self._get_loss(config)
        self.model_type = config["model"]
        self._init_network()

    def _get_loss(self, config):
        if config["dataset_name"] == "FastMriDataset_Prostate":
            weight = torch.tensor([0.0576, 0.9424]).to(self.device)
            return torch.nn.CrossEntropyLoss(weight)
        else:
            return torch.nn.CrossEntropyLoss()

    def _init_network(self):
        """
        Initialize the neural network model based on dataset type (2D or MIL).
        """
        if self.dataset_type == "MIL":
            self.network = model.MILModel(
                num_classes=self.num_classes,
                device=str(self.device),
                model=self.model_type,
            )
        else:
            self.network = ClassificationModel2D(
                num_classes=self.num_classes, model=self.model_type
            )
            self.network.to(self.device)

    def __repr__(self):
        """
        String representation of key training settings for logging.
        """
        return f"batch_size={self.batch_size} lr={self.lr} {self.model_name} dataset_type={self.dataset_type}"

    def train_fn(self, use_tqdm=True):
        """
        Run one training epoch over the training set.
        Args:
            use_tqdm (bool): Whether to use tqdm progress bar.
        """
        is_tqdm = use_tqdm
        loop = tqdm(self.train_loader, miniters=100) if is_tqdm else self.train_loader
        total_loss = 0

        for batch_data in loop:
            # Unpack batch and move to device
            if self.dataset_type == "MIL":
                bags, targets, _, _ = batch_data
                targets = targets.to(device=self.device, non_blocking=True)[0]
                prediction = self.model(bags[0])
            else:
                x, y = batch_data
                x, y = x.to(self.device, non_blocking=True), y.to(
                    self.device, non_blocking=True
                )
                prediction = self.model(x)
                targets = y

            loss = self.loss(prediction, targets)
            total_loss += loss.item()

            if torch.isnan(loss):
                # Stop if loss becomes NaN
                raise ValueError("-- Loss NaN --")

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            if is_tqdm and isinstance(loop, tqdm):
                loop.set_postfix(loss=loss.item())
        # Step the learning rate scheduler if present
        if hasattr(self, "scheduler") and self.scheduler is not None:
            self.scheduler.step()
        self.total_loss = total_loss / len(self.train_loader)
        # Log peak GPU memory usage
        if torch.cuda.is_available():
            max_vram_gb = torch.cuda.max_memory_allocated(self.device) / (1024**3)
            logging.info(f"Peak GPU VRAM {max_vram_gb:.2f} GB")
            torch.cuda.reset_peak_memory_stats(self.device)

    def validation(self):
        """
        Run validation on the validation set and update metrics.
        """
        # Plot confusion matrix every 10 epochs
        confusion_matrix = self.epoch % 10 == 0
        # Initialize metrics tracker
        validation_metrics = ValidationMetrics(
            num_classes=self.num_classes,
            plot_cm=confusion_matrix,
            save_folder=str(self.save_folder),
        )
        self.model.eval()
        total_val_loss = 0
        for batch_data in self.val_loader:
            # Unpack batch and move to device
            if self.dataset_type == "MIL":
                bags, targets, _, _ = batch_data
                targets = targets.to(self.device, non_blocking=True)[0]
                with torch.no_grad():
                    prediction = self.model(bags[0])
            else:
                x, y = batch_data
                x, y = x.to(self.device, non_blocking=True), y.to(
                    self.device, non_blocking=True
                )

                with torch.no_grad():
                    prediction = self.model(x)

                targets = y

            loss = self.loss(prediction, targets)
            total_val_loss += loss.item()
            validation_metrics.update(prediction.detach(), targets.detach())

        self.metrics = validation_metrics.compute()
        self.total_val_loss = total_val_loss / len(self.val_loader)
        logging.info(f"Validation loss: {self.total_val_loss}")
        logging.info(f"Validation metrics: {self.metrics}")
        self.metrics["val_loss"] = self.total_val_loss

        # Early stopping check
        self.early_stopping(self.total_val_loss)
        if self.early_stopping.early_stop:
            logging.info("Early stopping")

        self.model.train()

    def get_fold_score(self):
        """
        Return metrics for the current fold.
        Returns:
            dict: Metrics for the fold.
        """
        return self.fold_metrics

    def __call__(self):
        """
        Main training loop. Handles cross-validation or single split, checkpointing, and logging.
        """
        # Get folds as list of (train_idx, val_idx) tuples
        # Prepare folds for cross-validation or single split
        if self.cross_fold:
            folds_obj = Folds(csv_path=self.csv_path, n_folds=5)
            folds = [fold for fold in folds_obj]
        else:
            total_csv = pd.read_csv(self.csv_path)
            splitter = GroupShuffleSplit(n_splits=1, train_size=0.9, random_state=42)
            folds = list(splitter.split(total_csv, groups=total_csv["Patient_id"]))

        total_fold_metrics = {}

        # Create output directory and copy config for reproducibility
        Path(self.save_base_folder).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.config, Path(self.save_base_folder, args.config.name))

        for idx, (train_indices, val_indices) in enumerate(folds):
            # For each fold, set up training/validation split and run training
            self.fold_idx = idx if self.cross_fold else "final"
            if self.cross_fold:
                logging.info(f"---- Fold {idx + 1} ----")

            self.save_folder = Path(self.save_base_folder) / f"fold_{self.fold_idx}"
            logging.info(f"Save folder: {str(self.save_folder)}")
            Path(self.save_folder).mkdir(parents=True, exist_ok=True)

            # Initialize early stopping
            self.early_stopping = utilities.EarlyStopping(
                patience=15,
                verbose=True,
                monitor="val-loss",
                op_type="min",
                logger=logger,
            )

            total_csv = pd.read_csv(self.csv_path)
            total_csv.iloc[train_indices].to_csv(
                Path(self.save_folder, "train.csv"), index=False
            )
            total_csv.iloc[val_indices].to_csv(
                Path(self.save_folder, "val.csv"), index=False
            )
            self._init_network()
            self.model = self.network
            # Optionally load model checkpoint
            if self.checkpoint:
                logging.info("==> load best checkpoint from previous training")
                checkpoint = torch.load(
                    f"{self.checkpoint}/best_checkpoint.pth", weights_only=True
                )
                self.model.load_state_dict(checkpoint["model"])
            logging.info(f"Device: {self.device}")
            logging.info(f"\n{utilities.count_parameters(self.model)}")
            # Set up optimizer
            if self.optim in ["AdamW", "Adam", "SGD"]:
                self.optimizer = getattr(optim, self.optim)(
                    self.model.parameters(), lr=self.lr
                )
            else:
                raise ValueError("Select valid optimizer!")

            # Prepare dataloader arguments
            loader_kwargs = dict(
                csv_path=self.csv_path,
                train_idx=train_indices,
                val_idx=val_indices,
                kspace=self.kspace,
                crop_size=self.crop_size,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                dataset_name=self.dataset_name,
            )
            if self.dataset_type == "2D":
                loader_kwargs["twod"] = True
                loader_kwargs["hybrid"] = self.hybrid
            # Get PyTorch dataloaders
            self.train_loader, self.val_loader, self.data_length = get_loaders(
                **loader_kwargs
            )

            # Set up learning rate scheduler
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=self.optimizer, gamma=0.995
            )

            # Training loop for this fold
            for self.epoch in range(self.epochs):
                logging.info(f"Now training epoch {self.epoch}!")
                self.train_fn(use_tqdm=args.tqdm)

                logging.info(f"Train-loss: {self.total_loss}")
                self.validation()

                if self.early_stopping.early_stop:
                    break
                if self.early_stopping.save:
                    torch.save(
                        {
                            "epoch": self.epoch,
                            "model": self.model.state_dict(),
                            "optimizer": self.optimizer.state_dict(),
                        },
                        self.save_folder / "best_checkpoint.pth",
                    )
                    logging.info("Save checkpoint.")
                    self.fold_metrics = self.metrics

            # Save last checkpoint for this fold
            torch.save(
                {
                    "epoch": self.epoch,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                },
                self.save_folder / "last_checkpoint.pth",
            )
            self.fold_metrics = self.get_fold_score()
            if self.cross_fold:
                logging.info(f"Fold {idx + 1} Metrics: {self.fold_metrics}")
            for key, value in self.fold_metrics.items():
                total_fold_metrics.setdefault(key, []).append(value)

        # Aggregate and log metrics across all folds
        total_fold_metrics = {
            key: np.mean(value) for key, value in total_fold_metrics.items()
        }
        logging.info(f"Total Metrics: {total_fold_metrics}")


if __name__ == "__main__":
    # Enable fast training and TF32 support
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Parse arguments and load config
    args = parse_args()
    args.config = Path(args.config)
    with open(args.config, "r") as conf:
        config = yaml.safe_load(conf)

    torch.set_num_threads(config["num_threads"])
    # Set up logging
    ikim_logger = IKIMLogger(
        level="INFO",
        log_dir="../logs",
        comment=f"train_{config['lr']}_{config['comment']}",
    )
    logger = ikim_logger.create_logger()

    try:
        set_seed(1)
        training = TrainNetwork(args=args, config=config)
        logging.info(training)
        training()
    except Exception as e:
        logging.exception(e)
