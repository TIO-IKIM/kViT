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


parser = argparse.ArgumentParser(prog="UnifiedTraining")
parser.add_argument(
    "-e", type=int, default=None, help="Number of epochs (overrides config)"
)
parser.add_argument("--tqdm", action="store_false", help="Disable tqdm logging")
parser.add_argument("--gpu", type=int, default=0, help="GPU id")
parser.add_argument("--config", type=str, required=True, help="Config file")
parser.add_argument("-c", type=str, default=None, help="Checkpoint folder")
parser.add_argument(
    "-s",
    "--single_fold",
    action="store_false",
    help="Use single fold (default: cross-validation)",
)


def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)


class UnifiedTrainNetwork:
    """
    Unified training class for k-space and image models (2D, MIL, kspace 2D, kspace MIL).
    Behavior is controlled by the config file.
    """

    def __init__(self, args, config):
        self.config = config
        self.epochs = args.e if args.e is not None else config.get("epochs", 50)
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
        self.crop_size = config.get("crop_size", None)
        self.val_num = min(4, self.batch_size)
        self.dataset_type = config.get("dataset_type", "2D")
        self.kspace = config.get("kspace", False)
        self.hybrid = config.get("hybrid", False)
        self.dataset_name = config.get("dataset_name", None)
        self.num_classes = config["num_classes"]
        self.model_name = f"{config['lr']}_{config['comment']}"
        self.save_base_folder = Path(config["base_output"]) / f"train_{self.model_name}"
        self.cross_fold = args.single_fold
        self.loss = self._get_loss(config)
        self.model_type = config["model"]
        self.model = self._init_network(config)

    def _get_loss(self, config):
        if config["dataset_name"] == "FastMriDataset_Prostate":
            weight = torch.tensor([0.0576, 0.9424]).to(self.device)
            return torch.nn.CrossEntropyLoss(weight)
        else:
            return torch.nn.CrossEntropyLoss()

    def _init_network(self, config):
        # Dynamically import and instantiate the correct model
        if self.kspace:
            if self.dataset_type == "2D":
                from model.cTransformer import KSpace2DTransformer

                return KSpace2DTransformer(
                    k_space_dim=self.crop_size,
                    embed_dim=config["embed_dim"],
                    num_heads=config["num_heads"],
                    num_layers=config["num_layers"],
                    ff_dim=config["ff_dim"],
                    p=config["dropout"],
                    num_classes=self.num_classes,
                    hybrid=self.hybrid,
                    kspace=self.kspace,
                ).to(self.device)
            else:
                from model.cTransformer import KSpaceTransformer

                return KSpaceTransformer(
                    k_space_dim=self.crop_size,
                    embed_dim=config["embed_dim"],
                    num_heads=config["num_heads"],
                    num_layers=config["num_layers"],
                    ff_dim=config["ff_dim"],
                    num_classes=self.num_classes,
                    p=config["dropout"],
                ).to(self.device)
        else:
            if self.dataset_type == "MIL":
                from model import image_classification as model

                return model.MILModel(
                    num_classes=self.num_classes,
                    device=str(self.device),
                    model=self.model_type,
                )
            else:
                from model.image_classification import ClassificationModel2D

                return ClassificationModel2D(
                    num_classes=self.num_classes, model=self.model_type
                ).to(self.device)

    def __repr__(self):
        return f"batch_size={self.batch_size} lr={self.lr} {self.model_name} dataset_type={self.dataset_type} kspace={self.kspace}"

    def train_fn(self, use_tqdm=True):
        is_tqdm = use_tqdm
        loop = tqdm(self.train_loader, miniters=100) if is_tqdm else self.train_loader
        total_loss = 0

        for batch_data in loop:
            # Unpack batch and move to device
            if self.kspace:
                if self.dataset_type == "2D":
                    x, y = batch_data
                    x, y = x.to(self.device, non_blocking=True), y.to(
                        self.device, non_blocking=True
                    )
                    prediction = self.model(x)
                    targets = y
                else:
                    bags, targets, *_ = batch_data
                    bag = bags[0].to(self.device)
                    targets = targets.to(self.device, non_blocking=True)[0]
                    prediction = self.model(bag)
            else:
                if self.dataset_type == "MIL":
                    bags, targets, *_ = batch_data
                    targets = targets.to(self.device, non_blocking=True)[0]
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
                raise ValueError("-- Loss NaN --")

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            if is_tqdm and isinstance(loop, tqdm):
                loop.set_postfix(loss=loss.item())

        if hasattr(self, "scheduler") and self.scheduler is not None:
            self.scheduler.step()

        self.total_loss = total_loss / len(self.train_loader)

        if torch.cuda.is_available():
            max_vram_gb = torch.cuda.max_memory_allocated(self.device) / (1024**3)
            logging.info(f"Peak GPU VRAM {max_vram_gb:.2f} GB")
            torch.cuda.reset_peak_memory_stats(self.device)

    def validation(self):
        confusion_matrix = self.epoch % 10 == 0
        validation_metrics = ValidationMetrics(
            num_classes=self.num_classes,
            plot_cm=confusion_matrix,
            save_folder=str(self.save_folder),
        )

        self.model.eval()
        total_val_loss = 0

        for batch_data in self.val_loader:
            if self.kspace:
                if self.dataset_type == "2D":
                    x, y = batch_data
                    x, y = x.to(self.device, non_blocking=True), y.to(
                        self.device, non_blocking=True
                    )
                    with torch.no_grad():
                        prediction = self.model(x)
                    targets = y
                elif self.dataset_type == "MIL":
                    bags, targets, *_ = batch_data
                    targets = targets.to(self.device, non_blocking=True)[0]
                    with torch.no_grad():
                        prediction = self.model(bags[0])
            loss = self.loss(prediction, targets)
            total_val_loss += loss.item()
            validation_metrics.update(prediction.detach(), targets.detach())

        self.metrics = validation_metrics.compute()
        self.total_val_loss = total_val_loss / len(self.val_loader)

        logging.info(f"Validation loss: {self.total_val_loss}")
        logging.info(f"Validation metrics: {self.metrics}")

        self.metrics["val_loss"] = self.total_val_loss
        self.early_stopping(self.total_val_loss)

        if self.early_stopping.early_stop:
            logging.info("Early stopping")

        self.model.train()

    def get_fold_score(self):
        return self.fold_metrics

    def __call__(self):
        # Prepare folds for cross-validation or single split
        if self.cross_fold:
            folds_obj = Folds(csv_path=self.csv_path, n_folds=5)
            folds = [fold for fold in folds_obj]
        else:
            total_csv = pd.read_csv(self.csv_path)
            splitter = GroupShuffleSplit(n_splits=1, train_size=0.9, random_state=42)
            folds = list(splitter.split(total_csv, groups=total_csv["Patient_id"]))

        total_fold_metrics = {}

        Path(self.save_base_folder).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.config, Path(self.save_base_folder, args.config.name))

        for idx, (train_indices, val_indices) in enumerate(folds):
            self.fold_idx = idx if self.cross_fold else "final"
            if self.cross_fold:
                logging.info(f"---- Fold {idx + 1} ----")
            self.save_folder = Path(self.save_base_folder) / f"fold_{self.fold_idx}"
            logging.info(f"Save folder: {str(self.save_folder)}")
            Path(self.save_folder).mkdir(parents=True, exist_ok=True)

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

            self.model = self._init_network(self.config)

            if self.checkpoint:
                logging.info("==> load best checkpoint from previous training")
                checkpoint = torch.load(
                    f"{self.checkpoint}/best_checkpoint.pth", weights_only=True
                )
                self.model.load_state_dict(checkpoint["model"])

            logging.info(f"Device: {self.device}")
            logging.info(f"\n{utilities.count_parameters(self.model)}")

            if self.optim in ["AdamW", "Adam", "SGD"]:
                self.optimizer = getattr(optim, self.optim)(
                    self.model.parameters(), lr=self.lr
                )
            else:
                raise ValueError("Select valid optimizer!")

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
                loader_kwargs["dataset_name"] = self.dataset_name

            self.train_loader, self.val_loader, self.data_length = get_loaders(
                **loader_kwargs
            )
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=self.optimizer, gamma=0.995
            )

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

        total_fold_metrics = {
            key: np.mean(value) for key, value in total_fold_metrics.items()
        }
        logging.info(f"Total Metrics: {total_fold_metrics}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args = parser.parse_args()
    args.config = Path(args.config)
    with open(args.config, "r") as conf:
        config = yaml.safe_load(conf)

    torch.set_num_threads(config["num_threads"])

    ikim_logger = IKIMLogger(
        level="INFO",
        log_dir="src/logs",
        comment=f"train_{config['lr']}_{config['comment']}",
    )
    global logger
    logger = ikim_logger.create_logger()

    try:
        set_seed(1)
        training = UnifiedTrainNetwork(args=args, config=config)
        logging.info(training)
        training()
    except Exception as e:
        logging.exception(e)
