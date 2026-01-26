# -*- coding: utf-8 -*-

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

from utils.dataset import get_test_loader
import torch
from utils.metrics import ValidationMetrics
from utils.IKIMLogger import IKIMLogger
from model import image_classification as image_model
from model.cTransformer import KSpaceTransformer
import argparse
from pathlib import Path
from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import yaml


parser = argparse.ArgumentParser()
parser.add_argument(
    "-t",
    "--type",
    type=str,
    choices=["kspace", "image"],
    required=False,
    default=None,
    help="Type of model to test: 'kspace' or 'image'. If not provided, inferred from config.",
)
parser.add_argument(
    "-m",
    "--model_path",
    type=str,
    required=True,
    help="Path to the model checkpoint folder to be tested.",
)
parser.add_argument(
    "-c",
    "--csv_path",
    type=str,
    required=True,
    help="Path to the CSV file containing test data.",
)
parser.add_argument(
    "-o",
    "--output_path",
    type=str,
    default="./output",
    help="Path to save the test results.",
)
parser.add_argument(
    "-a",
    "--all",
    action="store_false",
    help="Run tests on all folds and average results.",
)
parser.add_argument(
    "-u",
    "--undersample",
    type=int,
    required=False,
    default=0,
    help="Undersampling rate for k-space data. Allowed rates are 0 (no undersampling), 2, 4, 6, 8, 10, 16, 24.",
)


class TestModelImageMIL:
    """Test harness for the image-domain MIL classifier."""

    def __init__(self, config: dict, checkpoint_path: str, save_folder: str):
        checkpoint = torch.load(
            checkpoint_path, map_location=torch.device("cpu"), weights_only=True
        )
        self.save_folder = Path(save_folder)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = config["num_classes"]
        self.crop_size = config["crop_size"]
        self.kspace = config["kspace"]
        self.hybrid = config["hybrid"]
        self.dataset_name = config["dataset_name"]
        self.dataset_type = config["dataset_type"]
        self.model_name = config["model"]

        self.model = image_model.MILModel(
            num_classes=self.num_classes, device=self.device, model=self.model_name
        )
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.model.to(self.device)

    def test(self, csv_path: str, undersample_list: list[int, float]):
        test_loader = get_test_loader(
            csv_path=csv_path,
            dataset_name=self.dataset_name,
            crop_size=self.crop_size,
            kspace=self.kspace,
            hybrid=self.hybrid,
            undersample_list=undersample_list,
            dataset_type=self.dataset_type,
        )

        metrics = ValidationMetrics(
            num_classes=self.num_classes, plot_cm=True, save_folder=self.save_folder
        )

        with torch.no_grad():
            for bags, targets, _, _ in test_loader:
                labels = targets[0].to(self.device)
                bag = bags[0]
                outputs = self.model(bag)
                metrics.update(outputs.detach(), labels.detach())
        self.results = metrics.compute()

    def get_results(self):
        logger.info(f"Test results: {self.results}")
        return self.results


class TestModelKspaceMIL:
    """Test harness for the k-space MIL transformer."""

    def __init__(self, config: dict, checkpoint_path: str, save_folder: str):
        checkpoint = torch.load(
            checkpoint_path, map_location=torch.device("cpu"), weights_only=True
        )
        self.save_folder = Path(save_folder)
        self.crop_size = config["crop_size"]
        self.kspace = config["kspace"]
        self.hybrid = config["hybrid"]
        self.num_classes = config["num_classes"]
        self.dataset_name = config["dataset_name"]
        self.dataset_type = config["dataset_type"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = KSpaceTransformer(
            k_space_dim=config["crop_size"],
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            p=config["dropout"],
            ff_dim=config["ff_dim"],
            num_classes=self.num_classes,
        )
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.model.to(self.device)

    def test(self, csv_path: str, undersample_list: list[int, float]):
        test_loader = get_test_loader(
            csv_path=csv_path,
            batch_size=1,
            crop_size=self.crop_size,
            kspace=True,
            undersample_list=undersample_list,
            dataset_type=self.dataset_type,
            dataset_name=self.dataset_name,
            hybrid=self.hybrid,
        )

        metrics = ValidationMetrics(
            num_classes=self.num_classes, plot_cm=True, save_folder=self.save_folder
        )

        with torch.no_grad():
            for bags, targets, patient_id, seq_list in test_loader:
                labels = targets[0].to(self.device)
                bag = bags[0].to(self.device)
                seq_list = seq_list[0]
                outputs, attn_maps, attn_weights = self.model(bag, visualize=True)
                self.visualize_attention_overlay(
                    attention_weights=attn_weights,
                    attention_maps=attn_maps,
                    patient_id=patient_id,
                    outputs=outputs.argmax().item(),
                    seq_list=seq_list,
                )
                metrics.update(outputs.detach(), labels.detach())
        self.results = metrics.compute()

    def get_results(self):
        logger.info(f"Test results: {self.results}")
        return self.results

    def visualize_attention_overlay(
        self, attention_weights, attention_maps, patient_id, outputs, seq_list
    ):
        attention_weights = attention_weights.cpu().detach()
        attention_maps = [m.cpu().detach() for m in attention_maps]
        embedder = self.model.embedding
        k_space_dim = embedder.k_space_dim
        num_rings = embedder.num_rings
        mil_weights = attention_weights.numpy().flatten()
        unique_seqs = list(set(seq_list))
        color_map = plt.get_cmap("tab10")
        seq_to_color = {seq: color_map(i % 10) for i, seq in enumerate(unique_seqs)}
        colors = [seq_to_color[seq] for seq in seq_list]
        plt.figure(figsize=(10, 8))
        for idx, (weight, color) in enumerate(zip(mil_weights, colors)):
            plt.bar(idx, weight, color=color)
        plt.xlabel("Slice Index")
        plt.ylabel("MIL Attention Weight (Importance)")
        plt.title(f"Slice Importance for the Final Bag Prediction: {outputs}")
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=seq_to_color[seq]) for seq in unique_seqs
        ]
        plt.legend(handles, unique_seqs, title="Sequence", loc="lower right")
        plt.savefig(self.save_folder / f"mil_attention_weights_id_{patient_id}.png")
        plt.close()
        last_layer_attn = attention_maps[-1]
        important_slice_idx = np.argmax(mil_weights)
        attn_map_to_view = last_layer_attn[important_slice_idx].mean(dim=0)
        ring_importance_scores = attn_map_to_view[:, 1:].sum(axis=0)
        deltas = torch.softmax(embedder.radius_deltas.cpu().detach(), dim=0)
        radius_max = embedder.radius_max.cpu().detach()
        boundaries = torch.cat(
            [torch.tensor([0.0]), torch.cumsum(deltas * (radius_max - 1e-6), dim=0)]
        )
        radius_grid = embedder.radius_grid.view(k_space_dim, k_space_dim).cpu().detach()
        segmentation_map_2d = torch.zeros_like(radius_grid, dtype=torch.long)
        for i in range(num_rings):
            low = boundaries[i]
            high = boundaries[i + 1] if i < num_rings - 1 else radius_max
            mask = (radius_grid >= low) & (radius_grid < high)
            segmentation_map_2d[mask] = i
        segmentation_map_2d = segmentation_map_2d.numpy()
        ring_importance_scores = ring_importance_scores.numpy()
        attention_overlay_map = ring_importance_scores[segmentation_map_2d]
        attn_min = attention_overlay_map.min()
        attn_max = attention_overlay_map.max()
        attention_overlay_map = (attention_overlay_map - attn_min) / (
            attn_max - attn_min + 1e-9
        )
        plt.figure(figsize=(10, 10))
        plt.imshow(attention_overlay_map, cmap="hot")
        plt.title(
            f"Radial Self-Attention Map (Layer: Last, Slice: {important_slice_idx}) | Prediction: {outputs}"
        )
        plt.xlabel("Attention To Patch Index")
        plt.ylabel("Attention From Patch Index")
        plt.colorbar(label="Normalized Attention Weight")
        plt.savefig(
            self.save_folder
            / f"self_attention_map_layer_last_avghead_slice{important_slice_idx}_id_{patient_id}.png"
        )
        plt.close()


def average_results(model_results):
    average_metrics = {}
    std_metrics = {}
    for fold_results in model_results.values():
        for key, value in fold_results.items():
            average_metrics.setdefault(key, 0)
            average_metrics[key] += value
    for key in average_metrics:
        average_metrics[key] /= len(model_results)
        std_metrics[key] = np.std(
            [fold_results[key] for fold_results in model_results.values()]
        )
    model_results["average"] = average_metrics
    model_results["std"] = std_metrics
    return model_results


def save_results(results, save_path):
    with open(f"{save_path}/results.txt", "w") as f:
        for fold in results:
            f.write(f"{fold}:\n")
            for key, value in results[fold].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")


if __name__ == "__main__":
    args = parser.parse_args()
    save_path = f"{args.output_path}/{Path(args.model_path).name}"
    ikim_logger = IKIMLogger(
        level="INFO", log_dir=save_path, comment=f"Test_{args.undersample}"
    )
    logger = ikim_logger.create_logger()

    if args.all:
        checkpoint_list = glob(f"{args.model_path}/*/best_checkpoint.pth")
    else:
        checkpoint_list = glob(f"{args.model_path}/best_checkpoint.pth")

    yaml_files = glob(f"{Path(checkpoint_list[0]).parents[1]}/train_*_MIL.yaml")
    if not yaml_files:
        raise FileNotFoundError(
            f"No matching YAML config file found in {Path(checkpoint_list[0]).parents[1]}"
        )
    with open(yaml_files[0], "r") as conf:
        config = yaml.safe_load(conf)

    center_fraction_dict = {
        0: 1.0,
        2: 0.04,
        4: 0.08,
        6: 0.05,
        8: 0.04,
        10: 0.03,
        12: 0.02,
        16: 0.015,
        24: 0.008,
    }

    undersample_factor = args.undersample
    center_fraction = center_fraction_dict.get(undersample_factor, 1.0)
    undersample_list = [center_fraction, undersample_factor]

    if args.type is None:
        args.type = "kspace" if config["kspace"] else "image"

    tester_cls = TestModelKspaceMIL if args.type == "kspace" else TestModelImageMIL
    model_results = {}
    logger.info(f"Found {len(checkpoint_list)} folds for testing.")
    for checkpoint in checkpoint_list:
        logger.info(f"Testing model: {checkpoint}")
        fold_save_path = f"{save_path}/{Path(checkpoint).parent.name}"
        os.makedirs(fold_save_path, exist_ok=True)
        test_model = tester_cls(
            config=config, checkpoint_path=checkpoint, save_folder=fold_save_path
        )
        test_model.test(csv_path=args.csv_path, undersample_list=undersample_list)
        model_results[Path(checkpoint).parent.name] = test_model.get_results()

    model_results = average_results(model_results)
    logger.info(f"Averaged results: {model_results['average']}")
    logger.info(f"Standard Deviations: {model_results['std']}")
    save_results(model_results, save_path)
