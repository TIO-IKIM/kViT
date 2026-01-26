# -*- coding: utf-8 -*-

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

from utils.dataset import get_test_loader
import torch
from utils.metrics import ValidationMetrics
from utils.IKIMLogger import IKIMLogger
from model.cTransformer import KSpace2DTransformer
from model.image_classification import ClassificationModel2D
import argparse
from pathlib import Path
from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import yaml
from utils.utilities import ifft


parser = argparse.ArgumentParser()
parser.add_argument(
    "-t",
    "--type",
    type=str,
    choices=["kspace", "image"],
    required=False,
    default=None,
    help="Type of model to test: 'kspace' or 'image'",
)
parser.add_argument(
    "-m",
    "--model_path",
    type=str,
    required=True,
    help="Path to the model to be tested.",
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


class TestModelKSpace:
    """
    TestModelKSpace is a utility class for loading a trained KSpace2DTransformer model,
    running inference on test data, and visualizing transformer attention weights.

    Args:
        checkpoint_path (str): Path to the model checkpoint file.
        save_folder (str): Directory where results and visualizations will be saved.

    Attributes:
        save_folder (Path): Path object for saving results.
        crop_size (int): Size of the crop used in the model.
        num_classes (int): Number of output classes for classification.
        kspace (bool): Whether k-space domain is used.
        hybrid (bool): Whether hybrid domain is used.
        device (torch.device): Device for computation (CPU or CUDA).
        dataset_name (str): Name of the dataset.
        dataset_type (str): Type of the dataset.
        model (KSpace2DTransformer): The loaded transformer model.
        results (dict): Computed test metrics after evaluation.

    Methods:
        test(csv_path: str, undersample_tuple: tuple[int, float]):
            Runs inference on the test dataset and computes evaluation metrics.
        get_results():
            Returns the computed test metrics.
        visualize_transformer_weights(attention_maps, patient_id, outputs, inputs=None, true_labels=None):
            Visualizes and saves the attention maps and reconstructed images for a given test sample.
    """

    def __init__(self, config: dict, checkpoint_path: str, save_folder: str):
        checkpoint = torch.load(
            checkpoint_path, map_location=torch.device("cpu"), weights_only=True
        )
        self.save_folder = Path(save_folder)
        self.crop_size = config["crop_size"]
        self.num_classes = config["num_classes"]
        self.kspace = config["kspace"]
        self.hybrid = config["hybrid"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_name = config["dataset_name"]
        self.dataset_type = config["dataset_type"]

        self.model = KSpace2DTransformer(
            k_space_dim=config["crop_size"],
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            ff_dim=config["ff_dim"],
            p=config["dropout"],
            num_classes=self.num_classes,
            kspace=self.kspace,
            hybrid=self.hybrid,
        )

        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.model.to(self.device)

    def test(self, csv_path: str, undersample_list: list[int, float]):
        """
        Evaluates the model on a test dataset and computes validation metrics.

        Args:
            csv_path (str): Path to the CSV file containing test data information.
            undersample_list (list[int, float]): List specifying the undersampling configuration.

        Loads the test data using the provided CSV path and undersampling parameters, then iterates over the test batches.
        For each batch, moves data to the appropriate device, performs inference with the model (including attention map visualization),
        updates validation metrics, and periodically visualizes transformer attention weights. Finally, computes and stores the results.
        """
        data_loader = get_test_loader(
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

        num_samples = 0

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(data_loader):
                y = y.to(device=self.device, non_blocking=True)
                x = x.to(device=self.device, non_blocking=True)
                prediction, attention_maps = self.model(x, visualize=True)
                num_samples += 1
                metrics.update(prediction.detach(), y.detach())
                if batch_idx % 100 == 0:
                    self.visualize_transformer_weights(
                        attention_maps=attention_maps,
                        patient_id=batch_idx,
                        outputs=prediction.argmax(dim=1).cpu().numpy(),
                        inputs=x,
                        true_labels=y.cpu().numpy(),
                    )
        self.results = metrics.compute()

    def get_results(self):
        logger.info(f"Test results: {self.results}")
        return {**self.results}

    def visualize_transformer_weights(
        self, attention_maps, patient_id, outputs, inputs=None, true_labels=None
    ):
        """
        Visualizes the transformer self-attention weights and optionally the reconstructed image for a given patient.
        This method processes the attention maps from a transformer model, computes the radial self-attention overlay,
        and saves a visualization as a PDF. If input data is provided, it also reconstructs and saves the corresponding image.

        Args:
            attention_maps (list of torch.Tensor): List of attention map tensors from the transformer model.
            patient_id (str or int): Identifier for the patient, used in output filenames.
            outputs (Any): Model prediction outputs for the patient, displayed in the reconstructed image title.
            inputs (torch.Tensor, optional): Input data for the patient, used to reconstruct the image. Default is None.
            true_labels (Any, optional): True label for the patient, displayed in the reconstructed image title. Default is None.

        Saves:
            - Radial self-attention map visualization as a PDF file.
            - Reconstructed image magnitude visualization as a PDF file (if inputs are provided).
        """
        attention_maps = [m.cpu().detach() for m in attention_maps]
        last_layer_attn = attention_maps[-1]
        embedder = self.model.kspace_embedding
        k_space_dim = embedder.k_space_dim
        num_rings = embedder.num_rings
        attn_map_to_view = last_layer_attn[0].mean(dim=0)
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
        attention_overlay_map = np.log(
            (attention_overlay_map - attn_min) / (attn_max - attn_min + 1e-9)
        )

        plt.figure(figsize=(10, 10))
        plt.imshow(attention_overlay_map, cmap="hot")
        plt.title(f"Radial Self-Attention Map (Layer: Last)", fontsize=20)
        plt.xlabel("Attention To Patch Index")
        plt.ylabel("Attention From Patch Index")
        plt.colorbar(label=f"Attention Weight")
        plt.savefig(
            self.save_folder
            / f"self_attention_map_layer_last_avghead_id_{patient_id}.pdf",
            format="pdf",
            dpi=300,
        )
        plt.close()

        if inputs is not None:
            x_ifft = ifft(inputs)
            reconstructed_image = x_ifft[0, 0].cpu().numpy()
            x_ifft_magnitude = np.abs(reconstructed_image)
            plt.figure(figsize=(10, 10))
            plt.imshow(x_ifft_magnitude, cmap="gray")
            plt.title(
                f"Reconstructed Image Magnitude | Prediction: {outputs} | True Label: {true_labels}",
                fontsize=20,
            )
            plt.axis("off")
            plt.savefig(
                self.save_folder / f"reconstructed_image_id_{patient_id}.pdf",
                format="pdf",
                dpi=300,
            )
            plt.close()


class TestModelImage:
    """
    A class for testing a 2D image classification model using a specified checkpoint and configuration.

    Args:
        checkpoint_path (str): Path to the model checkpoint file.
        save_folder (str): Directory where results and plots will be saved.

    Attributes:
        save_folder (Path): Path object for the save directory.
        crop_size (int): Size to crop images for testing.
        num_classes (int): Number of output classes for classification.
        kspace (bool): Whether to use k-space data.
        hybrid (bool): Whether to use hybrid data.
        dataset_name (str): Name of the dataset.
        dataset_type (str): Type of the dataset.
        device (torch.device): Device to run the model on (CPU or CUDA).
        model (ClassificationModel2D): The loaded classification model.
        results (dict): Computed test metrics after evaluation.

    Methods:
        test(csv_path: str, undersample_tuple: tuple[int, float]):
            Runs the model on the test dataset and computes evaluation metrics.
        get_results():
            Logs and returns the computed test results as a dictionary.
    """

    def __init__(self, config: dict, checkpoint_path: str, save_folder: str):
        checkpoint = torch.load(
            checkpoint_path, map_location=torch.device("cpu"), weights_only=True
        )
        self.save_folder = Path(save_folder)
        self.crop_size = config["crop_size"]
        self.num_classes = config["num_classes"]
        self.kspace = config["kspace"]
        self.hybrid = config["hybrid"]
        self.dataset_name = config["dataset_name"]
        self.dataset_type = config["dataset_type"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ClassificationModel2D(
            num_classes=self.num_classes, model=config["model"]
        )
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.model.to(self.device)

    def test(self, csv_path: str, undersample_list: list[int, float]):
        """
        Evaluates the model on a test dataset and computes validation metrics.

        Args:
            csv_path (str): Path to the CSV file containing test data information.
            undersample_list (list[int, float]): List specifying the undersampling parameters.

        Loads the test data using the provided CSV path and undersampling parameters, performs inference
        with the model, and updates validation metrics for each batch. The computed metrics are stored in
        `self.results`.
        """
        data_loader = get_test_loader(
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
        num_samples = 0

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(data_loader):
                y = y.to(device=self.device, non_blocking=True)
                x = x.to(device=self.device, non_blocking=True)
                prediction = self.model(x)
                num_samples += 1
                metrics.update(prediction.detach(), y.detach())

        self.results = metrics.compute()

    def get_results(self):
        logger.info(f"Test results: {self.results}")
        return {**self.results}


def average_results(model_results):
    """
    Computes the average and standard deviation of metrics across multiple folds.

    Args:
        model_results (dict): A dictionary where each key is a fold identifier and each value is a dictionary
            of metric names mapped to their respective values for that fold.

    Returns:
        dict: The input dictionary with two additional keys:
            - "average": A dictionary containing the average value for each metric across all folds.
            - "std": A dictionary containing the standard deviation for each metric across all folds.
    """
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
    """
    Saves the results of model evaluation to a text file.

    Args:
        results (dict): A dictionary where each key is a fold identifier and each value is a dictionary of metrics/results for that fold.
        save_path (str): The directory path where the results file ('results.txt') will be saved.

    Writes:
        A file named 'results.txt' in the specified directory, containing the results for each fold.
    """
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

    yaml_files = glob(f"{Path(checkpoint_list[0]).parents[1]}/train_*_2D.yaml")
    if not yaml_files:
        raise FileNotFoundError(
            f"No matching YAML config file found in {Path(checkpoint_list[0]).parents[1]}"
        )
    with open(yaml_files[0], "r") as conf:
        config = yaml.safe_load(conf)

    if args.type is None:
        args.type = "kspace" if config["kspace"] else "image"

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

    model_results = {}
    logger.info(f"Found {len(checkpoint_list)} folds for testing.")
    for checkpoint in checkpoint_list:
        logger.info(f"Testing model: {checkpoint}")
        fold_save_path = f"{save_path}/{Path(checkpoint).parent.name}"
        os.makedirs(fold_save_path, exist_ok=True)
        if args.type == "kspace":
            test_model = TestModelKSpace(
                config=config, checkpoint_path=checkpoint, save_folder=fold_save_path
            )
        elif args.type == "image":
            test_model = TestModelImage(
                config=config, checkpoint_path=checkpoint, save_folder=fold_save_path
            )
        else:
            raise ValueError("Unknown model type. Use --type kspace or --type image.")

        test_model.test(
            csv_path=args.csv_path,
            undersample_list=undersample_list,
        )
        model_results[Path(checkpoint).parent.name] = test_model.get_results()

    model_results = average_results(model_results)
    logger.info(f"Averaged results: {model_results['average']}")
    logger.info(f"Standard Deviations: {model_results['std']}")
    save_results(model_results, save_path)
