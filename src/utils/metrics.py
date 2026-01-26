# -*- coding: utf-8 -*-

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
 
import torch
import torchmetrics
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class ValidationMetrics:
    """
    A class to handle stateful metric calculation over an epoch using torchmetrics.

    This class correctly accumulates predictions and targets from all batches in an epoch
    before computing the final metrics, which is the proper way to calculate metrics like
    AUROC, AUPRC, F1-score, etc.
    """

    def __init__(
        self, num_classes: int, save_folder: str = None, plot_cm: bool = False
    ):
        """
        Initializes all required metrics from torchmetrics.

        Args:
            num_classes (int): The number of classes in the classification task.
            save_folder (str, optional): Folder to save the confusion matrix plot. Defaults to None.
            plot_cm (bool, optional): Whether to plot the confusion matrix. Defaults to False.
        """
        self.num_classes = num_classes
        self.save_folder = save_folder
        self.plot_cm = plot_cm

        # Use a MetricCollection for simplicity and efficiency
        self.metrics = torchmetrics.MetricCollection(
            {
                "AUROC": torchmetrics.AUROC(
                    task="multiclass", average="macro", num_classes=num_classes
                ),
                "AUPRC": torchmetrics.AveragePrecision(
                    task="multiclass", average="macro", num_classes=num_classes
                ),
                "F1": torchmetrics.F1Score(task="multiclass", num_classes=num_classes),
                "Accuracy": torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes
                ),
                "Recall": torchmetrics.Recall(
                    task="multiclass", num_classes=num_classes
                ),
                "Precision": torchmetrics.Precision(
                    task="multiclass", num_classes=num_classes
                ),
            }
        )

        if self.plot_cm:
            self.confusion_matrix = torchmetrics.ConfusionMatrix(
                task="multiclass", num_classes=num_classes
            )

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update the state of the metrics with a new batch of predictions and targets.

        Args:
            predictions (torch.Tensor): The model's output logits or probabilities.
            targets (torch.Tensor): The ground truth labels.
        """
        if predictions.ndim == 1:
            predictions = predictions.unsqueeze(0)
        if targets.ndim == 0:
            targets = targets.unsqueeze(0)

        # torchmetrics expects probabilities for AUROC/AUPRC, and logits or probs for others
        predictions_softmax = torch.softmax(predictions, dim=-1)

        # Move metrics to the same device as the data
        self.metrics.to(predictions.device)
        self.metrics.update(predictions_softmax, targets)

        if self.plot_cm:
            self.confusion_matrix.to(predictions.device)
            # For CM, we need class indices
            class_predictions = torch.argmax(predictions_softmax, dim=-1)
            self.confusion_matrix.update(class_predictions, targets)

    def compute(self, epoch: int = 0) -> dict:
        """
        Compute the final metrics over all accumulated batches for the epoch.

        Args:
            epoch (int): The current epoch number, used for saving the plot.

        Returns:
            dict: A dictionary of computed metric scores.
        """
        # Compute all metrics in the collection
        computed_metrics = self.metrics.compute()

        if self.plot_cm and self.save_folder:
            self.plot_confusion_matrix(epoch)

        # Reset metrics for the next epoch
        self.metrics.reset()
        if self.plot_cm:
            self.confusion_matrix.reset()

        # Return a dictionary of scalar values
        return {key: value.item() for key, value in computed_metrics.items()}

    def plot_confusion_matrix(self, epoch: int):
        """
        Computes, plots, and saves the confusion matrix.
        """
        cm = self.confusion_matrix.compute().cpu().numpy()

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=range(self.num_classes),
            yticklabels=range(self.num_classes),
        )

        plt.title(f"Confusion Matrix at Epoch {epoch}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()

        save_path = f"{self.save_folder}/confusion_matrix_epoch_{epoch}.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Confusion matrix saved to {save_path}")
