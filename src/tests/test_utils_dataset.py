# -*- coding: utf-8 -*-

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import tempfile
import unittest
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import torch

from utils import dataset


class TestDatasetUtils(unittest.TestCase):
    @staticmethod
    def _make_fastmri_sample(tmp_path, crop_size=16):
        data_dir = tmp_path / "data"
        data_dir.mkdir(exist_ok=True)
        tensor = torch.ones((8, crop_size, crop_size), dtype=torch.complex64)
        file_path = data_dir / "sample.pt"
        torch.save({"kspace": tensor}, file_path)
        return file_path

    @staticmethod
    def _make_csv(tmp_path, filenames, labels, patient_ids=None):
        data = {
            "filename": [str(path) for path in filenames],
            "label": labels,
        }
        if patient_ids is not None:
            data["Patient_id"] = patient_ids
        df = pd.DataFrame(data)
        csv_path = tmp_path / "dataset.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_collate_bags_returns_expected_structure(self):
        batch = [
            (torch.ones(1), torch.tensor(1), "pid", ["seq"]),
            (torch.zeros(1), torch.tensor(2), "pid2", ["seq2"]),
        ]
        bags, labels, pids, seqs = dataset.collate_bags(batch)
        self.assertEqual(len(bags), 2)
        self.assertEqual(labels.shape, (2,))
        self.assertEqual(pids, ["pid", "pid2"])
        self.assertEqual(seqs, [["seq"], ["seq2"]])

    def test_collate_bags_handles_slice_labels(self):
        batch = [
            (torch.ones(1), torch.tensor(1), "pid", ["seq"], 0, "m1"),
        ]
        bags, labels, pids, seqs, slice_labels, modalities = dataset.collate_bags(batch)
        self.assertEqual(slice_labels.shape, (1,))
        self.assertEqual(modalities, ["m1"])

    def test_get_loaders_fastmri(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            file_path = self._make_fastmri_sample(tmp_path)
            csv_path = self._make_csv(tmp_path, [file_path, file_path], [0, 1])

            train_loader, val_loader, total = dataset.get_loaders(
                csv_path=str(csv_path),
                train_idx=[0],
                val_idx=[1],
                kspace=True,
                crop_size=16,
                batch_size=1,
                num_workers=1,
                pin_memory=False,
                log=None,
                twod=True,
                hybrid=False,
                dataset_name="FastMriDataset2D_Knee",
            )

            self.assertEqual(total, 2)
            self.assertEqual(len(train_loader.dataset), 1)
            self.assertEqual(len(val_loader.dataset), 1)
            batch = next(iter(val_loader))
            self.assertIsInstance(batch[0], torch.Tensor)

    def test_get_loaders_invalid_name_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = self._make_csv(Path(tmpdir), [], [])
            with self.assertRaises(ValueError):
                dataset.get_loaders(
                    csv_path=str(csv_path),
                    train_idx=[],
                    val_idx=[],
                    kspace=True,
                    dataset_name="Unknown",
                )

    def test_get_test_loader_returns_loader(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            file_path = self._make_fastmri_sample(tmp_path)
            csv_path = self._make_csv(tmp_path, [file_path], [0])

            loader = dataset.get_test_loader(
                csv_path=str(csv_path),
                batch_size=1,
                num_workers=1,
                pin_memory=False,
                crop_size=16,
                kspace=True,
                twod=True,
                dataset_name="FastMriDataset2D_Knee",
            )

            batch = next(iter(loader))
            self.assertIsInstance(batch[0], torch.Tensor)

    def test_folds_creates_folds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            filenames = [tmp_path / f"sample{i}.pt" for i in range(4)]
            for path in filenames:
                torch.save(torch.ones((1, 4, 4)), path)
            csv_path = self._make_csv(
                tmp_path, filenames, [0, 1, 0, 1], patient_ids=["a", "b", "a", "b"]
            )
            folds = dataset.Folds(str(csv_path), n_folds=2)
            self.assertEqual(len(folds), 2)
            train_idx, test_idx = folds[0]
            self.assertTrue(len(train_idx) > 0)
            self.assertTrue(len(test_idx) > 0)


if __name__ == "__main__":
    unittest.main()
