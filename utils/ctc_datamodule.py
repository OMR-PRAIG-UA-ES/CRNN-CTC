from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from utils.ctc_dataset import CTCDataset
from utils.data_preprocessing import ctc_batch_preparation


class CTCDataModule(LightningDataModule):
    def __init__(
        self,
        ds_name: str,
        train_split: str,
        val_split: str,
        test_split: str,
        batch_size: int = 16,
        num_workers: int = 20,
        width_reduction: int = 2,
        sample_files=None,
    ):
        super().__init__()
        self.ds_name = ds_name
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.width_reduction = width_reduction
        self.sample_files = sample_files

    def setup(self, stage: str):
        if stage == "fit":
            self.train_ds = CTCDataset(
                ds_name=self.ds_name,
                split_files=[self.train_split, self.val_split, self.test_split],
                split="train",
                width_reduction=self.width_reduction,
            )
            self.val_ds = CTCDataset(
                ds_name=self.ds_name,
                split_files=[self.train_split, self.val_split, self.test_split],
                split="val",
                width_reduction=self.width_reduction,
            )

        if stage == "test":
            self.test_ds = CTCDataset(
                ds_name=self.ds_name,
                split_files=[self.train_split, self.val_split, self.test_split],
                split="test",
                width_reduction=self.width_reduction,
            )

        if stage == "predict":
            self.predict_ds = CTCDataset(
                ds_name=self.ds_name,
                split_files=[self.train_split, self.val_split, self.test_split],
                split="predict",
                width_reduction=self.width_reduction,
                sample_files=self.sample_files,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=ctc_batch_preparation,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_ds,
            batch_size=1,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def get_w2i_and_i2w(self):
        try:
            return self.train_ds.w2i, self.train_ds.i2w
        except AttributeError:
            return self.test_ds.w2i, self.test_ds.i2w

    def get_max_seq_len(self):
        try:
            return self.train_ds.max_seq_len
        except AttributeError:
            return self.test_ds.max_seq_len

    def get_max_img_len(self):
        try:
            return self.train_ds.max_img_len
        except AttributeError:
            return self.test_ds.max_img_len
