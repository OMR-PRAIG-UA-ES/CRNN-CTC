import gc
import os

import fire
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers.wandb import WandbLogger

from utils.utils import read_samples
from nn.crnn.model import CTCTrainedCRNN
from utils.ctc_datamodule import CTCDataModule
from utils.seed import seed_everything

seed_everything(42, benchmark=False)


def predict(
    ds_name="",
    samples="example/samples.dat",
    n_fold=0,
    model_type: str = "crnn",
    checkpoint_path: str = "",
    vocab: str = "",
):
    if ds_name == "":
        raise ValueError("Please provide a dataset name")

    gc.collect()
    torch.cuda.empty_cache()

    if os.path.exists(samples):
        samples_files = read_samples(samples)
    else:
        print(f"Samples file {samples} does not exist.")
        return

    # Check if checkpoint path is empty or does not exist
    if checkpoint_path == "":
        print("-checkpoint_path path not provided")
        return
    if not os.path.exists(checkpoint_path):
        print(f"-checkpoint_path {checkpoint_path} does not exist")
        return

    if vocab == "":
        print("-vocab path not provided")
        return

    # Get source dataset name
    src_ds_name = os.path.basename(checkpoint_path).split(".")[0]

    # Experiment info
    print("PREDICT SAMPLES")
    print(f"\tSource dataset: {src_ds_name}")
    # print(f"\tTest dataset: {ds_name}")

    print(f"\tModel type: {model_type}")
    print(f"\tCheckpoint path: {checkpoint_path}")
    print(f"\tVocab: {vocab}")

    # Data module
    datamodule = CTCDataModule(
        ds_name=ds_name,
        train_split="",
        val_split="",
        test_split="",
        sample_files=samples_files,
    )
    datamodule.setup(stage="predict")
    ytest_i2w = datamodule.predict_ds.i2w

    # Model
    model = CTCTrainedCRNN.load_from_checkpoint(
        checkpoint_path, ytest_i2w=ytest_i2w, n_fold=n_fold, test_vocab=vocab
    )

    # Test
    trainer = Trainer(
        precision="16-mixed",  # Mixed precision training
        accelerator="auto",
        devices="auto",
    )
    model.freeze()
    try:
        trainer.predict(model, datamodule=datamodule)
    except Exception as e:
        print(f"Error: {e}")
        print("Retrying...")


if __name__ == "__main__":
    fire.Fire(predict)
