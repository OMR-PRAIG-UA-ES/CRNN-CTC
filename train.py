import gc

import fire
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger

from nn.crnn.model import CTCTrainedCRNN
from utils.ctc_datamodule import CTCDataModule
from utils.seed import seed_everything

seed_everything(42, benchmark=True)


def train(
    ds_name: str = "example_dataset",
    split_number: str = 0,
    epochs: int = 200,
    patience: int = 4,  # take into account that I evaluate every 5 epochs, so patience is 5*patience
    batch_size: int = 16,
    logger: bool = False,
):
    if ds_name == "":
        raise ValueError("Please provide a dataset name")

    gc.collect()
    torch.cuda.empty_cache()

    print("EXPERIMENT TRAINING")
    print(f"\tDataset: {ds_name}")
    model_type = "crnn"
    print(f"\tModel type: {model_type}")
    print(f"\tEpochs: {epochs}")
    print(f"\tPatience: {patience}")
    print(f"\tBatch size: {batch_size}")

    train_split = f"data/splits/{ds_name}/train_{split_number}.dat"
    test_split = f"data/splits/{ds_name}/test_{split_number}.dat"
    val_split = f"data/splits/{ds_name}/val_{split_number}.dat"

    print(f"\tTrain split: {train_split}")
    print(f"\tVal split: {val_split}")
    print(f"\tTest split: {test_split}")

    datamodule = CTCDataModule(
        ds_name=ds_name,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        batch_size=batch_size,
    )
    datamodule.setup(stage="fit")
    w2i, i2w = datamodule.get_w2i_and_i2w()

    model = CTCTrainedCRNN(
        w2i=w2i,
        i2w=i2w,
        max_image_len=datamodule.get_max_img_len(),
        n_fold=split_number,
    )

    datamodule.width_reduction = model.width_reduction

    callbacks = [
        ModelCheckpoint(
            dirpath=f"weigths/{model_type}",
            filename=ds_name + f"_{split_number}",
            monitor="val_ser",
            verbose=True,
            save_top_k=1,
            save_last=False,
            save_weights_only=False,
            mode="min",
            auto_insert_metric_name=False,
            every_n_epochs=5,
            save_on_train_epoch_end=False,
        ),
        EarlyStopping(
            monitor="val_ser",
            min_delta=0.1,
            patience=patience,
            verbose=True,
            mode="min",
            strict=True,
            check_finite=True,
            divergence_threshold=100.0,
            check_on_train_epoch_end=False,
        ),
    ]

    wandb_logger = None
    if logger:
        wandb_logger = (
            WandbLogger(
                project="crnn-ctc",
                group=f"{model_type}",
                name=f"Fold-{split_number}",
                log_model=True,
            ),
        )

    trainer = Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        max_epochs=epochs,
        check_val_every_n_epoch=5,
        deterministic=False,
        benchmark=False,
        precision="16-mixed",
    )
    trainer.fit(model=model, datamodule=datamodule)

    model = CTCTrainedCRNN.load_from_checkpoint(callbacks[0].best_model_path)

    model.freeze()
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    fire.Fire(train)
