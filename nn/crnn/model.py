import random

import torch
from lightning.pytorch import LightningModule
from torch.nn import CTCLoss
from torchinfo import summary

from nn.crnn.modules import CRNN
from utils.data_preprocessing import IMG_HEIGHT, NUM_CHANNELS
from utils.metrics import compute_metrics
import json


class CTCTrainedCRNN(LightningModule):
    def __init__(
        self,
        w2i,
        i2w,
        n_fold=0,
        ytest_i2w=None,
        max_image_len=100,
        freeze=False,
        model_loaded=None,
        test_vocab=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.w2i = w2i
        self.i2w = i2w
        self.ytest_i2w = ytest_i2w if ytest_i2w is not None else i2w

        print("Test vocab: ", test_vocab)
        self.test_vocab = self.load_test_vocab(test_vocab)

        self.model = CRNN(
            output_size=len(self.w2i) + 1, freeze_cnn=freeze, model_loaded=model_loaded
        )

        self.width_reduction = self.model.cnn.width_reduction
        self.summary(max_image_len)
        self.compute_ctc_loss = CTCLoss(blank=len(self.w2i), zero_infinity=True)

        self.Y = []
        self.Y_hat = []
        self.names = []
        self.n_fold = n_fold

    def load_test_vocab(self, test_vocab):
        if test_vocab:
            with open(test_vocab, "r") as f:
                return json.load(f)
        return None

    def summary(self, max_image_len):
        summary(self.model, input_size=[1, NUM_CHANNELS, IMG_HEIGHT, max_image_len])

    def configure_optimizers(self):
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=1e-3,
            weight_decay=1e-6,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, xl, y, yl = batch
        y_hat = self.model(x)

        y_hat = y_hat.log_softmax(dim=2)
        y_hat = y_hat.permute(1, 0, 2)

        loss = self.compute_ctc_loss(y_hat, y, xl, yl)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_epoch=True)
        return loss

    def ctc_greedy_decoder(self, y_pred, i2w):
        # Check if values in i2w[y_pred[0]] are not in test_vocab, set them to 0% probability
        if self.test_vocab is not None:
            for i in range(y_pred.size(0)):  # time steps
                for j in range(y_pred.size(1) - 1):  # token probabilities per frame
                    # read from values in i2w and check if they are in test_vocab
                    # print(f"Probability: {y_pred[i, j]}  Token: {i2w[j]}")

                    if i2w[j] not in self.test_vocab.keys():
                        y_pred[i, j] = -float("inf")

        y_pred_decoded = torch.argmax(y_pred, dim=1)
        y_pred_decoded = torch.unique_consecutive(y_pred_decoded, dim=0).tolist()

        y_pred_decoded = [i2w[i] for i in y_pred_decoded if i != len(i2w)]
        return y_pred_decoded

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.model(x)[0]
        y_hat = y_hat.log_softmax(dim=1).detach().cpu()
        y_hat = self.ctc_greedy_decoder(y_hat, self.i2w)

        y = [self.ytest_i2w[i.item()] for i in y[0]]
        self.Y.append(y)
        self.Y_hat.append(y_hat)

    def test_step(self, batch, batch_idx):
        x, y, name = batch

        y_hat = self.model(x)[0]
        y_hat = y_hat.log_softmax(dim=1).detach().cpu()
        # make the decoding based on the tokens that of the test dataset
        y_hat = self.ctc_greedy_decoder(y_hat, self.i2w)

        y = [self.ytest_i2w[i.item()] for i in y[0]]
        self.Y.append(y)
        self.Y_hat.append(y_hat)
        self.names.append(name)

    def predict_step(self, batch, batch_idx):
        x, _, name = batch
        print(name)

        y_hat = self.model(x)[0]
        y_hat = y_hat.log_softmax(dim=1).detach().cpu()
        y_hat = self.ctc_greedy_decoder(y_hat, self.i2w)

        self.Y_hat.append(y_hat)
        self.names.append(name)

    def on_validation_epoch_end(
        self,
        name="val",
        print_random_samples: bool = True,
        print_all_samples: bool = True,
    ):
        metrics = compute_metrics(y_true=self.Y, y_pred=self.Y_hat)
        for k, v in metrics.items():
            self.log(f"{name}_{k}", v, prog_bar=True, logger=True, on_epoch=True)

        if print_random_samples:
            index = random.randint(0, len(self.Y) - 1)
            print(f"Ground truth - {self.Y[index]}")
            print(f"Prediction - {self.Y_hat[index]}")

        self.Y.clear()
        self.Y_hat.clear()
        return metrics

    def on_test_epoch_end(self):
        print("on test epoch end")
        return self.on_validation_epoch_end(
            name="test", print_random_samples=False, print_all_samples=True
        )

    def on_predict_epoch_end(self):
        for y_hat, name in zip(self.Y_hat, self.names):
            print(f"Name - {name[0]}")
            print(f"Prediction - {y_hat}")

            # save predictions to json
            prediction_results = []
            for name, prediction in zip(self.names, self.Y_hat):
                prediction_results.append({"name": name[0], "prediction": prediction})
            with open("predictions.json", "w") as f:
                json.dump(prediction_results, f)
            print("Predictions saved to predictions.json")
