import torch
import torch.nn as nn

from utils.data_preprocessing import IMG_HEIGHT, NUM_CHANNELS


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # in channels, out channels/filters, kernel size, stride, padding
        self.conv1 = nn.Conv2d(NUM_CHANNELS, 64, 5, padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.pool1 = nn.MaxPool2d((2, 2), stride=(2, 2))

        self.conv2 = nn.Conv2d(64, 64, 5, padding="same", bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.pool2 = nn.MaxPool2d((2, 1), stride=(2, 1))

        self.conv3 = nn.Conv2d(64, 128, 3, padding="same", bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.lrelu3 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.pool3 = nn.MaxPool2d((2, 1), stride=(2, 1))

        self.conv4 = nn.Conv2d(128, 128, 3, padding="same", bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.lrelu4 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.pool4 = nn.MaxPool2d((2, 1), stride=(2, 1))

        # calculate the width reduction of the image taking into account all the pooling
        self.width_reduction = 2**1  # number of poolings in second dimension
        self.height_reduction = 2**4  # number of poolings in first dimension
        self.out_channels = 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lrelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lrelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.lrelu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.lrelu4(x)
        x = self.pool4(x)
        return x


class RNN(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()

        self.blstm = nn.LSTM(
            input_size,
            256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.5,
        )
        # (batch_size, seq_len, 2 * 256)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(256 * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, (hidden_state, cell_state) = self.blstm(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x


class CRNN(nn.Module):
    def __init__(self, output_size: int, freeze_cnn: bool = False, model_loaded=None):
        super().__init__()
        # CNN
        if freeze_cnn:
            self.cnn = model_loaded.model.cnn
            for param in self.cnn.parameters():
                param.requires_grad = False
            print("CNN freezed")
        else:
            self.cnn = CNN()
        # RNN
        # self.num_frame_repeats = self.cnn.width_reduction * frame_multiplier_factor
        self.rnn_input_size = self.cnn.out_channels * (
            IMG_HEIGHT // self.cnn.height_reduction
        )
        self.rnn = RNN(input_size=self.rnn_input_size, output_size=output_size)

    def forward(self, x):
        # CNN
        # x: [b, NUM_CHANNELS, IMG_HEIGHT, w]

        x = self.cnn(x.float())
        # x: [b, self.cnn.out_channels, nh = IMG_HEIGHT // self.height_reduction, nw = w // self.width_reduction]
        # Prepare for RNN
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        # x: [b, w, c, h]

        x = x.reshape(b, w, self.rnn_input_size)
        # x: [b, w, self.rnn_input_size]

        # RNN
        x = self.rnn(x)
        return x
