import json
import os

import torch
from torch.utils.data import Dataset
from utils.data_preprocessing import preprocess_image
from utils.encoding_convertions import gtParser
import cv2


class CTCDataset(Dataset):
    def __init__(
        self,
        ds_name: str,
        split: str,
        split_files,
        width_reduction: int = 2,
        sample_files=None,
    ):
        super().__init__()
        self.ds_name = ds_name
        self.split = split
        self.split_files = split_files
        self.width_reduction = width_reduction
        self.sample_files = sample_files
        self.init(vocab_name="ctc_w2i")

    def init(self, vocab_name: str = "w2i"):
        self.gt_parser = gtParser(single_line_data=True, no_sep_tok=True)

        # Check dataset
        # assert self.ds_name in DATASETS, f"Dataset {self.ds_name} not supported."

        # Check split
        assert self.split in [
            "train",
            "val",
            "test",
            "predict",
        ], f"Split {self.split} not supported."

        # Get data
        self.X, self.Y = self.get_images_and_transcripts_files()

        # Get vocab
        vocab_folder = os.path.join("data", "vocabs")
        os.makedirs(vocab_folder, exist_ok=True)
        vocab_name = self.ds_name + f"_{vocab_name}.json"
        self.w2i_path = os.path.join(vocab_folder, vocab_name)
        self.w2i, self.i2w = self.check_and_retrieve_vocabulary()

        self.set_max_lens()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = preprocess_image(path=self.X[idx], split=self.split)

        if self.split == "predict":
            return x, x, self.X[idx]

        y = self.preprocess_transcript(path=self.Y[idx])
        if self.split == "train":
            return (x, (x.shape[2] // self.width_reduction), y, len(y))
        if self.split == "test":
            return x, y, self.X[idx]
        return x, y

    def preprocess_transcript(self, path: str):
        y = self.gt_parser.convert(src_file=path)
        y = [self.w2i[w] for w in y]
        return torch.tensor(y, dtype=torch.int32)

    def get_images_and_transcripts_files(self):
        # partition_file = f"data/splits/{self.ds_name}/{self.split}.txt"
        if self.split == "train":
            partition_file = self.split_files[0]
        elif self.split == "val":
            partition_file = self.split_files[1]
        elif self.split == "test":
            partition_file = self.split_files[2]

        elif self.split == "predict":
            for i in self.sample_files:
                image = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
                h, w = image.shape
                width_reduction = 2**1  # number of poolings in second dimension
                height_reduction = 2**4  # number of poolings in first dimension
                h_final = h // height_reduction
                w_final = w // width_reduction

                if h_final < 1 or w_final < 1:
                    print(
                        f"Image {i} is too small for the model poolings. Skipping it."
                    )
                    # pop las element from transcripts
                    self.sample_files.remove(i)
                    continue
            return self.sample_files, self.sample_files

        images = []
        transcripts = []
        with open(partition_file, "r") as f:
            for line in f.read().splitlines():
                line = line.strip()
                # data/gt/example_dataset/rrTvg_11
                transcripts.append(f"{line}.txt")

                line = line.replace("/gt", "/jpg")

                # check if an image size is too small for the model poolings
                if not os.path.isfile(f"{line}.jpg"):
                    print(f"Image {line} does not exist. Skipping it.")
                    # pop las element from transcripts
                    transcripts.pop()
                    continue
                # print(f"Checking image {line}.jpg")
                image = cv2.imread(f"{line}.jpg", cv2.IMREAD_GRAYSCALE)
                h, w = image.shape
                width_reduction = 2**1  # number of poolings in second dimension
                height_reduction = 2**4  # number of poolings in first dimension
                h_final = h // height_reduction
                w_final = w // width_reduction

                if h_final < 1 or w_final < 1:
                    print(
                        f"Image {line} is too small for the model poolings. Skipping it."
                    )
                    # pop las element from transcripts
                    transcripts.pop()
                    continue

                images.append(f"{line}.jpg")

        return images, transcripts

    def check_and_retrieve_vocabulary(self):
        w2i = {}
        i2w = {}

        if os.path.isfile(self.w2i_path):
            with open(self.w2i_path, "r") as f:
                w2i = json.load(f)
            i2w = {v: k for k, v in w2i.items()}
        else:
            w2i, i2w = self.make_vocabulary()
            with open(self.w2i_path, "w") as f:
                json.dump(w2i, f)

        return w2i, i2w

    def make_vocabulary(self):
        vocab = []
        for split in self.split_files:
            # partition_file = f"data/splits/{self.ds_name}/{split}.txt"
            partition_file = split
            with open(partition_file, "r") as f:
                for line in f.read().splitlines():
                    line = line.strip()
                    transcript = self.gt_parser.convert(src_file=f"{line}.txt")
                    vocab.extend(transcript)
        vocab = sorted(set(vocab))

        w2i = {}
        i2w = {}
        for i, w in enumerate(vocab):
            w2i[w] = i + 1
            i2w[i + 1] = w
        w2i["<blank>"] = 0
        i2w[0] = "<blank>"

        return w2i, i2w

    def set_max_lens(self):
        # set the max length for the collection
        # 1) get the max gt length
        # 2) get the max img length

        max_seq_len = 0
        max_img_len = 0

        # Recursively list files in the directory and its subdirectories
        # for root, dirs, files in os.walk(f"data/gt/{self.ds_name}"):
        # TODO
        for root, dirs, files in os.walk("data/gt/"):
            for t in files:
                if t.endswith(".txt") and not t.startswith("."):
                    path_gt_file = os.path.join(root, t)

                    transcript = self.gt_parser.convert(src_file=path_gt_file)
                    max_seq_len = max(max_seq_len, len(transcript))

                    image_path = path_gt_file.replace(".txt", ".jpg").replace(
                        "gt", "jpg"
                    )
                    image = preprocess_image(path=image_path, split=self.split)
                    max_img_len = max(max_img_len, image.shape[2])

        self.max_seq_len = max_seq_len
        self.max_img_len = max_img_len
