import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from sonar_img_cnn_lstm_prj.utils import fix_csv, sonar_preprocess


class SonarDataset(Dataset):
    def __init__(self, root_dir, seq_len=5, augment=False):
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.augment = augment

        self.samples = []
        self.cache = {}
        self._load_all_subsets()

    def _load_all_subsets(self):
        all_sequences = []

        for subset in os.listdir(self.root_dir):
            subset_path = os.path.join(self.root_dir, subset)
            csv_path = os.path.join(subset_path, "topics_combined.csv")

            if not os.path.exists(csv_path):
                continue

            df = fix_csv(csv_path, subset_path)

            future_offset = 1

            # iterate through dataframe to create sequences of images and corresponding targets of future SOG
            for i in range(0, len(df) - self.seq_len - future_offset, self.seq_len):

                future_idx = i + self.seq_len
                future_row = df.iloc[future_idx]

                sequence = df.iloc[i:future_idx]  # get current sequence of images

                image_paths = sequence["filepath"].values

                target = future_row["sog"]

                target_timestamp = future_row["timestamp"]

                all_sequences.append((image_paths, target, subset, target_timestamp))

        print("\n-------- DATASET SUMMARY --------")
        print(f"Total sequences created : {len(all_sequences)}")
        print(f"Sequence length : {self.seq_len}")
        print(f"Root directory : {self.root_dir}")
        print("-----------------------\n")

        self.samples = all_sequences

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(f"Image not found at path: {path}")

        img = sonar_preprocess(img)
        return img

    def __getitem__(self, idx):
        img_paths, target, subset, target_timestamp = self.samples[idx]

        images = []

        for path in img_paths:
            img = self._load_image(path)

            if self.augment:
                noise = torch.randn_like(img) * 0.01
                img = torch.clamp(img + noise, 0.0, 1.0)

            images.append(img)

        images = torch.stack(images)
        target = torch.tensor(target, dtype=torch.float32)

        return images, target, subset, target_timestamp
