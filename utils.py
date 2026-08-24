import cv2
import pandas as pd
import numpy as np
import os
import torch


def fix_csv(csv_path, subset_path):
    df = pd.read_csv(csv_path)
    # because I have data separated into subsets, I need to load them and combine them into one dataframe
    df["filepath"] = df["filepath"].apply(lambda x: os.path.join(subset_path, x))

    if "timestamp" in df.columns:
        df = df.drop(
            columns=["timestamp_x", "timestamp_y"]
        )  # drop redundant timestamps
    else:
        df["timestamp"] = df["timestamp_x"]
        df = df.drop(columns=["timestamp_y"])

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def sonar_preprocess(img):

    # idea for improvement: crop left triangle of image, since it contains no useful information
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

    img = img.astype(np.float32) / 255.0

    img = torch.from_numpy(img).unsqueeze(0)

    return img


# def transform_sonar(img):
# if random.random() < 0.5:
# img = functional.hflip(img)
#
# return img
