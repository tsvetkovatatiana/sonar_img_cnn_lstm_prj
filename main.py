import time
import torch
from torch.utils.data import DataLoader

from sonar_img_cnn_lstm_prj.model import CNN_LSTM
from sonar_img_cnn_lstm_prj.trainer import Trainer
from sonar_img_cnn_lstm_prj.dataset import SonarDataset
from sonar_img_cnn_lstm_prj.split_subsets import train_set, val_set, test_set


# batch(images, target, subset, target_timestamp)
# custom collate fn to handle sequences and return subset and timestamp info for each sample
def sequence_collate_fn(batch):
    images = torch.stack([images[0] for images in batch])
    targets = torch.stack([target[1] for target in batch])

    subsets = [subset[2] for subset in batch]
    timestamps = [timestamp[3] for timestamp in batch]

    return images, targets, subsets, timestamps


if __name__ == "__main__":
    print(" ----------- DATALOADER CONFIG ----------- ")
    print("Batch size : 16")
    print("Train workers : 6")
    print("Validation workers : 6")
    print("Sequence length : 5")
    print("---------------------------------\n")

    train_loader = DataLoader(
        train_set,
        batch_size=16,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=sequence_collate_fn,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=16,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=sequence_collate_fn,
    )
    test_loader = DataLoader(test_set, batch_size=16, collate_fn=sequence_collate_fn)

    model = CNN_LSTM()

    model.count_params()

    trainer = Trainer(model, train_loader, val_loader)

    trainer.fit(epochs=100)
