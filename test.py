import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
from collections import defaultdict

from sonar_img_cnn_lstm_prj.model import CNN_LSTM
from sonar_img_cnn_lstm_prj.dataset import SonarDataset
from torch.utils.data import DataLoader
from sonar_img_cnn_lstm_prj.main import sequence_collate_fn
from sonar_img_cnn_lstm_prj.split_subsets import test_set


def evaluate(model, loader, device="cuda"):
    model.eval()

    mse_total = 0
    mae_total = 0
    count = 0

    grouped_results = defaultdict(list)

    with torch.no_grad():
        print(f"-------- Evaluation batches: {len(loader)}")
        for x, y, subsets, timestamps in loader:

            x = x.to(device)
            y = y.to(device).float()

            preds = model(x)

            mse_total += ((preds - y) ** 2).sum().item()
            mae_total += torch.abs(preds - y).sum().item()
            count += y.numel()

            preds = preds.cpu().numpy()
            targets = y.cpu().numpy()

            for pred, target, subset, timestamp in zip(
                preds,
                targets,
                subsets,
                timestamps,
            ):
                grouped_results[subset].append(
                    {
                        "timestamp": timestamp,
                        "pred": pred,
                        "target": target,
                    }
                )

    mse_avg = mse_total / count
    mae_avg = mae_total / count

    print(f"\nTest Results:")
    print(f"MSE: {mse_avg:.6f}")
    print(f"MAE: {mae_avg:.6f}")

    return grouped_results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # same split logic as training
    dataset = SonarDataset(root_dir="data", seq_len=5)

    test_loader = DataLoader(
        test_set,
        batch_size=16,
        num_workers=4,
        pin_memory=True,
        collate_fn=sequence_collate_fn,
    )

    model = CNN_LSTM().to(device)
    model.load_state_dict(torch.load("models/best_model.pt", map_location=device))

    # evaluate on test set and get results grouped by subset
    grouped_results = evaluate(model, test_loader)

    for subset, values in grouped_results.items():
        values = sorted(values, key=lambda x: x["timestamp"])

        preds = [value["pred"] for value in values]
        targets = [value["target"] for value in values]

        plt.figure(figsize=(10, 15))

        plt.plot(targets, label="True")
        plt.plot(preds, label="Pred")

        plt.xlabel("Sequence Step / Timestamp")
        plt.ylabel("Speed Over Ground (knots)")

        plt.legend()
        plt.title(f"Prediction vs Ground Truth - {subset}")

        plt.savefig(f"plots/prediction_plot_{subset}.png", dpi=150)
        plt.close()
