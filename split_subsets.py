from collections import defaultdict
import random

from sonar_img_cnn_lstm_prj.dataset import SonarDataset
from torch.utils.data import Subset

train_dataset = SonarDataset(root_dir="data", seq_len=5, augment=True)
eval_dataset = SonarDataset(root_dir="data", seq_len=5, augment=False)

# group indices by subset
subset_to_indices = defaultdict(list)

# sample (image_paths, target, subset, target_timestamp)
for i, sample in enumerate(eval_dataset.samples):
    subset = sample[2]
    subset_to_indices[subset].append(i)

subsets = list(subset_to_indices.keys())
random.seed(42)
random.shuffle(subsets)

# split subsets
train_subsets = subsets[: int(0.7 * len(subsets))]
val_subsets = subsets[int(0.7 * len(subsets)) : int(0.85 * len(subsets))]
test_subsets = subsets[int(0.85 * len(subsets)) :]


print("Train subsets:", train_subsets)
print("Val subsets:", val_subsets)
print("Test subsets:", test_subsets)


def collect_indices(subset_list):
    indices = []
    for subset in subset_list:
        indices.extend(subset_to_indices[subset])

    return indices


train_indices = collect_indices(train_subsets)
val_indices = collect_indices(val_subsets)
test_indices = collect_indices(test_subsets)


# train uses augmented dataset
# val/test use clean dataset

train_set = Subset(train_dataset, train_indices)
val_set = Subset(eval_dataset, val_indices)
test_set = Subset(eval_dataset, test_indices)

print("\n--------- SPLIT SUMMARY ---------")
print(f"Train samples : {len(train_indices)}")
print(f"Val samples   : {len(val_indices)}")
print(f"Test samples  : {len(test_indices)}")
print("--------------------\n")
