# Sonar Image Speed Prediction using CNN-LSTM

This project predicts vessel speed (Speed Over Ground, SOG) from sequences of sonar images.

The main idea is to use a CNN to extract spatial information from each sonar image and an LSTM to learn how that information changes over time. Instead of using one image at a time, the model looks at multiple consecutive frames to learn motion patterns.

The idea was partly inspired by this [Udacity project](https://github.com/udacity/self-driving-car/blob/master/steering-models/community-models/komanda/solution-komanda.ipynb)

## Data extraction

To extract the required sonar images and metadata, I used a provided by my teacher extraction script and modified it to:

- support newer Python versions

- work with my dataset structure

#### _One of the difficult parts was setting up a compatible ROS environment._

Because the extraction environment required:

- Ubuntu 20.04

- ROS Noetic

- rosbag

Only after creating a compatible environment I was able to extract sonar images and metadata correctly.

It was interesting to work with ROS tooling and old dependencies again 😄.

## Dataset

I created a custom PyTorch dataset for sequence learning.

The main preprocessing steps are:

- convert images to grayscale

- resize from 500 × 530 to 256 × 256

- normilized pixel values to the range [0, 1]

- convert to PyTorch tensors

The grayscale conversion removes colour information that is not useful for this sonar data. Resizing also reduces the computational cost, which is useful for a relatively small model.

During training, small Gaussian noise is added to the images as a simple form of data augmentation. The goal is to reduce overfitting.

## Sequence generation

The dataset creates sequences of five consecutive sonar frames and uses a future frame to provide the target SOG.

_The configuration is:_

- sequence length: 5

- stride: 2

- future offset: 1

**For example:**

frames 1–5 → predict frame 6 speed
frames 3–7 → predict frame 8 speed

Each sample contains: the image sequence, target SOG, subset name, target timestamp.

This logic was implemented inside the custom dataset loader.

## Preventing data leakage

A very important decision was splitting data by recording subset, not by individual sample.

Problem:

- train gets frames 1–5
- validation gets frames 2–6
- test gets frames 3–7

This causes leakage because samples overlaped.

I decided split by subset / recording session instead.

Final example split:

- Train samples: 1194
- Validation samples: 153
- Test samples: 301

## Model

The model consists of:

1. CNN encoder
2. LSTM sequence model
3. Regression head

Pipeline:

- CNN extracts spatial features from each frame
- LSTM learns temporal relationships
- Fully connected head predicts speed

### CNN encoder

The CNN uses residual convolution blocks inspired by ResNet.

The encoder progressively extracts higher-level features from each sonar frame. It uses dropout for regularization and adaptive average pooling before projecting the image features to a 128-dimensional embedding.

The main feature dimensions are:

1 → 16 → 32 → 64 → 64 → 128

Each image in the sequence is processed independently by the CNN.

### LSTM

The CNN embeddings are then passed to an LSTM.

Configuration:

input size: 128

hidden size: 128

layers: 2

dropout: 0.2

The LSTM is used because the problem is sequential. It can learn relationships between consecutive sonar frames and use those temporal patterns for speed prediction.

The final LSTM output is passed to a small fully connected regression head that predicts SOG.
