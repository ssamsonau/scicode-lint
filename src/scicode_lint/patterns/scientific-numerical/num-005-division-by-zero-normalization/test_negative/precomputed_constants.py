import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def normalize_image_imagenet(image):
    image = image / 255.0

    for c in range(3):
        image[:, :, c] = (image[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]

    return image


def get_image_transforms():
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    )


def normalize_with_sklearn(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def minmax_scale_data(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


def normalize_cifar(batch):
    batch = batch.astype(np.float32) / 255.0
    for i, (m, s) in enumerate(zip(CIFAR_MEAN, CIFAR_STD)):
        batch[:, :, :, i] = (batch[:, :, :, i] - m) / s
    return batch
