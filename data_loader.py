from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import dct_transform
import torch
import time
import numpy as np

# data path
trainset_path = "train"
valset_path = "val"
testset_path = "test"

#Hyper-parameters about data
CHANNELS_IMG = 3
IMG_SIZE = 224
BATCH_SIZE = 64

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5 for _ in range(CHANNELS_IMG)], std=[0.5 for _ in range(CHANNELS_IMG)])
])

def load(path, shuffle=True):
    img_data = DataLoader(
        dataset=datasets.ImageFolder(root=path, transform=transform),
        batch_size=BATCH_SIZE,
        shuffle=shuffle
    )
    data = []
    for (images, label) in img_data:
        dct_imgs = torch.stack([dct_transform.dct_transform(images[i].numpy()) for i in range(images.shape[0])])
        data.append((images, dct_imgs, label))
    return data


def get_datasets():
    return load(trainset_path), load(valset_path), load(testset_path, False)

def test():
    start_stamp = time.time()
    train_set, val_set, test_set = get_datasets()
    cost = int(time.time() - start_stamp)
    print("loading time cost: {} min {} sec".format(int(cost / 60), cost % 60))
    img, dct_imgs, label = next(iter(train_set))
    assert img.shape == (BATCH_SIZE, CHANNELS_IMG, IMG_SIZE, IMG_SIZE)
    assert label.shape[0] == BATCH_SIZE
    assert dct_imgs.shape == (BATCH_SIZE, 64, 250)

    img, dct_imgs, label = next(iter(val_set))
    assert img.shape == (BATCH_SIZE, CHANNELS_IMG, IMG_SIZE, IMG_SIZE)
    assert label.shape[0] == BATCH_SIZE
    assert dct_imgs.shape == (BATCH_SIZE, 64, 250)

    img, dct_imgs, label = next(iter(test_set))
    assert img.shape == (BATCH_SIZE, CHANNELS_IMG, IMG_SIZE, IMG_SIZE)
    assert label.shape[0] == BATCH_SIZE
    assert dct_imgs.shape == (BATCH_SIZE, 64, 250)
    print("test pass")