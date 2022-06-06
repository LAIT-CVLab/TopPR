# Import pickle
import pickle
import numpy as np
from typing import Any, Callable, Optional, Tuple
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from PIL import Image

def open_crater_dataset(path):
    """
    Open crater Datset
    return: crater dataset
    """
    f = open(path, "rb")
    crater = pickle.load(f)
    f.close()
    return crater


def realdataset(data_path:str, desc:str, transforms=None, train=True, **kwargs):
    """
    get Real Dataset (provided by Pytorch official) with MNIST / fashionMNIST / CIFAR10
    :data_path: dataset path for downloading dataset
    :desc: kind of dataset
    :transforms: transform for dataset
    :train: load train dataset (default : True)
    
    :return: dataset
    """
    if desc == 'mnist':
        from torchvision.datasets import MNIST
        dataset = MNIST(data_path, transform=transforms, train=train, download=True)
    elif desc == 'fashionMNIST':
        from torchvision.datasets import FashionMNIST
        dataset = FashionMNIST(data_path, train=train, transform=transforms, download=True)
    elif desc == 'cifar10':
        from torchvision.datasets import CIFAR10
        dataset = CIFAR10(data_path, train=train, transform=transforms, download=True)

    elif desc == 'custom':
        # if you have custom datasets, you can modify this code line.
        dataset == CustomDataset(data_path, kwargs['npfile'], transform=transforms)
    return dataset


def label_split(dataset: datasets, num: int, random_sample: bool) -> Tuple[Any, Any]:
    """
    Input
    :dataset: input dataset
    :num: the number of datas per each class
    :random_sample: Random sampling index in each class

    Output
    :dataset datasets: the data(image) and targets(label) is updated in dataset
    :label_data dict: label index per each class
    """
    # class 개수
    num_original_class = len(np.unique(dataset.targets))
    # class list 
    classes = list(np.array(range(num_original_class)))

    # label index
    indexes = []
    for c in range(num_original_class):
        indexes.append(list(np.where(dataset.targets == c)[0]))

    # if random_sample is True, Index is random sampled.
    if random_sample:
        initbox = [np.random.choice(index, num, False) for index in indexes]
    else:
        initbox = [index[:num] for index in indexes]

    
    label_data = {}
    for idx, lists in enumerate(initbox):
        label_data[idx] = lists
    initbox = np.array(initbox).flatten()
    dataset.data = dataset.data[initbox]
    dataset.targets = dataset.targets[initbox]
    
    return dataset, label_data


class CustomDataset(Dataset):
    def __init__(self, path, npfile, transform=None, target_transform=None):
        super(Dataset).__init__()
        numpy = np.load(path + npfile, allow_pickle=True)
        self.data = numpy['data']
        self.data = np.concatenate(self.data, axis=0)
        self.target = numpy['target']
        self.target = np.concatenate(self.target, axis=0)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx].squeeze(-1)
        img = Image.fromarray(img, mode="L")

        target = self.target[idx]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def get_dataset(data_path:str, desc:str, split=True, transforms=None) -> Tuple[Any, Any]:
    """
    Input
    :data_path: path for downloding or loading dataset
    :desc: description for dataset
    :split: label split
    :transforms: If preprocess is true, transforms is set to torchvision libraries.

    Output
    :dataset datasets: the data(image) and targets(label) is updated in dataset
    :label_data dict: label index per each class
    """
    # Preprocessing setting
    # Resize to image_size for VGG16 or InceptionV3
    # If the dataset is grayscale, convert 1 channel to 3 channel
    # 0 ~ 255 image normalize to 0 ~ 1
    # 0 ~ 1 image normalize to -1 ~ 1
    dataset = realdataset(data_path, desc, transforms=transforms, train=True)
    if split:
        dataset, label_data = label_split(dataset, 1000)
        return dataset, label_data
    return dataset

def preprocessing(torchvision, desc, image_size, fair_size = None):
    if torchvision:
        transformation = []
        # 만약 Generator가 학습한 이미지가 원본 이미지 사이즈와 다르다면 
        # 먼저 해당 학습된 크기에 맞게 resize를 수행한다.
        if fair_size is not None:
            transformation.extend([transforms.Resize((fair_size, fair_size))])

        transformation.extend([transforms.Resize((image_size,image_size))])
        if desc in ['mnist', 'fashionMNIST']:
            transformation.extend([transforms.Grayscale(3)])
        transformation.extend([transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        return transformation


# preprocessing(torchvision=True, desc='fashionMNIST', image_size=args.image_size, fair_size=args.fair_size)

if __name__ == '__main__':
    get_dataset('/disk1/softjin/dataset/data', 'fashionMNIST')