import torch
from torch.utils.data import Dataset, random_split
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomCrop, ToTensor, Normalize

class BaseDataset(Dataset):
    def __init__(self, transform=None):
        super(BaseDataset, self).__init__()
        self.transform = transform

class MetaTaskDataset(BaseDataset):
    def __init__(self, data, labels, transform=None):
        super(MetaTaskDataset, self).__init__(transform=transform)
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label
    
class AugmentedDataset(Dataset):
    def __init__(self, dataset, num_transforms=2):
        self.dataset = dataset
        self.num_transforms = num_transforms
        self.transform = Compose([
            RandomHorizontalFlip(),
            RandomCrop(32, padding=4),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        x, y = self.dataset[index]
        for _ in range(self.num_transforms):
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.dataset)
    
def train_test_split(dataset, train_ratio=0.8):
    train_len = int(len(dataset) * train_ratio)
    test_len = len(dataset) - train_len
    return random_split(dataset, [train_len, test_len])

def get_data_augmentation():
    return Compose([
        RandomHorizontalFlip(),
        RandomCrop(28, padding=4),
        ToTensor()
    ])

class OtherDataset:
    def __init__(self):
        pass

    # Add the methods for the OtherDataset here
