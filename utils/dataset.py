import os
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset

class Caltech101Dataset(Dataset):
    def __init__(self, root_dir, split_file, transform=None):
        self.root = root_dir
        self.transform = transform
        with open(split_file) as f:
            lines = f.read().splitlines()
        self.samples = [line.split() for line in lines]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        img_path = os.path.join(self.root, rel_path)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, int(label)
