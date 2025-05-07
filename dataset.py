import os

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


class DataSet(Dataset):
    def __init__(self, directory, transform=None):
        super().__init__()

        self.directory = directory
        self.transform = transform

        self.echogenicity = []
        self.elasticity = []
        self.masks = []

        self.load()

    def load(self):
        for file in os.listdir(self.directory):
            if file.endswith("mask.png"):
                base = file[:-8]

                mask_path = os.path.join(self.directory, base + "mask.png")
                echogenicity_path = os.path.join(self.directory, base + "bmode.png")
                elasticity_path = os.path.join(self.directory, base + "axial.png")

                try:
                    Image.open(mask_path).verify()
                    Image.open(echogenicity_path).verify()
                    Image.open(elasticity_path).verify()

                    self.masks.append(mask_path)
                    self.echogenicity.append(echogenicity_path)
                    self.elasticity.append(elasticity_path)

                except Exception:
                    continue

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, index):
        echogenicity = Image.open(self.echogenicity[index]).convert("L")
        elasticity = Image.open(self.elasticity[index]).convert("L")
        mask = Image.open(self.masks[index]).convert("L")

        if self.transform:
            echogenicity = self.transform(echogenicity)
            elasticity = self.transform(elasticity)
            mask = self.transform(mask)

            to_tensor = transforms.ToTensor()
            echogenicity = to_tensor(echogenicity)
            elasticity = to_tensor(elasticity)
            mask = to_tensor(mask)

            # normalize = transforms.Normalize([0.5], [0.5])
            # echogenicity = normalize(echogenicity)
            # elasticity = normalize(elasticity)

            mask = (mask > 0.5).float()

        return echogenicity, elasticity, mask
