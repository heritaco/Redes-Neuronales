# %% 
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch, numpy as np

painter_to_genre = {
    "Joan": "Surrealismo",
    "Salvador": "Surrealismo",
    "Rene": "Surrealismo",
    "Pablo": "Cubismo",
    "Jackson": "Expresionismo abstracto",
    "Rembrandt": "Barroco",
    "Caravaggio": "Barroco",
    "Camille": "Impresionismo",
    "Alfred": "Impresionismo",
    "Claude": "Impresionismo",
    "Vincent": "Impresionismo",  
}

genre_to_label = {
    "Surrealismo": 0,
    "Cubismo": 1,
    "Expresionismo abstracto": 2,
    "Barroco": 3,
    "Impresionismo": 4,
}

# %% 
from pathlib import Path
import pandas as pd

root = Path("08 Clasificacion de Movimientos Artisticos/data")

# recoge todos los .jpg (ajusta extensiones si hay .png, etc.)
paths = sorted(root.glob("*.jpg"))

rows = []
for p in paths:
    stem = p.stem  # 'Joan_Miro_001'
    # ejemplo simple: pintor = todo menos la parte final si es número
    painter = stem.split("_")[0]
    genre = painter_to_genre[painter]
    label = genre_to_label[genre]

    rows.append({
        "path": str(p),
        "painter": painter,
        "genre": genre,
        "label": label,
    })

df = pd.DataFrame(rows)
# %%

from torch.utils.data import Dataset
from PIL import Image

class PaintingDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        y = int(row["label"])
        if self.transform is not None:
            img = self.transform(img)
        return img, y
    
transf1 = transforms.Compose([
    transforms.RandomResizedCrop(448),
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])

transf2 = transforms.Compose([
    transforms.RandomResizedCrop(448), # 448 means zoom in a lot (like 75%)
    transforms.Resize(128),
    transforms.RandomHorizontalFlip(p=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), # normalize for pre-trained models
])

transf3 = transforms.Compose([
    transforms.RandomResizedCrop(448), # 448 means zoom in a lot (like 75%)
    transforms.Resize(128),
    transforms.RandomVerticalFlip(p=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), # normalize for pre-trained models
])

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, ConcatDataset

K_FOLDS = 10
skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

X = df["path"].values
y = df["label"].values

folds = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    train_df = df.iloc[train_idx]
    val_df   = df.iloc[val_idx]

    # base: todas las imágenes con transf2
    train_base = PaintingDataset(train_df, transform=transf2)
    # extra: mismas imágenes con transf1 (rotadas/augment)
    train_aug  = PaintingDataset(train_df, transform=transf1)

    # dataset de entrenamiento = concatenación
    train_ds = ConcatDataset([train_base, train_aug])

    # validación SOLO con transf2 (sin augment loco)
    val_ds   = PaintingDataset(val_df, transform=transf2)

    train_ld = DataLoader(train_ds, batch_size=16, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_ld   = DataLoader(val_ds, batch_size=32, shuffle=False,
                          num_workers=4, pin_memory=True)

    folds.append((train_ld, val_ld))
    print(f"Fold {fold}: base={len(train_base)}, aug={len(train_aug)}, "
          f"train_total={len(train_ds)}, val={len(val_ds)}")


import torch
import matplotlib.pyplot as plt

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

def denorm(img):
    # img: tensor C×H×W en espacio normalizado
    return (img * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)

def show_img(img, title=None):
    # img: C×H×W
    # img = denorm(img)
    img_np = img.permute(1, 2, 0).cpu().numpy()
    plt.imshow(img_np)
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.show()


idx = np.random.randint(0, len(train_ld.dataset)/2-1)

img2, y2 = train_base[idx]   # transf2
img1, y1 = train_aug[idx]    # misma imagen, transf1

# print labels
print("Showing random image index:", idx)
print("-------------------------")
print(f"Class: {y1} ({list(genre_to_label.keys())[list(genre_to_label.values()).index(y1)]})")

print("label transf1:", y1)
show_img(img1, title="transf1 (rotated/aug)")

print("label transf2:", y2)
show_img(img2, title="transf2 (clean)")

import matplotlib.pyplot as plt
import torch

i = idx
# take one example from your *training* dataset (with transform)
img, y = train_ld.dataset[i]          # C×H×W tensor, already transformed
print("label:", y, "genre:", train_ld.dataset.datasets[0].df.iloc[i]["genre"])
print("shape:", img.shape)

# view the image (need to unnormalize and convert to H×W×C)
mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
# img_view = img * std + mean      # unnormalize
img_view = img.permute(1,2,0).numpy()  # C×H×W -> H×W×C

img_view = img * std + mean      # unnormalize
img_view = img_view.permute(1,2,0).numpy()  # C×H×W -> H×W×C

plt.imshow(img_view)
plt.axis("off")
plt.show()


# %%
