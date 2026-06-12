"""
LeapGestRecog dataset loader.
"""

from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms


CLASS_NAMES = [
    "palm",        # 01_palm
    "l",           # 02_l
    "fist",        # 03_fist
    "fist_moved",  # 04_fist_moved
    "thumb",       # 05_thumb
    "index",       # 06_index
    "ok",          # 07_ok
    "palm_moved",  # 08_palm_moved
    "c",           # 09_c
    "down",        # 10_down
]


class LeapGestRecogDataset(Dataset):
    """Indexes all (image_path, class_idx) pairs found under root."""

    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        for subject_dir in sorted(self.root.iterdir()):
            if not subject_dir.is_dir():
                continue
            for class_dir in sorted(subject_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                try:
                    class_idx = int(class_dir.name.split("_")[0]) - 1
                except ValueError:
                    continue
                if not (0 <= class_idx < 10):
                    continue
                for img_path in class_dir.glob("*.png"):
                    self.samples.append((str(img_path), class_idx))

        if not self.samples:
            raise RuntimeError(
                f"No images found under {self.root}. "
                f"Check that the path points to the leapGestRecog folder."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_transforms(image_size=128, augment=True):
    """Returns (train_transform, eval_transform).

    Set augment=False for the no-augmentation ablation.
    """
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    if augment:
        train_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    else:
        train_tf = eval_tf

    return train_tf, eval_tf


class _TransformedSubset(Dataset):
    """Wraps a Subset so the train split can use a different transform than val/test."""

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        real_idx = self.subset.indices[i]
        path, label = self.subset.dataset.samples[real_idx]
        img = Image.open(path).convert("L")
        return self.transform(img), label

def _subject_from_path(path):
    """Extract the subject folder name ('00'..'09') from an image path.

    LeapGestRecog layout is <root>/<subject>/<class>/<image>.png, so the
    grandparent of the image path is the subject ID.
    """
    return Path(path).parent.parent.name


def _split_by_subject(full_dataset, train_subjects, val_subjects, test_subjects):
    """Assign WHOLE subjects to each split — no person appears in two splits."""
    all_subjects = sorted({_subject_from_path(p) for p, _ in full_dataset.samples})
    if train_subjects is None and val_subjects is None and test_subjects is None:
        train_subjects = all_subjects[:7]
        val_subjects = all_subjects[7:8]
        test_subjects = all_subjects[8:]

    train_subjects = set(train_subjects or [])
    val_subjects = set(val_subjects or [])
    test_subjects = set(test_subjects or [])


    overlap = ((train_subjects & val_subjects)
               | (train_subjects & test_subjects)
               | (val_subjects & test_subjects))
    if overlap:
        raise ValueError(f"Subjects appear in more than one split: {sorted(overlap)}")

    unknown = (train_subjects | val_subjects | test_subjects) - set(all_subjects)
    if unknown:
        raise ValueError(
            f"Unknown subject IDs {sorted(unknown)}. "
            f"Available subjects: {all_subjects}"
        )

    if not train_subjects or not val_subjects or not test_subjects:
        raise ValueError("Each of train/val/test must include at least one subject.")

    print(f"Subject split:")
    print(f"  train: {sorted(train_subjects)}")
    print(f"  val:   {sorted(val_subjects)}")
    print(f"  test:  {sorted(test_subjects)}")

    def _indices_for(subj_set):
        return [i for i, (p, _) in enumerate(full_dataset.samples)
                if _subject_from_path(p) in subj_set]

    return (
        Subset(full_dataset, _indices_for(train_subjects)),
        Subset(full_dataset, _indices_for(val_subjects)),
        Subset(full_dataset, _indices_for(test_subjects)),
    )
def get_dataloaders(
    root,
    batch_size=64,
    image_size=128,
    num_workers=4,
    augment=True,
    val_split=0.15,
    test_split=0.15,
    seed=42,
    subject_split=False,
    train_subjects=None,
    val_subjects=None,
    test_subjects=None,
):
    """Build train/val/test DataLoaders.

    Default: random 70/15/15 split across all images. The same person's hand
    appears in every split (subject leakage), inflating accuracy.

    subject_split=True: assign WHOLE subjects to each split. No subject
    appears in more than one. This is the honest evaluation.
    """
    train_tf, eval_tf = get_transforms(image_size=image_size, augment=augment)
    full = LeapGestRecogDataset(root, transform=eval_tf)

    if subject_split:
        train_set, val_set, test_set = _split_by_subject(
            full, train_subjects, val_subjects, test_subjects,
        )
    else:
        n = len(full)
        n_test = int(n * test_split)
        n_val = int(n * val_split)
        n_train = n - n_val - n_test
        gen = torch.Generator().manual_seed(seed)
        train_set, val_set, test_set = random_split(
            full, [n_train, n_val, n_test], generator=gen,
        )

    train_set = _TransformedSubset(train_set, train_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


print("Dataset and dataloader utilities loaded.")