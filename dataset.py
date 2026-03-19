import os
from typing import Iterable, List, Optional

from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms


IMG_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}


def _is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMG_EXTENSIONS


def _list_images(root: str, recursive: bool = True) -> List[str]:
    if not os.path.isdir(root):
        return []

    images: List[str] = []
    if recursive:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                full_path = os.path.join(dirpath, fn)
                if _is_image_file(full_path):
                    images.append(full_path)
    else:
        for fn in os.listdir(root):
            full_path = os.path.join(root, fn)
            if os.path.isfile(full_path) and _is_image_file(full_path):
                images.append(full_path)
    images.sort()
    return images


class Vanilla(Dataset):
    """
    Simple image-folder dataset.
    Unlike the original version, this loader can traverse nested folders,
    which makes it suitable for CelebA-HQ and similar custom datasets.
    """
    def __init__(self, root, transform=None, recursive: bool = True):
        self.root = root
        self.transform = transform
        self.imgs = _list_images(root, recursive=recursive)
        if len(self.imgs) == 0:
            raise RuntimeError(f"No images found under: {root}")

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, 0  # dummy label for reconstruction tasks

    def __len__(self):
        return len(self.imgs)


def resolve_split_root(data_root: str, split: str) -> str:
    """
    Resolve common split-directory layouts.

    Supported layouts include, for example:
      data_root/train, data_root/val, data_root/test
      data_root/validation
      data_root itself (flat folder, when no split subfolders exist)
    """
    split = split.lower()
    aliases = {
        "train": ["train", "training"],
        "val": ["val", "valid", "validation", "test"],
        "test": ["test", "val", "valid", "validation"],
    }
    for name in aliases.get(split, [split]):
        cand = os.path.join(data_root, name)
        if os.path.isdir(cand) and len(_list_images(cand, recursive=True)) > 0:
            return cand

    # Fallback: use data_root directly if it contains images.
    if os.path.isdir(data_root) and len(_list_images(data_root, recursive=True)) > 0:
        return data_root

    raise RuntimeError(
        f"Could not resolve split='{split}' under data_root='{data_root}'. "
        "Expected image files directly under the root or inside train/val/test subfolders."
    )


def build_transform(
    image_size: int,
    is_train: bool = False,
    random_flip: bool = False,
):
    ops = []
    if image_size and image_size > 0:
        ops.extend([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ])
    if is_train and random_flip:
        ops.append(transforms.RandomHorizontalFlip())
    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)


def create_dataset(
    dataset_name: str,
    split: str,
    data_root: str = "",
    image_size: int = 32,
    random_flip: bool = False,
):
    name = dataset_name.lower()

    if name == "cifar10":
        return datasets.CIFAR10(
            root="./data",
            train=(split.lower() == "train"),
            download=True,
            transform=build_transform(image_size=image_size, is_train=(split.lower() == "train"), random_flip=random_flip),
        )

    if name in ("celebahq", "folder"):
        if not data_root:
            raise ValueError(f"--data_root is required when --dataset {dataset_name}")
        split_root = resolve_split_root(data_root, split)
        tfm = build_transform(image_size=image_size, is_train=(split.lower() == "train"), random_flip=random_flip)
        return Vanilla(root=split_root, transform=tfm, recursive=True)

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def infer_eval_split(dataset_name: str) -> str:
    return "test" if dataset_name.lower() == "cifar10" else "test"


def main():
    data_path = './dataset'
    os.makedirs(data_path, exist_ok=True)
    if not os.path.exists('./dataset/ILSVRC2012_img_train.tar') or not os.path.exists('./dataset/ILSVRC2012_img_val.tar'):
        print('ILSVRC2012_img_train.tar and ILSVRC2012_img_val.tar should be downloaded from https://image-net.org/')
        print('Please download the dataset from https://image-net.org/challenges/LSVRC/2012/2012-downloads and put it in ./dataset')
        raise Exception('not find dataset')
    phases = ['train', 'val']
    for phase in phases:
        print("extracting {} dataset".format(phase))
        path = './dataset/ImageNet/{}'.format(phase)
        print('path is {}'.format(path))
        os.makedirs(path, exist_ok=True)
        print('tar -xf ./dataset/ILSVRC2012_img_{}.tar -C {}'.format(phase, path))
        os.system('tar -xf ./dataset/ILSVRC2012_img_{}.tar -C {}'.format(phase, path))
        if phase == 'train':
            for tar in os.listdir(path):
                print('tar -xf {}/{} -C {}/{}'.format(path, tar, path, tar.split('.')[0]))
                os.makedirs('{}/{}'.format(path, tar.split('.')[0]), exist_ok=True)
                os.system('tar -xf {}/{} -C {}/{}'.format(path, tar, path, tar.split('.')[0]))
                os.remove('{}/{}'.format(path, tar))


if __name__ == '__main__':
    main()
