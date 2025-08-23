from typing import Tuple
import math
import random
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.utils as vutils
import open_clip
import click
from tqdm import tqdm
import matplotlib.pyplot as plt


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275,  0.40821073]).view(3, 1, 1)
CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

def denorm_clip(x: torch.Tensor) -> torch.Tensor:
    """x: (B,3,H,W) normalized with CLIP stats -> de-normalized and clipped to [0,1]."""
    x = x.detach().cpu()
    x = x * CLIP_STD + CLIP_MEAN
    return x.clamp(0.0, 1.0)


def get_dataloaders(
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    batch_size: int = 64,
    num_workers: int = 2,
    root: str = "./data",
) -> Tuple[DataLoader, DataLoader, torch.nn.Module]:
    """
    Returns CIFAR-10 train/test DataLoaders with OpenCLIP's official preprocess.
    """
    _, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )

    train_ds = datasets.CIFAR10(root=root, train=True, download=True, transform=preprocess)
    test_ds = datasets.CIFAR10(root=root, train=False, download=True, transform=preprocess)

    pin = False if torch.backends.mps.is_available() else True

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin
    )

    return train_loader, test_loader, preprocess


def save_batch_grid(images: torch.Tensor,
                    labels: torch.Tensor,
                    class_names,
                    out_path: str,
                    max_cols: int | None = None) -> None:
    """
    Save a grid of images with per-image titles.
    images: (B,3,H,W) normalized with CLIP stats.
    labels: (B,)
    class_names: list of class names, e.g. train_loader.dataset.classes
    """
    imgs = denorm_clip(images)  # (B,3,H,W) in [0,1]

    B = imgs.size(0)

    if max_cols is None:
        cols = int(math.ceil(math.sqrt(B)))
    else:
        cols = max_cols
    rows = int(math.ceil(B / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r][c]
            ax.axis("off")
            if idx < B:
                img = imgs[idx].permute(1, 2, 0).numpy()  # HWC
                ax.imshow(img)
                y = int(labels[idx].item())
                ax.set_title(class_names[y], fontsize=8)
            idx += 1

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


@click.command()
@click.option("--model-name", default="ViT-B-32", show_default=True,
              help="OpenCLIP model name.")
@click.option("--pretrained", default="laion2b_s34b_b79k", show_default=True,
              help="Pretrained weights tag for OpenCLIP.")
@click.option("--batch-size", default=64, show_default=True, type=int,
              help="Batch size.")
@click.option("--num-workers", default=2, show_default=True, type=int,
              help="Number of DataLoader workers.")
@click.option("--root", default="./data", show_default=True,
              help="Dataset root directory.")
@click.option("--preview-batches", default=1, show_default=True, type=int,
              help="Preview N batches with tqdm (0 to skip).")
@click.option("--save-grid-path", default="./sample_grid.png", show_default=True,
              help="If set, save a random batch as a labeled image grid to this path.")
@click.option("--grid-cols", default=0, type=int, show_default=True,
              help="Force number of columns in the grid (0 = auto).")
def main(model_name, pretrained, batch_size, num_workers, root,
         preview_batches, save_grid_path, grid_cols):

    train_loader, test_loader, _ = get_dataloaders(
        model_name=model_name,
        pretrained=pretrained,
        batch_size=batch_size,
        num_workers=num_workers,
        root=root,
    )

    train_len = len(train_loader.dataset)
    test_len = len(test_loader.dataset)
    click.echo(
        f"Data is ready | model={model_name} ({pretrained}), "
        f"batch_size={batch_size}, workers={num_workers}, "
        f"samples: train={train_len}, test={test_len}"
    )

    if preview_batches > 0:
        batches = min(preview_batches, len(train_loader))
        pbar = tqdm(range(batches), desc="Previewing train batches")
        it = iter(train_loader)
        for _ in pbar:
            images, labels = next(it)
            pbar.set_postfix({
                "img": tuple(images.shape),
                "labels_min": int(labels.min().item()),
                "labels_max": int(labels.max().item()),
            })
        click.echo("Preview completed.")

    if save_grid_path:
        print("Saving a sample batch from train loader into an image for your inspection...")
        rand_idx = random.randrange(len(train_loader))

        it = iter(train_loader)
        for _ in range(rand_idx):
            next(it)
        images, labels = next(it)

        class_names = getattr(train_loader.dataset, "classes", [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ])

        cols = None if grid_cols <= 0 else grid_cols
        save_batch_grid(images, labels, class_names, save_grid_path, max_cols=cols)
        click.echo(f"Saved labeled grid to: {save_grid_path}")


if __name__ == "__main__":
    main()
