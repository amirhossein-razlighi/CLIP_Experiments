import os
import json
import time
from dataclasses import dataclass

import click
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from src.data import get_dataloaders
from src.model import load_openclip, encode_images_tensor


@dataclass
class ProbeResults:
    overall_top1: float
    per_class_top1: dict
    confmat: torch.Tensor


def _standardize_features(X: np.ndarray) -> tuple[np.ndarray, dict]:
    """Zero-mean, unit-variance per dimension; returns X_std and stats for reproducibility."""
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-8
    Xs = (X - mu) / sigma
    return Xs, {"mean": mu.squeeze(0).tolist(), "std": sigma.squeeze(0).tolist()}


def _extract_features(model, device, loader) -> tuple[np.ndarray, np.ndarray]:
    """Return (features, labels) with CLIP encoder frozen. Features are L2-normalized (like zero-shot)."""
    model.eval()
    feats = []
    labels = []
    with torch.no_grad():
        for images, ys in tqdm(loader, desc="Extracting features", ncols=100):
            images = images.to(device)
            f = encode_images_tensor(images, model, device)  # (B, D)
            feats.append(f.cpu().numpy())
            labels.append(ys.numpy())
    X = np.concatenate(feats, axis=0)
    y = np.concatenate(labels, axis=0)
    return X, y


def _evaluate_predictions(
    preds: np.ndarray, y: np.ndarray, class_names: list[str]
) -> ProbeResults:
    num_classes = len(class_names)
    confmat = torch.zeros(num_classes, num_classes, dtype=torch.long)
    correct = 0
    cls_total = torch.zeros(num_classes, dtype=torch.long)
    cls_correct = torch.zeros(num_classes, dtype=torch.long)

    for yi, pi in zip(y, preds):
        confmat[yi, pi] += 1
        cls_total[yi] += 1
        if yi == pi:
            correct += 1
            cls_correct[yi] += 1

    overall = correct / len(y)
    per_class = {
        name: (cls_correct[i].float() / max(1, cls_total[i].item())).item()
        for i, name in enumerate(class_names)
    }

    return ProbeResults(overall_top1=overall, per_class_top1=per_class, confmat=confmat)


def _save_confmat_image(
    confmat: torch.Tensor, class_names: list[str], out_path: str, title: str
):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    confmat = confmat.float()
    confnorm = confmat / confmat.sum(dim=1, keepdim=True).clamp(min=1)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confnorm.numpy(),
        annot=False,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


@click.command()
@click.option("--model-name", default="ViT-B-32", show_default=True)
@click.option("--pretrained", default="laion2b_s34b_b79k", show_default=True)
@click.option("--batch-size", default=256, type=int, show_default=True)
@click.option("--num-workers", default=2, type=int, show_default=True)
@click.option(
    "--standardize/--no-standardize",
    default=True,
    show_default=True,
    help="Standardize features before training the classifier.",
)
@click.option(
    "--save-metrics", default="", show_default=True, help="Path to save metrics JSON."
)
@click.option(
    "--save-confmat",
    default="",
    show_default=True,
    help="Path to save confusion matrix tensor (.pt).",
)
@click.option(
    "--save-confmat-img",
    default="",
    show_default=True,
    help="Path to save confusion heatmap (PNG/JPG).",
)
@click.option(
    "--save-features-prefix",
    default="",
    show_default=True,
    help='If set, saves train/test features as "<prefix>_train.npz" and "<prefix>_test.npz".',
)
def main(
    model_name,
    pretrained,
    batch_size,
    num_workers,
    standardize,
    save_metrics,
    save_confmat,
    save_confmat_img,
    save_features_prefix,
):
    """
    Linear probe on frozen CLIP image features (CIFAR-10).
    """

    train_loader, test_loader, _ = get_dataloaders(
        model_name=model_name,
        pretrained=pretrained,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    class_names = train_loader.dataset.classes
    num_classes = len(class_names)

    model, tokenizer, device = load_openclip(model_name, pretrained)

    t0 = time.time()
    X_train, y_train = _extract_features(model, device, train_loader)
    X_test, y_test = _extract_features(model, device, test_loader)
    t_extract = time.time() - t0

    if save_features_prefix:
        os.makedirs(os.path.dirname(save_features_prefix) or ".", exist_ok=True)
        np.savez_compressed(f"{save_features_prefix}_train.npz", X=X_train, y=y_train)
        np.savez_compressed(f"{save_features_prefix}_test.npz", X=X_test, y=y_test)

    std_stats = None
    if standardize:
        X_train, std_stats = _standardize_features(X_train)
        mu = np.array(std_stats["mean"], dtype=np.float32, ndmin=2)
        sd = np.array(std_stats["std"], dtype=np.float32, ndmin=2)
        X_test = (X_test - mu) / sd

    t1 = time.time()

    D = X_train.shape[1]
    LR = 1e-1
    clf = torch.nn.Linear(D, num_classes)
    opt = torch.optim.SGD(clf.parameters(), lr=LR, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()
    device_head = (
        torch.device("mps")
        if (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    )
    clf = clf.to(device_head)
    Xtr = torch.from_numpy(X_train).to(device_head).float()
    ytr = torch.from_numpy(y_train).to(device_head).long()
    Xte = torch.from_numpy(X_test).to(device_head).float()

    EPOCHS = 100
    for epoch in range(EPOCHS):
        clf.train()
        opt.zero_grad(set_to_none=True)
        logits = clf(Xtr)
        loss = loss_fn(logits, ytr)
        loss.backward()
        opt.step()
    train_time = time.time() - t1
    clf.eval()
    with torch.no_grad():
        preds = clf(Xte).argmax(dim=-1).cpu().numpy()
    head_desc = f"torch.nn.Linear(D,{num_classes}), epochs={EPOCHS}, lr={LR}"

    results = _evaluate_predictions(preds, y_test, class_names)

    click.echo("\n=== Linear Probe Results ===")
    click.echo(f"Model: {model_name} ({pretrained})")
    click.echo(f"Probe: {head_desc}")
    click.echo(
        f"Feature extraction time: {t_extract:.1f}s | Train time: {train_time:.1f}s"
    )
    click.echo(f"Overall Top-1: {results.overall_top1*100:.2f}%\n")
    click.echo("Per-class Top-1:")
    for i, name in enumerate(class_names):
        click.echo(f"  {i:2d}. {name:<12} {results.per_class_top1[name]*100:6.2f}%")

    if save_confmat:
        os.makedirs(os.path.dirname(save_confmat) or ".", exist_ok=True)
        torch.save(
            {"confusion_matrix": results.confmat, "classes": class_names}, save_confmat
        )
        click.echo(f"\nSaved confusion matrix tensor to: {save_confmat}")

    if save_confmat_img:
        _save_confmat_image(
            results.confmat,
            class_names,
            save_confmat_img,
            title=f"Linear Probe Confusion Matrix ({model_name})",
        )
        click.echo(f"Saved confusion matrix heatmap image to: {save_confmat_img}")

    if save_metrics:
        metrics = {
            "model_name": model_name,
            "pretrained": pretrained,
            "probe": head_desc,
            "overall_top1": results.overall_top1,
            "per_class_top1": results.per_class_top1,
            "feature_extraction_seconds": t_extract,
            "train_seconds": train_time,
            "standardized": standardize,
        }
        if std_stats is not None:
            metrics["standardize_stats_shape"] = {
                "mean": len(std_stats["mean"]),
                "std": len(std_stats["std"]),
            }
        os.makedirs(os.path.dirname(save_metrics) or ".", exist_ok=True)
        with open(save_metrics, "w") as f:
            json.dump(metrics, f, indent=2)
        click.echo(f"Saved metrics JSON to: {save_metrics}")


if __name__ == "__main__":
    main()
