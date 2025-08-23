import json
import os

import torch
import click
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from src.data import get_dataloaders
from src.model import load_openclip, build_zeroshot_classifier, encode_images_tensor


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--model-name", default="ViT-B-32", show_default=True)
@click.option("--pretrained",   default="laion2b_s34b_b79k", show_default=True)
@click.option("--batch-size",   default=128, type=int, show_default=True)
@click.option("--num-workers",  default=2, type=int, show_default=True)
@click.option("--templates-json", required=True,
              help="Path to a JSON file containing a list of prompt templates (strings).")
@click.option("--save-confmat", default="", show_default=True,
              help="Optional path to save confusion matrix tensor (torch .pt).")
@click.option("--save-metrics", default="", show_default=True,
              help="Optional path to save metrics (JSON).")
def main(model_name, pretrained, batch_size, num_workers,
         templates_json, save_confmat, save_metrics):
    """
    Evaluate zero-shot OpenCLIP on CIFAR-10 using templates from a JSON file.
    Reports overall accuracy and per-class accuracy.
    """

    _, test_loader, preprocess = get_dataloaders(
        model_name=model_name,
        pretrained=pretrained,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    class_names = test_loader.dataset.classes
    num_classes = len(class_names)

    model, tokenizer, device = load_openclip(model_name, pretrained)

    with open(templates_json, "r") as f:
        templates = json.load(f)
    if not isinstance(templates, list) or not all(isinstance(t, str) for t in templates):
        raise ValueError("templates-json must be a JSON array of strings.")

    click.echo(f"Building zero-shot weights for {num_classes} classes using {len(templates)} templates...")
    zs_weights = build_zeroshot_classifier(
        class_names, model, tokenizer, device, templates=templates
    )  # (C, D)

    click.echo("Evaluating on CIFAR-10 test set...")
    model.eval()
    logit_scale = model.logit_scale.exp().item() if hasattr(model, "logit_scale") else 100.0

    correct = 0
    total = 0

    cls_total = torch.zeros(num_classes, dtype=torch.long)
    cls_correct = torch.zeros(num_classes, dtype=torch.long)

    confmat = torch.zeros(num_classes, num_classes, dtype=torch.long)

    with torch.no_grad():
        for images, labels in tqdm(test_loader, ncols=100):
            images = images.to(device)
            labels = labels.to(device)

            img_feats = encode_images_tensor(images, model, device)  # (B, D)
            logits = logit_scale * (img_feats @ zs_weights.T)         # (B, C)
            preds = logits.argmax(dim=-1)                              # (B,)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for y, yhat in zip(labels.tolist(), preds.tolist()):
                cls_total[y] += 1
                if y == yhat:
                    cls_correct[y] += 1
                confmat[y, yhat] += 1

    overall_acc = correct / total
    per_class_acc = (cls_correct.float() / cls_total.clamp(min=1).float()).tolist()

    click.echo("\n=== Zero-shot results ===")
    click.echo(f"Model: {model_name} ({pretrained})")
    click.echo(f"Templates: {len(templates)} from {templates_json}")
    click.echo(f"Overall Top-1 Accuracy: {overall_acc*100:.2f}%\n")

    click.echo("Per-class Top-1 Accuracy:")
    for i, (name, acc) in enumerate(zip(class_names, per_class_acc)):
        click.echo(f"  {i:2d}. {name:<12} {acc*100:6.2f}%")

    if save_confmat:
        os.makedirs(os.path.dirname(save_confmat) or ".", exist_ok=True)
        plt.figure(figsize=(10, 8))
        confmat_norm = confmat.float() / confmat.sum(dim=1, keepdim=True).clamp(min=1)
        sns.heatmap(confmat_norm.numpy(), annot=False, cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, cbar=True)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Zero-shot Confusion Matrix ({model_name})")
        plt.tight_layout()
        plt.savefig(save_confmat, dpi=300)
        plt.close()
        click.echo(f"Saved confusion matrix heatmap image to: {save_confmat}")

    if save_metrics:
        metrics = {
            "model_name": model_name,
            "pretrained": pretrained,
            "templates_json": templates_json,
            "overall_top1": overall_acc,
            "per_class_top1": {name: acc for name, acc in zip(class_names, per_class_acc)},
            "num_templates": len(templates),
            "num_classes": num_classes,
        }
        os.makedirs(os.path.dirname(save_metrics) or ".", exist_ok=True)
        with open(save_metrics, "w") as f:
            json.dump(metrics, f, indent=2)
        click.echo(f"Saved metrics JSON to: {save_metrics}")


if __name__ == "__main__":
    main()
