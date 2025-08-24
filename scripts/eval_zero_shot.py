import json, os
from typing import List, Tuple, Dict

import click
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

from src.data import get_dataloaders
from src.model import (
    load_openclip,
    build_zeroshot_classifier,
    encode_images_tensor,
    encode_texts,
)


def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def topk_hits(logits: torch.Tensor, labels: torch.Tensor, k: int) -> int:
    k = min(k, logits.shape[-1])
    topk = logits.topk(k, dim=-1).indices  # (B, k)
    return sum(int(labels[i].item() in topk[i].tolist()) for i in range(labels.size(0)))


def normalize_confmat(cm: torch.Tensor) -> torch.Tensor:
    cm = cm.float()
    return cm / cm.sum(dim=1, keepdim=True).clamp(min=1)


def confmat_image_saver(
    cm: torch.Tensor, class_names: List[str], out: str, title: str
) -> None:
    ensure_dir(out)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        normalize_confmat(cm).cpu().numpy(),
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def perclass_bar_saver(
    per_class_acc: List[float], class_names: List[str], out: str, title: str
) -> None:
    ensure_dir(out)
    plt.figure(figsize=(10, 4))
    vals = [a * 100 for a in per_class_acc]
    plt.bar(range(len(class_names)), vals)
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.ylabel("Top-1 (%)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def reliability_saver(
    confidences: List[float], correct_flags: List[bool], out: str
) -> float:
    ensure_dir(out)
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(confidences, bins) - 1
    bin_acc, bin_conf, bin_count = [], [], []
    for b in range(len(bins) - 1):
        ids = np.where(idx == b)[0]
        if len(ids) == 0:
            continue
        acc = np.mean([correct_flags[i] for i in ids])
        conf = np.mean([confidences[i] for i in ids])
        bin_acc.append(acc)
        bin_conf.append(conf)
        bin_count.append(len(ids))
    ece = float(
        np.sum([abs(a - c) * n for a, c, n in zip(bin_acc, bin_conf, bin_count)])
        / max(1, sum(bin_count))
    )

    plt.figure(figsize=(4, 4))
    plt.plot([0, 1], [0, 1], "--")
    plt.plot(bin_conf, bin_acc, marker="o")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"Reliability (ECE={ece:.3f})")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    return ece


def get_most_confused(
    cm: torch.Tensor, class_names: List[str], top_n: int = 10
) -> List[Tuple[str, str, float, int]]:
    cmn = normalize_confmat(cm)
    pairs = []
    for i in range(cmn.shape[0]):
        for j in range(cmn.shape[1]):
            if i == j:
                continue
            pairs.append(
                (
                    float(cmn[i, j].item()),
                    class_names[i],
                    class_names[j],
                    int(cm[i, j].item()),
                )
            )
    pairs.sort(key=lambda x: x[0], reverse=True)
    return [
        (a, b, r, c) for r, a, b, c in [(p[0], p[1], p[2], p[3]) for p in pairs[:top_n]]
    ]


def build_text_features_feature_ensemble(classes, model, tokenizer, device, templates):
    return build_zeroshot_classifier(
        classes, model, tokenizer, device, templates=templates
    )  # (C, D)


def build_text_features_logit_ensemble(classes, model, tokenizer, device, templates):
    texts = [[t.format(c) for t in templates] for c in classes]
    flat = [s for group in texts for s in group]
    feats = encode_texts(flat, model, tokenizer, device, normalize=True)  # (C*T, D)
    T = len(templates)
    return feats.view(len(classes), T, -1)  # (C, T, D)


def evaluate(
    test_loader,
    model,
    device,
    class_names,
    text_repr,
    mode: str,
    logit_scale: float,
    topk_max: int,
) -> Dict:
    num_classes = len(class_names)
    confmat = torch.zeros(num_classes, num_classes, dtype=torch.long)
    cls_total = torch.zeros(num_classes, dtype=torch.long)
    cls_correct = torch.zeros(num_classes, dtype=torch.long)

    total = acc1 = acc3 = acc5 = 0
    y_all, yhat_all = [], []
    confidences = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            feats = encode_images_tensor(images, model, device)  # (B, D)

            if mode == "feature":
                # text_repr: (C, D)
                logits = logit_scale * (feats @ text_repr.T)  # (B, C)
            else:
                # text_repr: (C, T, D) -> average logits over T
                C, T, D = text_repr.shape
                logits_per_t = logit_scale * (
                    feats @ text_repr.view(C * T, D).T
                )  # (B, C*T)
                logits = logits_per_t.view(feats.shape[0], C, T).mean(dim=-1)  # (B, C)

            preds = logits.argmax(dim=-1)
            probs = logits.softmax(dim=-1)
            conf = probs.max(dim=-1).values

            bsz = labels.size(0)
            acc1 += (preds == labels).sum().item()
            acc3 += topk_hits(logits, labels, k=min(3, topk_max))
            acc5 += topk_hits(logits, labels, k=min(5, topk_max))
            total += bsz

            y_all.extend(labels.cpu().tolist())
            yhat_all.extend(preds.cpu().tolist())
            confidences.extend(conf.cpu().tolist())

            for y, yhat in zip(labels.tolist(), preds.tolist()):
                cls_total[y] += 1
                if y == yhat:
                    cls_correct[y] += 1
                confmat[y, yhat] += 1

    per_class = (cls_correct.float() / cls_total.clamp(min=1).float()).tolist()
    metrics = {
        "top1": acc1 / total,
        "top3": acc3 / total,
        "top5": acc5 / total,
        "per_class": per_class,
        "y_true": y_all,
        "y_pred": yhat_all,
        "confidences": confidences,
        "confmat": confmat,
    }
    return metrics


@click.command()
@click.option("--model-name", default="ViT-B-32", show_default=True)
@click.option("--pretrained", default="laion2b_s34b_b79k", show_default=True)
@click.option("--batch-size", default=256, type=int, show_default=True)
@click.option("--num-workers", default=4, type=int, show_default=True)
@click.option("--templates-json", required=True, help="JSON list of prompt templates.")
@click.option(
    "--ensemble",
    type=click.Choice(["feature", "logit"]),
    default="feature",
    show_default=True,
)
@click.option("--save-confmat-img", default="", show_default=True)
@click.option("--save-confmat-pt", default="", show_default=True)
@click.option("--save-perclass-bar", default="", show_default=True)
@click.option("--save-reliability", default="", show_default=True)
@click.option("--save-metrics", default="", show_default=True)
@click.option(
    "--save-clfreport",
    default="",
    show_default=True,
    help="Sklearn classification_report JSON.",
)
@click.option("--most-confused", default=10, type=int, show_default=True)
def main(
    model_name,
    pretrained,
    batch_size,
    num_workers,
    templates_json,
    ensemble,
    save_confmat_img,
    save_confmat_pt,
    save_perclass_bar,
    save_reliability,
    save_metrics,
    save_clfreport,
    most_confused,
):
    """Zero-shot evaluation on CIFAR-10 with extra metrics & plots."""
    _, test_loader, _ = get_dataloaders(
        model_name=model_name,
        pretrained=pretrained,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    class_names = test_loader.dataset.classes
    model, tokenizer, device = load_openclip(model_name, pretrained)

    with open(templates_json) as f:
        templates = json.load(f)
    assert isinstance(templates, list) and all(isinstance(t, str) for t in templates)

    if ensemble == "feature":
        text_repr = build_text_features_feature_ensemble(
            class_names, model, tokenizer, device, templates
        )
    else:
        text_repr = build_text_features_logit_ensemble(
            class_names, model, tokenizer, device, templates
        )

    logit_scale = (
        model.logit_scale.exp().item() if hasattr(model, "logit_scale") else 100.0
    )
    res = evaluate(
        test_loader,
        model,
        device,
        class_names,
        text_repr,
        ensemble,
        logit_scale,
        topk_max=len(class_names),
    )

    click.echo(
        f"\nModel: {model_name} ({pretrained}) | Templates: {len(templates)} | Ensemble: {ensemble}"
    )
    click.echo(
        f"Top-1: {res['top1']*100:.2f}% | Top-3: {res['top3']*100:.2f}% | Top-5: {res['top5']*100:.2f}%\n"
    )

    for i, (name, acc) in enumerate(zip(class_names, res["per_class"])):
        click.echo(f"{i:2d}. {name:<12} {acc*100:6.2f}%")

    pairs = get_most_confused(res["confmat"], class_names, top_n=most_confused)
    if pairs:
        click.echo("\nMost-confused pairs:")
        for a, b, rate, cnt in pairs:
            click.echo(f"  {a:12} → {b:12}  rate={rate*100:5.2f}%  count={cnt}")

    if save_confmat_img:
        confmat_image_saver(
            res["confmat"],
            class_names,
            save_confmat_img,
            f"Zero-shot Confusion ({model_name}, {ensemble})",
        )
        click.echo(f"Saved confusion heatmap: {save_confmat_img}")

    if save_perclass_bar:
        perclass_bar_saver(
            res["per_class"],
            class_names,
            save_perclass_bar,
            f"Per-class Top-1 ({model_name}, {ensemble})",
        )
        click.echo(f"Saved per-class bar: {save_perclass_bar}")

    if save_reliability:
        ece = reliability_saver(
            res["confidences"],
            [yt == yp for yt, yp in zip(res["y_true"], res["y_pred"])],
            save_reliability,
        )
        click.echo(f"Saved reliability (ECE={ece:.3f}): {save_reliability}")

    if save_confmat_pt:
        ensure_dir(save_confmat_pt)
        torch.save(
            {"confusion_matrix": res["confmat"], "classes": class_names},
            save_confmat_pt,
        )
        click.echo(f"Saved confusion tensor: {save_confmat_pt}")

    if save_metrics:
        ensure_dir(save_metrics)
        payload = {
            "model_name": model_name,
            "pretrained": pretrained,
            "templates_json": templates_json,
            "ensemble": ensemble,
            "top1": res["top1"],
            "top3": res["top3"],
            "top5": res["top5"],
            "per_class_top1": {n: a for n, a in zip(class_names, res["per_class"])},
            "num_templates": len(templates),
            "num_classes": len(class_names),
        }
        with open(save_metrics, "w") as f:
            json.dump(payload, f, indent=2)
        click.echo(f"Saved metrics JSON: {save_metrics}")

    if save_clfreport:
        ensure_dir(save_clfreport)
        rep = classification_report(
            res["y_true"],
            res["y_pred"],
            target_names=class_names,
            digits=3,
            output_dict=True,
        )
        with open(save_clfreport, "w") as f:
            json.dump(rep, f, indent=2)
        click.echo(f"Saved classification report: {save_clfreport}")


if __name__ == "__main__":
    main()
