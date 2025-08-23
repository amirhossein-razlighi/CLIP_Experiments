from typing import Iterable, List, Optional
import json
import os

import torch
import open_clip
import click
from PIL import Image


def get_device(prefer_mps: bool = True) -> torch.device:
    """
    Resolve best available device: CUDA -> MPS (Apple Silicon) -> CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_openclip(
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    device: Optional[torch.device] = None,
    eval_mode: bool = True,
):
    """
    Returns (model, tokenizer). Preprocess is provided by data.py.
    """
    if device is None:
        device = get_device()

    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.to(device)
    if eval_mode:
        model.eval()
    return model, tokenizer, device


@torch.no_grad()
def encode_texts(
    texts: Iterable[str],
    model,
    tokenizer,
    device: torch.device,
    batch_size: int = 256,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Encode a list of texts -> (N, D) tensor.
    """

    all_feats = []
    texts = list(texts)
    for i in range(0, len(texts), batch_size):
        toks = tokenizer(texts[i : i + batch_size])
        toks = toks.to(device)
        feats = model.encode_text(toks)
        if normalize:
            feats = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats)
    return torch.cat(all_feats, dim=0) if all_feats else torch.empty(0, device=device)


@torch.no_grad()
def encode_images_tensor(
    images: torch.Tensor,
    model,
    device: torch.device,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Encode a batch of preprocessed images (B,3,H,W) -> (B, D) tensor.
    Assumes images were transformed with OpenCLIP preprocess.
    """
    images = images.to(device)
    feats = model.encode_image(images)
    if normalize:
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


@torch.no_grad()
def encode_image_file(
    image_path: str,
    model,
    preprocess,
    device: torch.device,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Encode a single image file path using given preprocess.
    """
    img = Image.open(image_path).convert("RGB")
    img_t = preprocess(img).unsqueeze(0).to(device)
    feats = model.encode_image(img_t)
    if normalize:
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.squeeze(0)



DEFAULT_TEMPLATES = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a close-up photo of a {}.",
    "a low-resolution photo of a {}.",
    "a bright photo of a {}.",
]

def make_prompts(class_names: List[str], templates: List[str]) -> List[List[str]]:
    """
    For each class name, build its list of templated prompts.
    """
    return [[t.format(name) for t in templates] for name in class_names]


@torch.no_grad()
def build_zeroshot_classifier(
    class_names: List[str],
    model,
    tokenizer,
    device: torch.device,
    templates: Optional[List[str]] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Build zero-shot classifier weights (C, D) by encoding prompt templates and
    averaging (logit-level or feature-level). We average normalized features.
    """
    if templates is None:
        templates = DEFAULT_TEMPLATES

    per_class_prompts = make_prompts(class_names, templates)
    weights = []
    for prompts in per_class_prompts:
        txt_feats = encode_texts(prompts, model, tokenizer, device, normalize=True)
        
        txt_feat = txt_feats.mean(dim=0, keepdim=True)
        if normalize:
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        weights.append(txt_feat)
    return torch.cat(weights, dim=0)  # (C, D)


@click.group()
def cli():
    """Utilities for OpenCLIP model loading and zero-shot weights."""


@cli.command("device")
def cli_device():
    """Print resolved device."""
    dev = get_device()
    click.echo(f"Resolved device: {dev.type.upper()}")


@cli.command("load")
@click.option("--model-name", default="ViT-B-32", show_default=True)
@click.option("--pretrained", default="laion2b_s34b_b79k", show_default=True)
def cli_load(model_name, pretrained):
    """Load model/tokenizer once and print basic info."""
    model, tokenizer, device = load_openclip(model_name, pretrained)

    with torch.no_grad():
        toks = tokenizer(["a test"])
        toks = toks.to(device)
        t = model.encode_text(toks)
        dim = t.shape[-1]
    click.echo(f"Loaded {model_name} ({pretrained}) on {device.type.upper()} | embed_dim={dim}")


@cli.command("zeroshot-dump")
@click.option("--model-name", default="ViT-B-32", show_default=True)
@click.option("--pretrained", default="laion2b_s34b_b79k", show_default=True)
@click.option("--classnames", default="airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck",
              show_default=True, help="Comma-separated class names.")
@click.option("--templates-json", default="", help="Optional path to a JSON list of templates.")
@click.option("--out", default="./zeroshot_weights.pt", show_default=True,
              help="Output path to save torch tensor of shape (C,D).")
def cli_zeroshot_dump(model_name, pretrained, classnames, templates_json, out):
    """
    Build zero-shot classifier weights for given classes and save to disk.
    """
    model, tokenizer, device = load_openclip(model_name, pretrained)
    classes = [c.strip() for c in classnames.split(",") if c.strip()]

    templates = None
    if templates_json:
        with open(templates_json, "r") as f:
            templates = json.load(f)
            if not isinstance(templates, list) or not all(isinstance(x, str) for x in templates):
                raise ValueError("templates-json must contain a JSON array of strings")

    zs = build_zeroshot_classifier(classes, model, tokenizer, device, templates=templates)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save({"weights": zs.cpu(), "classes": classes, "templates": templates or DEFAULT_TEMPLATES}, out)
    click.echo(f"Saved zero-shot weights to: {out} (shape={tuple(zs.shape)})")


@cli.command("predict-one")
@click.option("--model-name", default="ViT-B-32", show_default=True)
@click.option("--pretrained", default="laion2b_s34b_b79k", show_default=True)
@click.option("--image", required=True, help="Path to an RGB image.")
@click.option("--classnames", default="airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck",
              show_default=True)
@click.option("--templates-json", default="", help="Optional JSON list of templates.")
def cli_predict_one(model_name, pretrained, image, classnames, templates_json):
    """
    Run a single zero-shot prediction on an image path against provided class names.
    (Uses model.logit_scale.exp() when available; else 100.0 scalar is typical.)
    """
    
    model, tokenizer, device = load_openclip(model_name, pretrained)
    _, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)

    classes = [c.strip() for c in classnames.split(",") if c.strip()]
    templates = None
    if templates_json:
        with open(templates_json, "r") as f:
            templates = json.load(f)

    text_w = build_zeroshot_classifier(classes, model, tokenizer, device, templates=templates)  # (C,D)
    img_feat = encode_image_file(image, model, preprocess, device)  # (D,)

    logit_scale = getattr(model, "logit_scale", None)
    if logit_scale is not None:
        scale = logit_scale.exp().item()
    else:
        scale = 100.0

    logits = scale * (img_feat @ text_w.T)                # (C,)
    probs = logits.softmax(dim=-1)
    topk = min(5, len(classes))
    vals, idxs = probs.topk(topk)
    click.echo("Top predictions:")
    for p, i in zip(vals.tolist(), idxs.tolist()):
        click.echo(f"  {classes[i]:<15}  {p:.4f}")


if __name__ == "__main__":
    cli()
