"""Attention map generators (GradCAM / GradCAM++ / ScoreCAM) for CNN and ViT.

Public API used by classifier.py and pipeline.py:

    get_attention_generator(method, target_layer=None, arch="cnn")
        Factory; case-insensitive method, unknown -> ScoreCAM.

    AttentionMapGenerator (base):
        .generate(model, input_tensor, target_class) -> np.ndarray (H, W) in [0, 1]
        .overlay_on_image(cam, pil_image, alpha=0.5) -> PIL.Image

    reshape_vit_tokens(tensor) -> tensor (strip CLS, reshape (B,N+1,D)->(B,D,h,w)).

    compute_attention_diff(m1, m2) -> np.ndarray (resize m1 to m2.shape, in [-1, 1])
    render_diff_heatmap(diff, pil_image) -> PIL.Image
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _resolve_layer(model: torch.nn.Module, target_layer: str | None) -> torch.nn.Module:
    if target_layer is None:
        return model
    obj = model
    for part in target_layer.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def reshape_vit_tokens(tensor: torch.Tensor) -> torch.Tensor:
    """Strip CLS token and reshape (B, N+1, D) -> (B, D, sqrt(N), sqrt(N))."""
    b, n_plus_1, d = tensor.shape
    n = n_plus_1 - 1
    side = int(math.isqrt(n)) if n > 0 else 0
    if side * side != n or n == 0:
        side = max(int(math.isqrt(n)), 1)
    patches = tensor[:, 1:1 + side * side, :]
    return patches.transpose(1, 2).reshape(b, d, side, side)


class _LayerCapture:
    """Forward + backward hooks on one layer; cleaned up on remove()."""

    def __init__(self, layer: torch.nn.Module):
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._fh = layer.register_forward_hook(self._on_forward)
        self._bh = layer.register_full_backward_hook(self._on_backward)

    def _on_forward(self, _module, _inp, output):
        self.activations = output

    def _on_backward(self, _module, _grad_in, grad_out):
        self.gradients = grad_out[0]

    def remove(self):
        self._fh.remove()
        self._bh.remove()


def _normalize(cam: torch.Tensor) -> torch.Tensor:
    cam = cam.clamp(min=0)
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max - cam_min < 1e-8:
        return torch.zeros_like(cam)
    return (cam - cam_min) / (cam_max - cam_min)


class AttentionMapGenerator:
    """Base class. Subclasses implement _compute_cam()."""

    def __init__(self, target_layer: str | None = None, arch: str = "cnn"):
        self.target_layer = target_layer
        self.arch = arch

    def _to_spatial(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 4:
            return tensor
        if tensor.ndim == 3 and self.arch == "vit":
            return reshape_vit_tokens(tensor)
        return tensor

    def generate(
        self, model: torch.nn.Module, input_tensor: torch.Tensor, target_class: int,
    ) -> np.ndarray:
        layer = _resolve_layer(model, self.target_layer)
        cam = self._compute_cam(model, layer, input_tensor, target_class)
        cam = _normalize(cam)
        return cam.detach().cpu().numpy().astype(np.float32)

    def _compute_cam(self, model, layer, input_tensor, target_class) -> torch.Tensor:
        raise NotImplementedError

    def overlay_on_image(
        self, cam: np.ndarray, image: Image.Image, alpha: float = 0.5,
    ) -> Image.Image:
        cam_pil = Image.fromarray((cam * 255).astype(np.uint8)).resize(
            image.size, Image.BILINEAR
        )
        cam_arr = np.array(cam_pil, dtype=np.float32) / 255.0
        heatmap = _apply_jet(cam_arr)
        base = np.array(image.convert("RGB"), dtype=np.float32)
        blended = (1 - alpha) * base + alpha * heatmap
        return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


class GradCAM(AttentionMapGenerator):
    """Standard Grad-CAM (Selvaraju et al., 2017)."""

    def _compute_cam(self, model, layer, input_tensor, target_class):
        capture = _LayerCapture(layer)
        try:
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                input_tensor = input_tensor.detach().requires_grad_(True)
                logits = model(input_tensor)
                logits[0, target_class].backward(retain_graph=False)
            acts = self._to_spatial(capture.activations)
            grads = self._to_spatial(capture.gradients)
            weights = grads.mean(dim=(2, 3), keepdim=True)
            return (weights * acts).sum(dim=1).squeeze(0)
        finally:
            capture.remove()


class GradCAMPlusPlus(AttentionMapGenerator):
    """Grad-CAM++ (Chattopadhay et al., 2018) with weighted gradient pooling."""

    def _compute_cam(self, model, layer, input_tensor, target_class):
        capture = _LayerCapture(layer)
        try:
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                input_tensor = input_tensor.detach().requires_grad_(True)
                logits = model(input_tensor)
                logits[0, target_class].backward(retain_graph=False)
            acts = self._to_spatial(capture.activations)
            grads = self._to_spatial(capture.gradients)
            grads2, grads3 = grads.pow(2), grads.pow(3)
            denom = 2 * grads2 + (acts * grads3).sum(dim=(2, 3), keepdim=True)
            alpha = grads2 / denom.clamp(min=1e-8)
            weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)
            return (weights * acts).sum(dim=1).squeeze(0)
        finally:
            capture.remove()


class ScoreCAM(AttentionMapGenerator):
    """Score-CAM (Wang et al., 2020) — gradient-free, uses forward-pass scores."""

    def __init__(self, target_layer=None, arch="cnn", batch_size: int = 16):
        super().__init__(target_layer=target_layer, arch=arch)
        self.batch_size = batch_size

    def _compute_cam(self, model, layer, input_tensor, target_class):
        with torch.no_grad():
            acts = self._capture_activations(model, layer, input_tensor)
        masks = self._build_masks(acts, input_tensor.shape[-2:])
        weights = self._score_masks(model, input_tensor, masks, target_class)
        return (weights.view(-1, 1, 1) * acts.squeeze(0)).sum(dim=0)

    def _capture_activations(self, model, layer, input_tensor):
        capture = _LayerCapture(layer)
        try:
            model(input_tensor)
            return self._to_spatial(capture.activations)
        finally:
            capture.remove()

    def _build_masks(self, acts: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        c = acts.shape[1]
        upsampled = F.interpolate(acts, size=hw, mode="bilinear", align_corners=False)
        masks = upsampled.squeeze(0)
        flat = masks.view(c, -1)
        mn = flat.min(dim=1, keepdim=True).values.view(c, 1, 1)
        mx = flat.max(dim=1, keepdim=True).values.view(c, 1, 1)
        return (masks - mn) / (mx - mn).clamp(min=1e-8)

    def _score_masks(self, model, input_tensor, masks, target_class):
        scores = torch.zeros(masks.shape[0], device=input_tensor.device)
        with torch.no_grad():
            for start in range(0, masks.shape[0], self.batch_size):
                chunk = masks[start:start + self.batch_size]
                masked = input_tensor * chunk.unsqueeze(1)
                logits = model(masked)
                probs = F.softmax(logits, dim=1)[:, target_class]
                scores[start:start + chunk.shape[0]] = probs
        return F.relu(scores)


_REGISTRY = {
    "gradcam": GradCAM,
    "gradcam++": GradCAMPlusPlus,
    "scorecam": ScoreCAM,
}


def get_attention_generator(
    method: str, target_layer: str | None = None, arch: str = "cnn",
) -> AttentionMapGenerator:
    """Factory: case-insensitive method name, unknown -> ScoreCAM."""
    cls = _REGISTRY.get(method.lower(), ScoreCAM)
    return cls(target_layer=target_layer, arch=arch)


def _apply_jet(gray: np.ndarray) -> np.ndarray:
    """Map [0, 1] grayscale to RGB jet colormap (no matplotlib dependency)."""
    g = gray.clip(0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * g - 3), 0, 1)
    gr = np.clip(1.5 - np.abs(4 * g - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * g - 1), 0, 1)
    return np.stack([r, gr, b], axis=-1) * 255.0


def compute_attention_diff(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """Signed diff (m2 - m1) resized to m2.shape, clipped to [-1, 1]."""
    if m1.shape != m2.shape:
        m1 = _resize_2d(m1, m2.shape)
    diff = m2.astype(np.float32) - m1.astype(np.float32)
    return diff.clip(-1.0, 1.0)


def _resize_2d(arr: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    img = Image.fromarray((arr.clip(0, 1) * 255).astype(np.uint8))
    return np.array(img.resize((hw[1], hw[0]), Image.BILINEAR), dtype=np.float32) / 255.0


def render_diff_heatmap(diff: np.ndarray, image: Image.Image) -> Image.Image:
    """Blend a red(+)/blue(-) diff overlay onto an image."""
    diff_resized = _resize_diff_to_image(diff, image.size)
    overlay = _diff_to_rgb(diff_resized)
    base = np.array(image.convert("RGB"), dtype=np.float32)
    blended = 0.5 * base + 0.5 * overlay
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


def _resize_diff_to_image(diff: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    scaled = ((diff + 1) * 127.5).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(scaled).resize(size, Image.BILINEAR)
    return np.array(pil, dtype=np.float32) / 127.5 - 1.0


def _diff_to_rgb(diff: np.ndarray) -> np.ndarray:
    pos = diff.clip(0, 1)
    neg = (-diff).clip(0, 1)
    r = pos * 255.0
    b = neg * 255.0
    g = np.zeros_like(r)
    return np.stack([r, g, b], axis=-1)
