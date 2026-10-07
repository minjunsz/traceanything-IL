"""Trainable image encoder of the image-based baselines: one R3M-pretrained ResNet18 per camera.

Re-implementation of `R3MObsEncoder` (JUICER-style) from third_party/diffusion-policy-experiments, without the `r3m`
package: the R3M backbone is a torchvision ResNet18 whose weights are read from R3M's `model.pt`. Frames are encoded
independently (a flat (B*T) batch, weights shared across time) and projected to `projection_dim` per camera.

Augmentation (training only), as upstream:
    front cameras: ColorJitter + GaussianBlur + CenterCrop(240, 280) + RandomCrop(224)    (eval: CenterCrop(224))
    other cameras: ColorJitter + GaussianBlur + Resize(224)                                (eval: Resize(224))
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import v2

IMAGE_SIZE = 224
R3M_FEATURE_DIM = 512
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class R3MBackbone(nn.Module):
    """ResNet18 -> 512-d feature. Input (N, 3, H, W) in [0, 255]; /255 and ImageNet normalization happen inside."""

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, weights: Path | None = None) -> None:
        super().__init__()
        self.convnet = models.resnet18(weights=None)
        self.convnet.fc = nn.Identity()  # pyright: ignore[reportAttributeAccessIssue]
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        if weights is not None:
            self.load_r3m_weights(weights)

    def load_r3m_weights(self, path: Path) -> None:
        """Read R3M's `model.pt` ({"r3m": state dict saved from DataParallel}); the language heads are dropped."""
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("r3m", payload)
        prefix = "module.convnet."
        convnet = {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}
        assert convnet, f"{path} has no '{prefix}*' weights; is it an R3M checkpoint?"
        self.convnet.load_state_dict(convnet, strict=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convnet((x / 255.0 - self.mean) / self.std)


def _augmentation(crop: bool) -> v2.Compose:
    jitter_blur = [v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3), v2.GaussianBlur(5, sigma=(0.01, 2.0))]
    spatial = (
        [v2.CenterCrop((240, 280)), v2.RandomCrop((IMAGE_SIZE, IMAGE_SIZE))]
        if crop
        else [v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True)]
    )
    return v2.Compose([*jitter_blur, *spatial])


class MultiCameraEncoder(nn.Module):
    """Per-camera R3M ResNet18 + linear projection; parameter names (`backbones`, `projectors`) are checkpoint format."""

    def __init__(
        self,
        cameras: Sequence[str],
        front_cameras: Sequence[str] = ("scene",),
        projection_dim: int = 128,
        weights: Path | None = None,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        assert set(front_cameras) <= set(cameras), f"front cameras {front_cameras} must be among {cameras}"
        self.cameras, self.front_cameras = tuple(cameras), frozenset(front_cameras)
        self.output_dim = len(self.cameras) * projection_dim
        self.backbones = nn.ModuleDict({c: R3MBackbone(weights) for c in self.cameras})
        self.projectors = nn.ModuleDict({c: nn.Linear(R3M_FEATURE_DIM, projection_dim) for c in self.cameras})
        self.train_transforms = {front: _augmentation(crop=front) for front in (True, False)}
        self.eval_transforms = {
            True: v2.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
            False: v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        }
        if freeze:
            for module in (*self.backbones.values(), *self.projectors.values()):
                module.requires_grad_(False)

    def _encode(self, frames: torch.Tensor, camera: str) -> torch.Tensor:
        """frames (N, H, W, 3) in [0, 255] -> (N, projection_dim)."""
        front = camera in self.front_cameras
        x = frames.permute(0, 3, 1, 2).float().contiguous() / 255.0
        x = (self.train_transforms if self.training else self.eval_transforms)[front](x)
        return self.projectors[camera](self.backbones[camera](x * 255.0))

    def forward(self, frames: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """{camera: (B, T, H, W, 3)} in [0, 255] -> (B, T, n_cameras * projection_dim)."""
        first = frames[self.cameras[0]]
        batch, steps = first.shape[:2]
        feats = [self._encode(frames[c].reshape(batch * steps, *frames[c].shape[2:]), c) for c in self.cameras]
        return torch.cat(feats, dim=-1).reshape(batch, steps, -1)
