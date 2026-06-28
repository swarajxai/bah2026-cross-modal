"""
Pretrained backbones for feature extraction.

We use DINOv2 (ViT-B/14 by default) via timm — strong self-supervised features
that transfer very well to remote sensing.  ViT-S/14 also available for fast
experiments.  A ResNet-50 fallback is included.
"""

import torch
import torch.nn as nn
import timm


class DINOv2Backbone(nn.Module):
    """DINOv2 ViT-B/14 (or ViT-S/14) trunk.

    Returns CLS-token features.  Default input is 518x518 (DINOv2 native)
    but 224 also works (positional embeddings are interpolated).
    """

    def __init__(self, model_name: str = "vit_base_patch14_dinov2.lvd142m",
                 pretrained: bool = True, img_size: int = 518):
        super().__init__()
        # Try a few tags for compatibility across timm versions
        trunk = None
        tags = [
            model_name,
            "vit_base_patch14_dinov2.lvd142m",
            "vit_base_patch14_dinov2",
            "vit_small_patch14_dinov2.lvd142m",
            "vit_small_patch14_dinov2",
        ]
        for tag in tags:
            try:
                trunk = timm.create_model(tag, pretrained=pretrained,
                                          num_classes=0, img_size=img_size)
                print(f"[DINOv2] loaded tag: {tag}")
                break
            except Exception as e:
                print(f"[DINOv2] tag failed: {tag} ({e})")
                trunk = None
        if trunk is None:
            raise RuntimeError("Could not create DINOv2 backbone. Check internet / timm version.")
        self.trunk = trunk
        self.feat_dim = self.trunk.num_features
        self.img_size = img_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)


class ResNet50Backbone(nn.Module):
    """Lightweight fallback — ImageNet ResNet-50."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.trunk = timm.create_model("resnet50", pretrained=pretrained, num_classes=0)
        self.feat_dim = self.trunk.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)


def build_backbone(name: str = "resnet50") -> nn.Module:
    if name in ("dinov2_base", "dinov2_base_518"):
        return DINOv2Backbone(img_size=518)
    if name == "dinov2_base_224":
        return DINOv2Backbone(img_size=224)
    if name in ("dinov2_small", "dinov2_small_224"):
        return DINOv2Backbone(img_size=224)
    if name == "dinov2_small_518":
        return DINOv2Backbone(img_size=518)
    if name == "resnet50":
        return ResNet50Backbone()
    raise ValueError(f"Unknown backbone: {name}")


if __name__ == "__main__":
    m = build_backbone("dinov2_base_518")
    print("feat_dim:", m.feat_dim)
    x = torch.randn(2, 3, 518, 518)
    with torch.no_grad():
        y = m(x)
    print("out:", y.shape)