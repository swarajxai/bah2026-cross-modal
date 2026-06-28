"""
Modality-specific projection heads for the shared embedding space.

v2: stronger projector architecture for the GPU-trained model.
   - BatchNorm normalizes backbone features per-modality before projection
   - 3-layer MLP with residual + GELU + dropout
   - Optional class-conditional LayerNorm for richer per-class separation

v1 (ModalityProjectorV1): the original simple 2-layer MLP, kept for
   backward-compatibility with the CPU-trained ResNet-50 checkpoint.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityProjectorV1(nn.Module):
    """Original CPU projector — 2-layer MLP, no BN, no class-conditional norm.

    State dict keys: net.0.weight, net.0.bias, net.3.weight, net.3.bias
    """
    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, class_id=None) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1)


class ModalityProjector(nn.Module):
    """Deep MLP projector with L2-normalized output (cosine similarity).

    v2 architecture — used for Colab-trained DINOv2-Base checkpoints.
    Args:
        in_dim: input feature dim (e.g. 768 for DINOv2-Base)
        hidden_dim: hidden layer width
        out_dim: output embedding dim (256)
        n_classes: number of semantic classes (for class-conditional norm)
        dropout: dropout rate
    """

    def __init__(self, in_dim: int, hidden_dim: int = 1024, out_dim: int = 256,
                 n_classes: Optional[int] = None, dropout: float = 0.2):
        super().__init__()
        # Normalise n_classes — treat 0 or None as "no class-conditional norm"
        nc = n_classes if (n_classes is not None and n_classes > 0) else None
        self.in_norm = nn.BatchNorm1d(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        # residual: hidden_dim -> hidden_dim
        self.res = nn.Linear(hidden_dim, hidden_dim)
        # Optional class-conditional layer norm
        self.cls_gamma = None
        self.cls_beta = None
        if nc is not None:
            self.cls_gamma = nn.Embedding(nc, out_dim)
            self.cls_beta  = nn.Embedding(nc, out_dim)
            nn.init.zeros_(self.cls_gamma.weight)
            nn.init.zeros_(self.cls_beta.weight)

    def forward(self, x: torch.Tensor, class_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        # BatchNorm needs (B, F)
        if x.size(0) > 1:
            x = self.in_norm(x)
        else:
            # single-sample BN fallback
            x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-5)
        h = self.act(self.fc1(x))
        h = self.drop(h)
        # residual block
        h2 = self.act(self.fc2(h))
        h2 = self.drop(h2)
        h2 = h2 + self.res(h)   # residual addition
        h2 = self.act(h2)
        z = self.fc3(h2)
        # optional class-conditional affine
        if self.cls_gamma is not None and class_id is not None:
            gamma = self.cls_gamma(class_id)
            beta = self.cls_beta(class_id)
            z = z * (1.0 + gamma) + beta
        return F.normalize(z, dim=-1)


class ModalityProjectorV6(nn.Module):
    """DINOv2-era projector: 2-block residual MLP, 1024 hidden.

    State dict keys:
      in_proj.weight/bias, in_norm.weight/bias
      block1.0.weight/bias, block1.3.weight/bias
      norm1.weight/bias
      block2.0.weight/bias, block2.3.weight/bias
      norm2.weight/bias
      out.weight/bias
    """

    def __init__(self, in_dim: int = 768, hidden: int = 1024, out_dim: int = 256,
                 dropout: float = 0.15):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        self.in_norm = nn.LayerNorm(hidden)
        self.block1 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.norm1 = nn.LayerNorm(hidden)
        self.block2 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.norm2 = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, out_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, class_id=None) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = self.in_norm(self.act(self.in_proj(x)))
        h = self.norm1(h + self.block1(h))
        h = self.norm2(h + self.block2(h))
        z = self.out(h)
        return F.normalize(z, dim=-1)


class MultiModalEncoder(nn.Module):
    """One shared backbone + per-modality projector (v2)."""

    def __init__(self, backbone: nn.Module, n_classes: int = 14,
                 out_dim: int = 256, hidden_dim: int = 1024):
        super().__init__()
        self.backbone = backbone
        in_dim = backbone.feat_dim
        self.heads: Dict[str, nn.Module] = nn.ModuleDict({
            "ms":      ModalityProjector(in_dim, hidden_dim, out_dim, n_classes=n_classes, dropout=0.2),
            "optical": ModalityProjector(in_dim, hidden_dim, out_dim, n_classes=n_classes, dropout=0.2),
            "sar":     ModalityProjector(in_dim, hidden_dim, out_dim, n_classes=n_classes, dropout=0.2),
        })
        self.out_dim = out_dim
        self.n_classes = n_classes

    def encode(self, x: torch.Tensor, modality: str,
               class_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat = self.backbone(x)
        return self.heads[modality](feat, class_id)

    def forward(self, batch: Dict[str, torch.Tensor]):
        images = batch["image"]
        modalities = batch["modality"]
        labels = batch.get("label", None)
        outs = []
        for i in range(images.size(0)):
            cid = labels[i:i+1] if labels is not None else None
            outs.append(self.encode(images[i:i+1], modalities[i], cid))
        return torch.cat(outs, dim=0)