"""
Metric-learning losses for cross-modal satellite retrieval.

v2: stronger combination of losses for cross-modal alignment.
   - Symmetric InfoNCE (NT-Xent on the full similarity matrix)
   - Hard-class triplet (margin) with cosine distance
   - Class prototype loss: pull embeddings to their class centroid
   - Cross-modal alignment bonus: same-class different-modality pairs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b.t()


class TripletLoss(nn.Module):
    """Batch-hard triplet loss with cosine distance."""

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        sim = pairwise_cosine_sim(emb, emb)
        dist = 1.0 - sim
        N = emb.size(0)
        loss = torch.tensor(0.0, device=emb.device)
        n_pos = 0
        for i in range(N):
            pos_mask = (labels == labels[i]) & (torch.arange(N, device=emb.device) != i)
            neg_mask = (labels != labels[i])
            if pos_mask.any() and neg_mask.any():
                hardest_pos = dist[i][pos_mask].max()
                hardest_neg = dist[i][neg_mask].min()
                l = F.relu(hardest_pos - hardest_neg + self.margin)
                loss = loss + l
                n_pos += 1
        if n_pos == 0:
            return loss
        return loss / n_pos


class InfoNCELoss(nn.Module):
    """NT-Xent style contrastive loss (in-batch positives, symmetric)."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        sim = pairwise_cosine_sim(emb, emb) / self.temperature
        N = emb.size(0)
        mask_self = torch.eye(N, dtype=torch.bool, device=emb.device)
        sim = sim.masked_fill(mask_self, -1e9)
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~mask_self
        exp = torch.exp(sim)
        denom = exp.sum(dim=-1)
        log_prob = sim - torch.log(denom.unsqueeze(-1) + 1e-12)
        n_pos = pos_mask.sum(dim=-1).clamp(min=1)
        loss = -(log_prob * pos_mask).sum(dim=-1) / n_pos
        return loss.mean()


class CrossModalAlignmentLoss(nn.Module):
    """For every (anchor, positive) pair where they share a class but differ in
    modality, maximize cosine similarity.  Adds explicit cross-modal supervision
    that batch-level InfoNCE misses when one modality dominates a batch.
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, emb: torch.Tensor, labels: torch.Tensor, modalities: list) -> torch.Tensor:
        N = emb.size(0)
        mod_t = torch.tensor([hash(m) % (10**6) for m in modalities],
                             device=emb.device)
        # cross-modal mask: same label AND different modality
        eq_lab = labels.unsqueeze(0) == labels.unsqueeze(1)
        diff_mod = mod_t.unsqueeze(0) != mod_t.unsqueeze(1)
        cm_mask = eq_lab & diff_mod
        if not cm_mask.any():
            return torch.tensor(0.0, device=emb.device)
        # we want embeddings of same-class-different-mod pairs to be similar
        sim = pairwise_cosine_sim(emb, emb) / self.temperature
        # for each row, positives = same-class different-mod items
        pos_per_row = cm_mask.sum(dim=-1).clamp(min=1)
        log_prob = sim - torch.log(torch.exp(sim).sum(dim=-1, keepdim=True) + 1e-12)
        loss = -(log_prob * cm_mask).sum(dim=-1) / pos_per_row
        return loss.mean()


class ClassPrototypeLoss(nn.Module):
    """Pull each embedding toward its class centroid (computed in-batch).

    Helps the model cluster same-class items tightly, which directly improves
    precision@K.  Combined with Triplet/InfoNCE this gives cleaner F1 scores.
    """

    def __init__(self):
        super().__init__()

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        unique = torch.unique(labels)
        loss = torch.tensor(0.0, device=emb.device)
        n = 0
        for c in unique:
            mask = labels == c
            if mask.sum() < 2:
                continue
            cluster = emb[mask]
            centroid = cluster.mean(dim=0, keepdim=True)
            centroid = F.normalize(centroid, dim=-1)
            sim_to_centroid = (cluster * centroid).sum(dim=-1)
            # we want high similarity (close to 1), so loss = 1 - sim
            loss = loss + (1.0 - sim_to_centroid).mean()
            n += 1
        if n == 0:
            return loss
        return loss / n


class CombinedLoss(nn.Module):
    """Weighted combination of all losses for cross-modal metric learning."""

    def __init__(self, triplet_w=0.3, nce_w=0.4, cm_w=0.2, proto_w=0.1,
                 temperature=0.07, triplet_margin=0.2):
        super().__init__()
        self.triplet = TripletLoss(margin=triplet_margin)
        self.nce     = InfoNCELoss(temperature=temperature)
        self.cm      = CrossModalAlignmentLoss(temperature=temperature)
        self.proto   = ClassPrototypeLoss()
        self.tw, self.nw, self.cw, self.pw = triplet_w, nce_w, cm_w, proto_w

    def forward(self, emb, labels, modalities):
        return (
            self.tw * self.triplet(emb, labels) +
            self.nw * self.nce(emb, labels) +
            self.cw * self.cm(emb, labels, modalities) +
            self.pw * self.proto(emb, labels)
        )