from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch

from .frontends import BEVPlacePPAdapter, ASVLocAdapter, load_bevplacepp_adapter, load_asvloc_adapter


class HybridAdapter:
    """ASVLoc retrieval descriptors with BEVPlace++ REM local features."""

    name = "asvloc_bevplacepp_hybrid"
    local_feature_source = "bevplacepp_rem_local_features"
    query_uses_tta = True

    def __init__(self, asvloc: ASVLocAdapter, bevplacepp: BEVPlacePPAdapter, device: torch.device) -> None:
        self.asvloc = asvloc
        self.bevplacepp = bevplacepp
        self.device = device
        # Existing evaluator calls ``adapter.model.eval()`` during feature-bank extraction.
        self.model = self.asvloc.model

    def forward_global(self, images: torch.Tensor) -> torch.Tensor:
        return self.asvloc.forward_global(images.to(self.device, non_blocking=True))

    def forward_global_tta(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device, non_blocking=True)
        rotations = [images]
        rotations.extend(torch.rot90(images, k=k, dims=(-2, -1)) for k in (1, 2, 3))
        stacked = torch.cat(rotations, dim=0)
        descriptors = self.asvloc.forward_global(stacked)
        batch_size = int(images.shape[0])
        return descriptors.view(4, batch_size, -1).permute(1, 0, 2).contiguous()

    def forward_pair_features(
        self,
        query_images: torch.Tensor,
        candidate_images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.bevplacepp.forward_pair_features(
            query_images.to(self.device, non_blocking=True),
            candidate_images.to(self.device, non_blocking=True),
        )

    def forward_local_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.bevplacepp.forward_local_features(images.to(self.device, non_blocking=True))


def load_hybrid_adapter(
    asvloc_config_path: str | Path,
    asvloc_checkpoint_path: str | Path,
    bevplacepp_checkpoint_path: str | Path,
    device: torch.device,
    asvloc_overrides: list[str] | None = None,
) -> tuple[HybridAdapter, Dict]:
    asvloc, asvloc_meta = load_asvloc_adapter(
        config_path=asvloc_config_path,
        checkpoint_path=asvloc_checkpoint_path,
        device=device,
        overrides=asvloc_overrides or [],
    )
    bevplacepp, bevplacepp_meta = load_bevplacepp_adapter(
        checkpoint_path=bevplacepp_checkpoint_path,
        device=device,
    )
    adapter = HybridAdapter(asvloc=asvloc, bevplacepp=bevplacepp, device=device)
    metadata: Dict[str, object] = {
        "model_type": "hybrid_asvloc_retrieval_bevplacepp_geometry",
        "retrieval_model": asvloc_meta,
        "geometry_model": bevplacepp_meta,
        "local_feature_source": adapter.local_feature_source,
        "query_tta_rotations_deg": [0, 90, 180, 270],
        "hybrid_note": (
            "Global retrieval descriptors are produced by ASVLoc with 4-rotation query TTA; "
            "sparse geometric verification uses BEVPlace++ REM local features and the same "
            "BEVPlace2-style RANSAC backend as the BEVPlace++ baseline."
        ),
    }
    return adapter, metadata
