"""A compact PyTorch spectral-attention model for AtomicVision."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from atomicvision.synthetic.types import CANDIDATE_DEFECTS, MaterialCase


@dataclass(frozen=True)
class DefectPrediction:
    """Thresholded prediction from DefectNetLite."""

    predicted_defects: list[str]
    predicted_concentrations: list[float]
    confidence: float


class DefectNetLite(nn.Module):
    """Small Conv1D + attention model for spectra-to-defect prediction."""

    def __init__(
        self,
        candidate_count: int = len(CANDIDATE_DEFECTS),
        hidden_size: int = 64,
        num_heads: int = 4,
        max_concentration: float = 0.25,
    ) -> None:
        super().__init__()
        self.max_concentration = max_concentration
        self.encoder = nn.Sequential(
            nn.Conv1d(3, hidden_size, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.defect_head = nn.Linear(hidden_size, candidate_count)
        self.concentration_head = nn.Linear(hidden_size, candidate_count)

    def forward(self, spectra: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return defect logits and concentration predictions.

        Args:
            spectra: Tensor shaped `[batch, 3, frequency_points]`.
        """

        if spectra.ndim != 3 or spectra.shape[1] != 3:
            raise ValueError("spectra must have shape [batch, 3, frequency_points]")

        tokens = self.encoder(spectra).transpose(1, 2)
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        pooled = self.norm(attended + tokens).mean(dim=1)
        defect_logits = self.defect_head(pooled)
        concentrations = torch.sigmoid(self.concentration_head(pooled)) * self.max_concentration
        return defect_logits, concentrations


def case_to_tensor(case: MaterialCase) -> torch.Tensor:
    """Convert a material case into model input channels."""

    pristine = torch.tensor(case.pristine_spectrum, dtype=torch.float32)
    defective = torch.tensor(case.defective_spectrum, dtype=torch.float32)
    delta = defective - pristine
    return torch.stack((pristine, defective, delta), dim=0)


def build_targets(case: MaterialCase) -> tuple[torch.Tensor, torch.Tensor]:
    """Build multi-label species and concentration targets."""

    labels = torch.zeros(len(CANDIDATE_DEFECTS), dtype=torch.float32)
    concentrations = torch.zeros(len(CANDIDATE_DEFECTS), dtype=torch.float32)
    index_by_species = {species: index for index, species in enumerate(CANDIDATE_DEFECTS)}
    for defect in case.defects:
        index = index_by_species[defect.species]
        labels[index] = 1.0
        concentrations[index] = defect.concentration
    return labels, concentrations


@torch.no_grad()
def predict_case(
    model: DefectNetLite,
    case: MaterialCase,
    threshold: float = 0.5,
) -> DefectPrediction:
    """Run a thresholded model prediction for one material case."""

    model.eval()
    logits, concentrations = model(case_to_tensor(case).unsqueeze(0))
    probabilities = torch.sigmoid(logits).squeeze(0)
    concentrations = concentrations.squeeze(0)
    selected = probabilities >= threshold
    predicted_defects = [
        species
        for species, is_selected in zip(CANDIDATE_DEFECTS, selected.tolist(), strict=True)
        if is_selected
    ]
    predicted_concentrations = [
        round(float(value), 5)
        for value, is_selected in zip(concentrations.tolist(), selected.tolist(), strict=True)
        if is_selected
    ]
    confidence = float(probabilities[selected].mean()) if bool(selected.any()) else float(probabilities.max())
    return DefectPrediction(
        predicted_defects=predicted_defects,
        predicted_concentrations=predicted_concentrations,
        confidence=round(confidence, 5),
    )

