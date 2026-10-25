"""Carregador de imagens para treino e avaliação (configs/base.yaml, blocos `imagem` e `aumento`)."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import Dataset

MEDIA_IMAGENET = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
DP_IMAGENET = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def aumento_geometrico(x: torch.Tensor, escala=(0.9, 1.0), rotacao_graus=5.0,
                       gerador: torch.Generator | None = None) -> torch.Tensor:
    """Escala e rotação leves (sem nenhum aumento fotométrico). x: (C, H, W) em [0, 1]."""
    s = escala[0] + (escala[1] - escala[0]) * torch.rand(1, generator=gerador).item()
    a = math.radians(rotacao_graus * (2 * torch.rand(1, generator=gerador).item() - 1))
    theta = torch.tensor([[math.cos(a) / s, -math.sin(a) / s, 0.0],
                          [math.sin(a) / s, math.cos(a) / s, 0.0]], dtype=x.dtype).unsqueeze(0)
    grade = torch.nn.functional.affine_grid(theta, x.unsqueeze(0).shape, align_corners=False)
    return torch.nn.functional.grid_sample(x.unsqueeze(0), grade, padding_mode="border",
                                           align_corners=False)[0]


class ImagensRX(Dataset):
    """imagens: lista de arrays uint8 (H, W) já em 320 px; rotulos: lista de 0/1."""

    def __init__(self, imagens, rotulos, aumento: dict | None = None, semente: int = 0):
        self.imagens, self.rotulos, self.aumento = imagens, rotulos, aumento
        self.gerador = torch.Generator().manual_seed(semente)

    def __len__(self):
        return len(self.imagens)

    def __getitem__(self, i):
        x = torch.from_numpy(np.ascontiguousarray(self.imagens[i])).float().div(255.0)
        x = x.unsqueeze(0).repeat(3, 1, 1)                 # canal único replicado em 3
        if self.aumento:
            x = aumento_geometrico(x, tuple(self.aumento["escala"]),
                                   float(self.aumento["rotacao_graus"]), self.gerador)
        x = (x - MEDIA_IMAGENET) / DP_IMAGENET
        return x, torch.tensor([float(self.rotulos[i])])
