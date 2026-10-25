"""Modelo, reprodutibilidade e correção de prior (protocolo §7.1 e §7.2)."""
from __future__ import annotations

import math
import os
import random

import numpy as np
import torch
import torchvision


def fixar_sementes(semente: int) -> None:
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)
    torch.cuda.manual_seed_all(semente)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def criar_densenet121(pesos_imagenet: bool = True) -> torch.nn.Module:
    pesos = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1 if pesos_imagenet else None
    m = torchvision.models.densenet121(weights=pesos)
    m.classifier = torch.nn.Linear(m.classifier.in_features, 1)
    return m


def correcao_prior(logit: np.ndarray, pi_natural: float, pi_treino: float) -> np.ndarray:
    """logit + ln(pi_nat/(1-pi_nat)) - ln(pi_treino/(1-pi_treino)).

    Desfaz o efeito da subamostragem de negativos no treino.
    """
    return logit + math.log(pi_natural / (1 - pi_natural)) - math.log(pi_treino / (1 - pi_treino))
