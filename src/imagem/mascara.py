"""Máscara do pulmão — mesmo procedimento das auditorias (lungs-segmentation, resnet34).

Procedimento: imagem 8 bits → 3 canais → mínimo-máximo → 512 x 512 → U-Net resnet34
→ sigmoide > 0,2 → maior componente conectado por pulmão → união dos dois pulmões.

Instalação (o pacote fixa versões antigas de torch; por isso sem dependências):
    pip install --no-deps lungs-segmentation==1.1.2
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

_MODELO = None


def carregar_modelo(dispositivo: str = "cpu"):
    global _MODELO
    if _MODELO is None:
        from lungs_segmentation.pre_trained_models import create_model
        _MODELO = create_model("resnet34").eval().to(dispositivo)
    return _MODELO


def _maior_componente(m: np.ndarray) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 1:
        return m.astype(np.uint8)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == k).astype(np.uint8)


def segmentar(img8: np.ndarray, limiar: float = 0.2, dispositivo: str = "cpu") -> np.ndarray:
    """Devolve a máscara (0/1) no tamanho de img8."""
    modelo = carregar_modelo(dispositivo)
    x = np.repeat(img8[..., None], 3, axis=2).astype(np.float32)
    x = (x - x.min()) / max(float(x.max() - x.min()), 1e-6)
    x = cv2.resize(x, (512, 512), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(dispositivo)
    with torch.no_grad():
        p = torch.sigmoid(modelo(t))[0].cpu().numpy()
    pulmoes = np.stack([_maior_componente(m) for m in (p > limiar)])
    roi = pulmoes.max(axis=0)
    return cv2.resize(roi, (img8.shape[1], img8.shape[0]), interpolation=cv2.INTER_NEAREST)


def mascara_central(forma: tuple[int, int], margem: float = 0.15) -> np.ndarray:
    """Máscara central fixa (análise de sensibilidade): retângulo sem as margens."""
    h, w = forma
    m = np.zeros((h, w), dtype=np.uint8)
    m[int(h * margem):int(h * (1 - margem)), int(w * margem):int(w * (1 - margem))] = 1
    return m
