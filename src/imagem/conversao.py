"""Conversão para 8 bits e redimensionamento — regra única nas três bases (configs/base.yaml, bloco `imagem`)."""
from __future__ import annotations

import cv2
import numpy as np


def para_8bits(px: np.ndarray, p_baixo: float = 0.5, p_alto: float = 99.5,
               monochrome1: bool = False) -> np.ndarray:
    """Mínimo-máximo por percentis: recorta em [p_baixo, p_alto], escala para 0–255.

    MONOCHROME1 (osso escuro) é invertido para que osso seja sempre claro.
    Sem equalização (CLAHE ou normalização por imagem não são usados).
    """
    px = px.astype(np.float32)
    lo, hi = np.percentile(px, [p_baixo, p_alto])
    x = np.clip((px - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    if monochrome1:
        x = 1.0 - x
    return (x * 255.0).round().astype(np.uint8)


def redimensionar(img: np.ndarray, tamanho: int = 320) -> np.ndarray:
    """Redimensiona direto para tamanho x tamanho (sem manter proporção), como no cache do PadChest."""
    if img.shape[:2] == (tamanho, tamanho):
        return img
    return cv2.resize(img, (tamanho, tamanho), interpolation=cv2.INTER_AREA)
