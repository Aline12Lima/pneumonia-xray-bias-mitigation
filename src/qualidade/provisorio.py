"""Medidas de qualidade PROVISÓRIAS, só para o teste de fumaça.

As oito medidas do protocolo (§6) são escritas no trabalho de auditoria (F3, passo 3.12)
e substituem esta função. Aqui há apenas duas medidas simples, para provar que a
coluna "q_" chega até a tabela de predições.
"""
from __future__ import annotations

import numpy as np


def medidas_provisorias(img8: np.ndarray, roi: np.ndarray) -> dict[str, float]:
    v = img8[roi.astype(bool)] if roi.any() else img8.ravel()
    v = v.astype(np.float32) / 255.0
    return {
        "q_prov_brilho": float(v.mean()),
        "q_prov_contraste": float(v.std()),
        "q_prov_area_roi": float(roi.mean()),
    }
