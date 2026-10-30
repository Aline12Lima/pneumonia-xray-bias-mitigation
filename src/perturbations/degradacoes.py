"""Degradações do eixo B (protocolo §8.4; F4 › Todas › Degradações do eixo B, opção C).

Onde cada degradação é aplicada:

- **fotométricas** (gama, contraste, brilho) mudam o valor de cada pixel; aplicar antes ou depois
  da redução para 320 px dá praticamente a mesma imagem. São aplicadas na imagem de **320 px**;
- **espaciais** (desfoque, ruído, resolução) mudam a relação entre vizinhos; a redução para 320 px
  suaviza a imagem. São aplicadas na **resolução original**, depois da conversão para 8 bits e
  antes da redução para 320 px.

Todas recebem e devolvem uma imagem de 8 bits (uint8, 2D). O parâmetro de cada nível é escolhido
pela ancoragem nos percentis da própria base (piloto e F8), não aqui.
"""
from __future__ import annotations

import cv2
import numpy as np

FOTOMETRICAS = ["gama", "contraste", "brilho"]
ESPACIAIS = ["desfoque", "ruido", "resolucao"]

# medida que cada degradação desloca e em que direção (o lado dos percentis usados na ancoragem)
MEDIDA_ALVO = {
    "gama": ("q_brilho", "dois lados"),
    "contraste": ("q_contraste", "baixo"),
    "brilho": ("q_brilho", "dois lados"),
    "desfoque": ("q_nitidez", "baixo"),
    "ruido": ("q_ruido", "alto"),
    "resolucao": ("q_nitidez", "baixo"),
}


def _u8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def gama(img: np.ndarray, g: float) -> np.ndarray:
    """g > 1 escurece, g < 1 clareia (0 e 255 ficam fixos)."""
    return _u8(255.0 * (img.astype(np.float32) / 255.0) ** g)


def contraste(img: np.ndarray, fator: float, centro: float | None = None) -> np.ndarray:
    """Encolhe (fator < 1) ou estica (fator > 1) os tons em torno do centro (padrão: média da imagem)."""
    x = img.astype(np.float32)
    c = float(x.mean()) if centro is None else float(centro)
    return _u8(c + fator * (x - c))


def brilho(img: np.ndarray, deslocamento: float) -> np.ndarray:
    """Soma um valor constante a todos os pixels (com corte em 0 e 255)."""
    return _u8(img.astype(np.float32) + deslocamento)


def desfoque(img: np.ndarray, sigma: float) -> np.ndarray:
    """Desfoque gaussiano com desvio sigma em pixels da imagem recebida."""
    if sigma <= 0:
        return img.copy()
    return cv2.GaussianBlur(img, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma),
                            borderType=cv2.BORDER_REFLECT)


def ruido(img: np.ndarray, desvio: float, semente: int = 0) -> np.ndarray:
    """Ruído gaussiano aditivo com desvio em tons de 8 bits. Semente fixa: a mesma imagem
    recebe sempre o mesmo ruído, para o resultado ser reprodutível."""
    if desvio <= 0:
        return img.copy()
    rng = np.random.default_rng(semente)
    return _u8(img.astype(np.float32) + rng.normal(0.0, float(desvio), img.shape).astype(np.float32))


def resolucao(img: np.ndarray, fator: float) -> np.ndarray:
    """Reduz a imagem pelo fator e volta ao tamanho original (perde detalhe fino)."""
    if fator <= 1:
        return img.copy()
    h, w = img.shape[:2]
    pequena = cv2.resize(img, (max(1, round(w / fator)), max(1, round(h / fator))), interpolation=cv2.INTER_AREA)
    return cv2.resize(pequena, (w, h), interpolation=cv2.INTER_LINEAR)


FUNCOES = {"gama": gama, "contraste": contraste, "brilho": brilho,
           "desfoque": desfoque, "ruido": ruido, "resolucao": resolucao}


def aplicar(img: np.ndarray, tipo: str, parametro: float, semente: int = 0) -> np.ndarray:
    if tipo == "ruido":
        return ruido(img, parametro, semente)
    return FUNCOES[tipo](img, parametro)


def interpolar_parametro(parametros, medianas, alvo: float) -> float:
    """Parâmetro que leva a mediana da medida ao valor-alvo, por interpolação linear na grade.
    NaN se o alvo estiver fora do que a grade alcança (registrado como "não alcançado")."""
    p = np.asarray(parametros, dtype=float)
    m = np.asarray(medianas, dtype=float)
    ordem = np.argsort(m)
    m, p = m[ordem], p[ordem]
    if not (m[0] <= alvo <= m[-1]):
        return float("nan")
    return float(np.interp(alvo, m, p))
