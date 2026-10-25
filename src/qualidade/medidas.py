"""As oito medidas de qualidade do protocolo (§6, passo 3.12 do plano), na resolução original.

Entrada
-------
- ``img8``: a imagem já convertida para 8 bits pela regra única (``src/imagem/conversao.py``,
  percentis 0,5–99,5, MONOCHROME1 invertido), **na resolução original**, antes do
  redimensionamento para 320 px.
- ``roi``: a máscara do pulmão (0/1) no mesmo tamanho (``src/imagem/mascara.py``), ou a
  máscara central fixa da análise de sensibilidade.
- ``bruta`` (opcional): a imagem original antes da conversão, só para a saturação nativa.

Saída
-----
Um dicionário de colunas com o prefixo ``q_``, pronto para a tabela de predições
(``configs/schema_predicoes.yaml``). Medidas que não podem ser calculadas viram NaN.

Definições operacionais
-----------------------
As definições da §6 do protocolo deixam alguns detalhes em aberto. Os valores abaixo são
a proposta registrada em ``docs/decisoes.md`` (F3 › Todas › Medidas de qualidade) e só
mudam por decisão antes do congelamento.

| Coluna            | Medida        | Definição                                                               | Papel                         |
|-------------------|---------------|-------------------------------------------------------------------------|-------------------------------|
| q_brilho          | Brilho        | Média dos valores de 8 bits na ROI (0–255)                              | Primária                      |
| q_contraste       | Contraste     | Desvio-padrão dos valores de 8 bits na ROI                              | Primária                      |
| q_saturacao       | Saturação     | Fração dos pixels da ROI em ≤ 1 ou ≥ 254                                | Primária                      |
| q_enquadramento   | Enquadramento | Fração da imagem coberta por regiões pretas (≤ 5) ligadas à borda       | Primária                      |
| q_amplitude       | Amplitude     | Percentil 95 menos percentil 5 na ROI                                   | Secundária                    |
| q_entropia        | Entropia      | Entropia de Shannon (bits) do histograma de 256 tons na ROI             | Secundária                    |
| q_nitidez         | Nitidez       | Variância do Laplaciano (3 × 3) na ROI                                  | Secundária, só dentro da base |
| q_ruido           | Ruído         | 1,4826 × MAD do resíduo passa-alta (imagem − gaussiana σ = 1,5)         | Secundária, só dentro da base |
|                   |               | nos 20% de pixels mais homogêneos da ROI (menor desvio local 7 × 7)     |                               |

Medidas auxiliares (prefixo ``q_aux_``) são descritivas e **não** entram nos testes:

- ``q_aux_area_roi``: fração da imagem coberta pela máscara (serve ao teste "máscara × doença").
- ``q_aux_roi_vazia``: 1 se a máscara tem menos de ``MIN_PIXELS_ROI`` pixels.
- ``q_aux_saturacao_baixa`` e ``q_aux_saturacao_alta``: as duas metades de ``q_saturacao``.
- ``q_aux_saturacao_nativa``: fração dos pixels da ROI no valor mínimo ou máximo da imagem
  **original** (detecta saturação do aparelho, como o teto do Imaging Dynamics no PadChest,
  que a conversão por percentis esconde).

Leitura
-------
Como a conversão estica cada imagem entre os seus percentis 0,5 e 99,5, brilho e contraste
descrevem a **apresentação** da imagem distribuída, não a exposição. Por construção, cerca de
0,5% dos pixels de cada imagem vão a 0 e 0,5% a 255; ``q_saturacao`` mede quanto disso cai
dentro do pulmão. Nitidez e ruído dependem da resolução e da compressão, por isso só são
comparados dentro de cada base (Plano › Todas › Nitidez e ruído).
"""
from __future__ import annotations

import warnings

import cv2
import numpy as np

PRIMARIAS = ["q_brilho", "q_contraste", "q_saturacao", "q_enquadramento"]
SECUNDARIAS = ["q_amplitude", "q_entropia", "q_nitidez", "q_ruido"]
SO_DENTRO_DA_BASE = ["q_nitidez", "q_ruido"]
AUXILIARES = ["q_aux_area_roi", "q_aux_roi_vazia", "q_aux_saturacao_baixa",
              "q_aux_saturacao_alta", "q_aux_saturacao_nativa"]
COLUNAS = PRIMARIAS + SECUNDARIAS + AUXILIARES

SATURACAO_BAIXA = 1          # ≤ 1 conta como preto saturado
SATURACAO_ALTA = 254         # ≥ 254 conta como branco saturado
LIMIAR_MARGEM = 5            # ≤ 5 conta como preto de margem
RUIDO_SIGMA = 1.5            # suavização gaussiana do passa-alta
RUIDO_JANELA = 7             # janela do desvio-padrão local
RUIDO_QUANTIL = 0.20         # fração mais homogênea da ROI usada no ruído
MIN_PIXELS_ROI = 1000        # abaixo disso a ROI é considerada vazia
LADO_MIN_ORIGINAL = 512      # abaixo disso avisa que a imagem não parece estar na resolução original


def margem_preta(img8: np.ndarray, limiar: int = LIMIAR_MARGEM) -> float:
    """Fração da imagem em regiões escuras (≤ limiar) conectadas à borda: fundo, ar fora do corpo, preenchimento."""
    escuro = (img8 <= limiar).astype(np.uint8)
    if not escuro.any():
        return 0.0
    n, rotulos = cv2.connectedComponents(escuro, connectivity=4)
    if n <= 1:
        return 0.0
    borda = np.unique(np.concatenate([rotulos[0, :], rotulos[-1, :], rotulos[:, 0], rotulos[:, -1]]))
    borda = borda[borda != 0]
    if borda.size == 0:
        return 0.0
    return float(np.isin(rotulos, borda).mean())


def _desvio_local(f: np.ndarray, janela: int) -> np.ndarray:
    media = cv2.blur(f, (janela, janela))
    media_q = cv2.blur(f * f, (janela, janela))
    return np.sqrt(np.maximum(media_q - media * media, 0.0))


def medidas_qualidade(img8: np.ndarray, roi: np.ndarray, bruta: np.ndarray | None = None) -> dict[str, float]:
    """Calcula as oito medidas da §6 (mais as auxiliares) para uma imagem."""
    if img8.dtype != np.uint8 or img8.ndim != 2:
        raise ValueError(f"img8 precisa ser uint8 de um canal; veio {img8.dtype} com forma {img8.shape}")
    if roi.shape != img8.shape:
        raise ValueError(f"roi {roi.shape} e img8 {img8.shape} precisam ter o mesmo tamanho")
    if bruta is not None and bruta.shape[:2] != img8.shape:
        raise ValueError(f"bruta {bruta.shape} e img8 {img8.shape} precisam ter o mesmo tamanho")
    if min(img8.shape) < LADO_MIN_ORIGINAL:
        warnings.warn(f"imagem {img8.shape}: as medidas devem ser calculadas na resolução original, "
                      "antes do redimensionamento", stacklevel=2)

    out = {c: float("nan") for c in COLUNAS}
    m = roi.astype(bool)
    n = int(m.sum())

    out["q_enquadramento"] = margem_preta(img8)        # medida da imagem inteira, não depende da ROI
    out["q_aux_area_roi"] = float(m.mean())
    out["q_aux_roi_vazia"] = float(n < MIN_PIXELS_ROI)
    if n < MIN_PIXELS_ROI:
        return out

    v = img8[m]
    vf = v.astype(np.float64)
    out["q_brilho"] = float(vf.mean())
    out["q_contraste"] = float(vf.std())

    baixa = float(np.mean(v <= SATURACAO_BAIXA))
    alta = float(np.mean(v >= SATURACAO_ALTA))
    out["q_saturacao"] = baixa + alta
    out["q_aux_saturacao_baixa"] = baixa
    out["q_aux_saturacao_alta"] = alta

    p5, p95 = np.percentile(vf, [5, 95])
    out["q_amplitude"] = float(p95 - p5)

    hist = np.bincount(v, minlength=256).astype(np.float64) / n
    hist = hist[hist > 0]
    out["q_entropia"] = float(-(hist * np.log2(hist)).sum())

    f = img8.astype(np.float32)
    lap = cv2.Laplacian(f, cv2.CV_32F, ksize=3)
    out["q_nitidez"] = float(lap[m].astype(np.float64).var())

    suave = cv2.GaussianBlur(f, (0, 0), RUIDO_SIGMA)
    residuo = f - suave
    dp = _desvio_local(suave, RUIDO_JANELA)
    limite = np.quantile(dp[m], RUIDO_QUANTIL)
    homogeneo = m & (dp <= limite)
    r = residuo[homogeneo].astype(np.float64)
    if r.size:
        out["q_ruido"] = float(1.4826 * np.median(np.abs(r - np.median(r))))

    if bruta is not None:
        b = bruta[m] if bruta.ndim == 2 else bruta[..., 0][m]
        out["q_aux_saturacao_nativa"] = float(np.mean((b == bruta.min()) | (b == bruta.max())))

    return out
