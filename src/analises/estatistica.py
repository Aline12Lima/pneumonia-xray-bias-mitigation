"""Ferramentas estatísticas sem statsmodels.

O statsmodels da imagem do Redivis não importa com o pandas instalado (``deprecate_kwarg``), então
a regressão logística com erro-padrão agrupado e as correções de multiplicidade ficam aqui, em numpy.
Os resultados conferem com ``statsmodels`` (GLM Binomial, ``cov_type="cluster"``) nos testes.
"""
from __future__ import annotations

import math

import numpy as np


def logistica_agrupada(x: np.ndarray, y: np.ndarray, grupos: np.ndarray, iters: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Regressão logística (Newton) com erro-padrão robusto agrupado (sanduíche, correção de amostra
    pequena igual à do statsmodels: G/(G−1) × (N−1)/(N−K)). ``x`` já inclui a constante.
    Retorna (coeficientes, erros-padrão)."""
    n, k = x.shape
    beta = np.zeros(k)
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-(x @ beta)))
        w = np.clip(mu * (1 - mu), 1e-12, None)
        h = x.T @ (x * w[:, None])
        passo = np.linalg.solve(h, x.T @ (y - mu))
        beta += passo
        if np.max(np.abs(passo)) < 1e-10:
            break
    mu = 1 / (1 + np.exp(-(x @ beta)))
    w = mu * (1 - mu)
    pao = np.linalg.inv(x.T @ (x * w[:, None]))
    escores = x * (y - mu)[:, None]
    _, g = np.unique(grupos, return_inverse=True)
    soma = np.zeros((g.max() + 1, k))
    np.add.at(soma, g, escores)
    G = soma.shape[0]
    meio = soma.T @ soma * (G / (G - 1)) * ((n - 1) / (n - k))
    cov = pao @ meio @ pao
    return beta, np.sqrt(np.diag(cov))


def p_normal(b: float, ep: float) -> float:
    return math.erfc(abs(b / ep) / math.sqrt(2))


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    m = len(p)
    ordem = np.argsort(p)
    aj = np.empty(m)
    acum = 0.0
    for i, j in enumerate(ordem):
        acum = max(acum, min(1.0, (m - i) * p[j]))
        aj[j] = acum
    return aj


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    m = len(p)
    ordem = np.argsort(p)[::-1]
    aj = np.empty(m)
    menor = 1.0
    for i, j in enumerate(ordem):
        menor = min(menor, p[j] * m / (m - i))
        aj[j] = menor
    return aj
