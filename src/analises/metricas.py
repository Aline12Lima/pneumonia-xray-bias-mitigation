"""Métricas do protocolo (§9) e bootstrap por paciente.

Discriminação: AUROC (principal) e AUPRC com a prevalência ao lado. Erros: FNR e FPR no limiar
fixado na validação, PPV como complemento. Calibração: ECE com 15 faixas de igual massa (principal),
intercepto e inclinação de calibração, Brier como complemento.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    """ECE com faixas de igual massa (igual ao usado no treino)."""
    ordem = np.argsort(p, kind="stable")
    erro = 0.0
    for g in np.array_split(ordem, bins):
        if len(g):
            erro += len(g) / len(p) * abs(y[g].mean() - p[g].mean())
    return float(erro)


def _logistica(x: np.ndarray, y: np.ndarray, offset: np.ndarray | None = None, iters: int = 50) -> np.ndarray:
    """Regressão logística por Newton (sem penalização). x já inclui a coluna de 1."""
    beta = np.zeros(x.shape[1])
    off = np.zeros(len(y)) if offset is None else offset
    for _ in range(iters):
        eta = x @ beta + off
        mu = 1 / (1 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-12, None)
        passo = np.linalg.solve(x.T @ (x * w[:, None]) + 1e-10 * np.eye(x.shape[1]), x.T @ (y - mu))
        beta += passo
        if np.max(np.abs(passo)) < 1e-10:
            break
    return beta


def calibracao(y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """(intercepto de calibração com o logit como offset, inclinação de calibração)."""
    um = np.ones((len(y), 1))
    intercepto = float(_logistica(um, y, offset=z)[0])
    inclinacao = float(_logistica(np.column_stack([um, z]), y)[1])
    return intercepto, inclinacao


def metricas(y: np.ndarray, z: np.ndarray, limiar: float, p_temp: np.ndarray | None = None) -> dict:
    """y: rótulo 0/1; z: logit corrigido (principal); limiar: em logit corrigido."""
    y = y.astype(int)
    p = 1 / (1 + np.exp(-z))
    pred = z >= limiar
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    inter, incl = calibracao(y.astype(float), z)
    out = {"n": int(len(y)), "positivos": pos, "prevalencia": pos / len(y),
           "auroc": float(roc_auc_score(y, z)), "auprc": float(average_precision_score(y, z)),
           "fnr": (pos - tp) / pos, "fpr": fp / neg, "ppv": tp / (tp + fp) if tp + fp else float("nan"),
           "ece": ece(y, p), "intercepto": inter, "inclinacao": incl, "brier": float(np.mean((p - y) ** 2)),
           "prob_media": float(p.mean())}
    if p_temp is not None and not np.isnan(p_temp).all():
        out["ece_temperatura"] = ece(y, p_temp)
    return out


def indices_bootstrap(pacientes: np.ndarray, n: int, semente: int = 2026):
    """Gera n listas de índices, reamostrando pacientes com reposição (bootstrap por paciente)."""
    rng = np.random.default_rng(semente)
    codigos, inv = np.unique(pacientes, return_inverse=True)
    ordem = np.argsort(inv, kind="stable")
    inicio = np.searchsorted(inv[ordem], np.arange(len(codigos)))
    fim = np.append(inicio[1:], len(ordem))
    for _ in range(n):
        escolhidos = rng.integers(0, len(codigos), len(codigos))
        yield np.concatenate([ordem[inicio[c]:fim[c]] for c in escolhidos])
