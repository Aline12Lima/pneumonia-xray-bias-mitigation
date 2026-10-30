"""Carregador do cache de 320 px para treino e validação (F5, passo 5.2; protocolo §7.2).

- Lê as listas congeladas (``/out/<base>/listas/``) e o cache de 320 px em 8 bits.
- Rótulo principal: positivo = 1, negativo = 0; incertos ficam fora (F3 › Plus › Rótulo).
- Canal único replicado em 3, normalização do ImageNet.
- Só aumento geométrico (escala 0,9 a 1,0; rotação até 5°), e só no treino.
- O aumento de cada imagem depende só de (semente, época, índice): o resultado é o mesmo com
  qualquer número de workers do DataLoader (determinismo, passo 5.6).
- Antes de usar, confere o SHA-256 de cada arquivo contra ``/out/<base>/cache_hashes.csv``
  (passo 5.1): se uma imagem mudou, o carregador para.

**Teste:** este módulo não lê as listas de teste, de propósito (trava do protocolo v1.1, §5).
O teste só é carregado depois de 21/11, por outro módulo, na F6.
"""
from __future__ import annotations

import hashlib
import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.treino.dados import DP_IMAGENET, MEDIA_IMAGENET, aumento_geometrico

PARTICOES = {
    "plus": {"treino": "/out/plus/listas/treino_subamostra.csv", "validacao": "/out/plus/listas/validacao.csv",
             "cache": "/out/plus/passada/c320_8bits", "hashes": "/out/plus/cache_hashes.csv"},
    "rsna": {"treino": "/out/rsna/listas/treino.csv", "validacao": "/out/rsna/listas/validacao.csv",
             "cache": "/out/rsna/passada/c320_8bits", "hashes": "/out/rsna/cache_hashes.csv"},
}
ROTULO_PRINCIPAL = {"positivo": 1, "negativo": 0}       # "incerto" fica fora


def nome_cache(ident: str) -> str:
    return str(ident).replace("/", "__").removesuffix(".dcm") + ".png"


def ler_particao(base: str, particao: str, verificar_hashes: bool = True, raiz: str | None = None) -> pd.DataFrame:
    """Tabela com id, paciente, rotulo (0/1), caminho e fonte. Só treino e validação."""
    if particao not in ("treino", "validacao"):
        raise ValueError("o carregador de treino só lê treino e validação; o teste está trancado até 21/11")
    cfg = PARTICOES[base]
    lista, cache, hashes = cfg[particao], cfg["cache"], cfg["hashes"]
    if raiz:                                                # testes: troca o "/out" por uma pasta temporária
        lista, cache, hashes = (p.replace("/out", raiz, 1) for p in (lista, cache, hashes))
    df = pd.read_csv(lista)
    col_pac = [c for c in df.columns if c not in ("id", "rotulo")][0]
    df = df[df["rotulo"].isin(ROTULO_PRINCIPAL)].copy()
    df["rotulo"] = df["rotulo"].map(ROTULO_PRINCIPAL).astype(int)
    df = df.rename(columns={col_pac: "paciente"})
    df["id"] = df["id"].astype(str)
    df["caminho"] = [os.path.join(cache, nome_cache(i)) for i in df["id"]]
    df["fonte"] = base
    if verificar_hashes:
        conferir_hashes(df, hashes)
    return df[["id", "paciente", "rotulo", "caminho", "fonte"]].reset_index(drop=True)


def conferir_hashes(df: pd.DataFrame, tabela_hashes: str) -> None:
    ref = pd.read_csv(tabela_hashes, usecols=["id", "sha256"]).astype(str).drop_duplicates("id").set_index("id")["sha256"]
    faltam = [i for i in df["id"] if i not in ref.index]
    if faltam:
        raise RuntimeError(f"{len(faltam)} imagens sem hash registrado (ex.: {faltam[0]}); rode o passo 5.1")
    ruins = []
    for ident, caminho in zip(df["id"], df["caminho"]):
        with open(caminho, "rb") as f:
            if hashlib.sha256(f.read()).hexdigest() != ref[ident]:
                ruins.append(ident)
    if ruins:
        raise RuntimeError(f"{len(ruins)} arquivos do cache mudaram desde o passo 5.1 (ex.: {ruins[0]})")


class ImagensCache(Dataset):
    """Lê os PNG do cache sob demanda. ``aumento`` = bloco ``aumento`` do configs/base.yaml, ou None."""

    def __init__(self, tabela: pd.DataFrame, aumento: dict | None = None, semente: int = 0):
        self.caminhos = tabela["caminho"].tolist()
        self.rotulos = tabela["rotulo"].astype(float).tolist()
        self.aumento, self.semente, self.epoca = aumento, semente, 0

    def definir_epoca(self, epoca: int) -> None:
        self.epoca = epoca

    def __len__(self):
        return len(self.caminhos)

    def __getitem__(self, i):
        img = cv2.imread(self.caminhos[i], cv2.IMREAD_GRAYSCALE)
        if img is None or img.shape != (320, 320):
            raise RuntimeError(f"imagem inválida no cache: {self.caminhos[i]}")
        x = torch.from_numpy(np.ascontiguousarray(img)).float().div(255.0).unsqueeze(0).repeat(3, 1, 1)
        if self.aumento:
            g = torch.Generator().manual_seed((self.semente * 1_000_003 + self.epoca) * 10_000_019 + i)
            x = aumento_geometrico(x, tuple(self.aumento["escala"]), float(self.aumento["rotacao_graus"]), g)
        x = (x - MEDIA_IMAGENET) / DP_IMAGENET
        return x, torch.tensor([self.rotulos[i]])


def resumo(tabela: pd.DataFrame) -> dict:
    return {"imagens": int(len(tabela)), "pacientes": int(tabela["paciente"].nunique()),
            "positivos": int(tabela["rotulo"].sum()), "prevalencia": round(float(tabela["rotulo"].mean()), 4)}
