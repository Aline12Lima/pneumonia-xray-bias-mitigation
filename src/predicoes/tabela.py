"""Tabela de predições da validação no formato do schema (F5, passo 5.7; protocolo §7.4).

Junta, para uma execução de treino (``/out/treinos/<pasta>``):

- ``predicoes_validacao.csv`` (logit com a correção de prior e probabilidade, por imagem);
- ``resumo.json`` (modelo, semente, base);
- as covariáveis da base (``/out/dados/<base>/covariaveis.csv``, vindas do ``cache_treino``).

Valida contra ``configs/schema_predicoes.yaml`` (``src/predicoes/schema.py``) e grava
``tabela_predicoes_validacao.csv`` na pasta da execução. Os IDs ficam no Redivis, como as listas;
no repositório entra só o resumo da validação.

``logit`` = logit **com** a correção de prior; ``probabilidade`` = sigmoide desse logit;
``probabilidade_temperatura`` fica vazia até a F6 (passo 6.4).

Uso (notebook "treino", sem GPU):
    python -m src.predicoes.tabela --execucao M1_s0_lr0.0001
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from src.predicoes.schema import carregar_schema, validar

TREINOS = "/out/treinos"
DADOS = "/out/dados"
FONTE = {"plus": "chexpert_plus", "rsna": "rsna"}
NOME = "tabela_predicoes_validacao.csv"


def montar(pasta: str, dados: str = DADOS) -> pd.DataFrame:
    res = json.load(open(os.path.join(pasta, "resumo.json"), encoding="utf-8"))
    if res.get("rotulos_embaralhados"):
        raise ValueError("execução com rótulos embaralhados (sanidade): não entra na tabela de predições")
    bases = res.get("bases", [res["base"]])                  # M3: Plus e RSNA
    pred = pd.read_csv(os.path.join(pasta, "predicoes_validacao.csv"), dtype={"id": str, "paciente": str})
    if "fonte" not in pred.columns:                         # execuções anteriores ao M3 (uma base só)
        pred["fonte"] = bases[0]
    covs = []
    for b in bases:
        cov_arq = os.path.join(dados, b, "covariaveis.csv")
        if not os.path.exists(cov_arq):
            raise FileNotFoundError(f"{cov_arq} não existe: rode src.cache.covariaveis no notebook da base e "
                                    "baixe de novo o cache (src.treino.baixar_cache)")
        covs.append(pd.read_csv(cov_arq, dtype={"id": str, "id_estudo": str, "fabricante": "object"}).assign(fonte=b))
    cov = pd.concat(covs, ignore_index=True)
    df = pred.merge(cov, on=["id", "fonte"], how="left")
    if df["posicao"].isna().any():
        raise RuntimeError(f"{int(df['posicao'].isna().sum())} imagens da validação sem covariáveis")
    medidas = [c for c in cov.columns if c.startswith("q_")]
    t = pd.DataFrame({
        "id_imagem": df["id"].astype(str), "id_paciente": df["paciente"].astype(str),
        "id_estudo": df["id_estudo"].astype(str), "fonte": df["fonte"].map(FONTE), "particao": "validacao",
        "modelo": res["modelo"], "semente": int(res["semente"]), "id_execucao": os.path.basename(pasta.rstrip("/")),
        "degradacao": "nenhuma", "nivel": "0", "rotulo": df["rotulo"].astype(int),
        "logit": df["logit_corrigido"].astype(float), "probabilidade": df["prob"].astype(float),
        "probabilidade_temperatura": np.nan, "posicao": df["posicao"].astype(str),
        "idade": pd.to_numeric(df["idade"]).round().astype("Int64"), "sexo": df["sexo"].astype(str),
        "fabricante": df["fabricante"].astype("object").where(df["fabricante"].notna(), None),
    })
    for c in medidas:
        t[c] = df[c].astype(float)
    return t[list(carregar_schema()["colunas"]) + medidas]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--execucao", required=True, help="pasta em /out/treinos")
    a = ap.parse_args(argv)
    pasta = os.path.join(TREINOS, a.execucao)
    t = montar(pasta)
    erros = validar(t)
    destino = os.path.join(pasta, NOME)
    t.to_csv(destino, index=False)
    print(f"{a.execucao}: {len(t)} linhas, {t['id_paciente'].nunique()} pacientes, {int(t['rotulo'].sum())} positivos | "
          f"{len(t.columns)} colunas ({sum(c.startswith('q_') for c in t.columns)} medidas de qualidade)")
    print(f"  posição {t['posicao'].value_counts().to_dict()} | sexo {t['sexo'].value_counts().to_dict()} | "
          f"idade {int(t['idade'].min())}–{int(t['idade'].max())} | fabricante "
          f"{t['fabricante'].value_counts(dropna=False).to_dict()}")
    print(f"  probabilidade média {t['probabilidade'].mean():.4f} | prevalência {t['rotulo'].mean():.4f}")
    if erros:
        print("Erros do schema:\n- " + "\n- ".join(erros))
    print("RESULTADO:", "VÁLIDA" if not erros else "INVÁLIDA", f"(salva em {destino})")
    return 0 if not erros else 1


if __name__ == "__main__":
    raise SystemExit(main())
