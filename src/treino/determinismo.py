"""Determinismo do treino (F5, passo 5.6 do plano de execução).

Compara duas execuções com a mesma configuração (modelo, semente, taxa, código e máquina): o
piloto original (``/out/treinos/M1_s0_lr0.0001``) e a repetição (``..._rep1``).

Tolerância (F5 › M1 › Determinismo, fixada em 02/11, **antes** de rodar a repetição):
|ΔAUROC| < 0,01 na validação, na melhor época de cada execução. É metade da margem de relevância
do AUROC (0,02, F4 › Todas › Estatística): a variação de uma execução idêntica tem de ficar bem
abaixo de qualquer diferença que o estudo leia como relevante.

O resto é descritivo: diferença por época, se a melhor época é a mesma, maior diferença de logit
por imagem e se as predições são idênticas bit a bit. Na GPU, ``use_deterministic_algorithms`` está
com ``warn_only=True`` e há precisão mista, então pequenas diferenças são possíveis.

Saída: ``/out/determinismo/<pasta original>.json``.

Uso (notebook "treino", sem GPU):
    python -m src.treino.determinismo --original M1_s0_lr0.0001 --repeticao M1_s0_lr0.0001_rep1
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

TREINOS = "/out/treinos"
SAIDA = "/out/determinismo"
TOLERANCIA_AUROC = 0.01


def _ler(pasta: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    res = json.load(open(os.path.join(pasta, "resumo.json"), encoding="utf-8"))
    return res, pd.read_csv(os.path.join(pasta, "curvas.csv")), pd.read_csv(os.path.join(pasta, "predicoes_validacao.csv"))


def comparar(original: str, repeticao: str) -> dict:
    r1, c1, p1 = _ler(original)
    r2, c2, p2 = _ler(repeticao)
    chaves = ("modelo", "semente", "taxa", "base")
    if any(r1[k] != r2[k] for k in chaves):
        raise ValueError(f"configurações diferentes: {[(k, r1[k], r2[k]) for k in chaves if r1[k] != r2[k]]}")
    n = min(len(c1), len(c2))
    d_epoca = (c2["auroc"].iloc[:n].to_numpy() - c1["auroc"].iloc[:n].to_numpy())
    junto = p1.merge(p2, on="id", suffixes=("_1", "_2"))
    if len(junto) != len(p1) or len(junto) != len(p2):
        raise ValueError("as duas execuções não têm as mesmas imagens de validação")
    d_logit = np.abs(junto["logit_2"].to_numpy() - junto["logit_1"].to_numpy())
    a1, a2 = r1["validacao_com_correcao"]["auroc"], r2["validacao_com_correcao"]["auroc"]
    delta = a2 - a1
    return {"original": os.path.basename(original.rstrip("/")), "repeticao": os.path.basename(repeticao.rstrip("/")),
            "auroc": [a1, a2], "delta_auroc": delta,
            "auprc": [r1["validacao_com_correcao"]["auprc"], r2["validacao_com_correcao"]["auprc"]],
            "epoca_melhor": [r1["epoca_melhor"], r2["epoca_melhor"]], "mesma_epoca_melhor": r1["epoca_melhor"] == r2["epoca_melhor"],
            "epocas_rodadas": [r1["epocas_rodadas"], r2["epocas_rodadas"]],
            "delta_auroc_por_epoca": [round(float(d), 6) for d in d_epoca],
            "maior_delta_logit": float(d_logit.max()), "predicoes_identicas": bool((d_logit == 0).all()),
            "gpu": [r1["maquina"]["gpu"], r2["maquina"]["gpu"]],
            "criterio": f"|ΔAUROC na melhor época| < {TOLERANCIA_AUROC}",
            "aprovado": abs(delta) < TOLERANCIA_AUROC}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original", required=True, help="pasta em /out/treinos")
    ap.add_argument("--repeticao", required=True, help="pasta em /out/treinos")
    a = ap.parse_args(argv)
    r = comparar(os.path.join(TREINOS, a.original), os.path.join(TREINOS, a.repeticao))
    print(f"{r['original']} × {r['repeticao']} | GPU {r['gpu']}")
    print(f"AUROC {r['auroc'][0]:.4f} × {r['auroc'][1]:.4f} | ΔAUROC {r['delta_auroc']:+.4f} | "
          f"AUPRC {r['auprc'][0]:.4f} × {r['auprc'][1]:.4f}")
    print(f"melhor época {r['epoca_melhor']} | épocas rodadas {r['epocas_rodadas']} | "
          f"maior Δlogit {r['maior_delta_logit']:.6f} | predições idênticas: {r['predicoes_identicas']}")
    print("ΔAUROC por época:", " ".join(f"{d:+.4f}" for d in r["delta_auroc_por_epoca"]))
    os.makedirs(SAIDA, exist_ok=True)
    destino = os.path.join(SAIDA, f"{r['original']}.json")
    json.dump(r, open(destino, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("RESULTADO:", "APROVADO" if r["aprovado"] else "REPROVADO", f"({r['criterio']}; salvo em {destino})")
    return 0 if r["aprovado"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
