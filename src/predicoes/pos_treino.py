"""Limiar e temperatura dos 15 modelos finais (F6, passos 6.3 e 6.4; protocolo §7.2).

Para cada modelo (M1, M2, M3) e semente (0 a 4), na taxa escolhida (F6 › M1/M2/M3 › Taxa), a partir de
``predicoes_validacao.csv`` (logit **com** a correção de prior principal):

- **Limiar (6.3):** o maior limiar com sensibilidade >= 0,90 na validação da fonte (M3: validação
  conjunta, Plus + RSNA). Fica salvo em unidades de logit corrigido, e é reaplicado sem mudança no
  teste interno e no externo. A temperatura não muda a ordem das imagens, então não muda o limiar.
- **Temperatura (6.4):** um escalar T > 0 que minimiza a log-verossimilhança de sigmoide(logit / T) na
  validação da fonte (M3: validação conjunta). ECE bruto (principal) e ECE após temperatura
  (secundário), ambos com 15 faixas de igual massa.

Saída: ``pos_treino.json`` na pasta de cada execução e ``/out/treinos/pos_treino_resumo.csv``.
Nada aqui lê o teste.

Uso (notebook "treino", sem GPU):
    python -m src.predicoes.pos_treino
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from src.treino.treinar import ece_igual_massa

TREINOS = "/out/treinos"
TAXA = {"M1": 1e-4, "M2": 3e-4, "M3": 1e-4}            # F6 › M1/M2/M3 › Taxa (decisoes.md)
SEMENTES = [0, 1, 2, 3, 4]
ALVO_SENSIBILIDADE = 0.90


def pasta_final(modelo: str, semente: int, treinos: str = TREINOS) -> str:
    return os.path.join(treinos, f"{modelo}_s{semente}_lr{TAXA[modelo]:g}")


def limiar_sensibilidade(y: np.ndarray, z: np.ndarray, alvo: float = ALVO_SENSIBILIDADE) -> dict:
    """Maior limiar t (em logit) com sensibilidade de (z >= t) >= alvo."""
    pos = np.sort(z[y == 1])[::-1]
    k = math.ceil(alvo * len(pos))
    t = float(pos[k - 1])
    pred = z >= t
    return {"limiar_logit": t, "limiar_prob": float(1 / (1 + math.exp(-t))),
            "sensibilidade": float(pred[y == 1].mean()), "especificidade": float((~pred[y == 0]).mean()),
            "fnr": float((~pred[y == 1]).mean()), "fpr": float(pred[y == 0].mean())}


def ajustar_temperatura(y: np.ndarray, z: np.ndarray) -> float:
    def nll(log_t: float) -> float:
        zz = z / math.exp(log_t)
        return float(np.mean(np.logaddexp(0, zz) - y * zz))   # -log-verossimilhança da sigmoide
    r = minimize_scalar(nll, bounds=(math.log(0.05), math.log(20)), method="bounded")
    return float(math.exp(r.x))


def processar(pasta: str) -> dict:
    res = json.load(open(os.path.join(pasta, "resumo.json"), encoding="utf-8"))
    pred = pd.read_csv(os.path.join(pasta, "predicoes_validacao.csv"))
    y, z = pred["rotulo"].to_numpy().astype(int), pred["logit_corrigido"].to_numpy().astype(float)
    T = ajustar_temperatura(y, z)
    sig = lambda v: 1 / (1 + np.exp(-v))                                    # noqa: E731
    lim = limiar_sensibilidade(y, z)
    out = {"modelo": res["modelo"], "semente": int(res["semente"]), "taxa": res["taxa"],
           "correcao_principal": res.get("correcao_principal", "unica"), "validacao": "conjunta" if
           len(res.get("bases", [res["base"]])) > 1 else "fonte", "imagens": int(len(y)), "positivos": int(y.sum()),
           "limiar": lim, "temperatura": T,
           "ece_bruto": ece_igual_massa(y, sig(z)), "ece_temperatura": ece_igual_massa(y, sig(z / T))}
    if "fonte" in pred.columns and pred["fonte"].nunique() > 1:              # M3: o mesmo limiar e T por fonte
        out["por_fonte"] = {}
        for f, g in pred.groupby("fonte"):
            yf, zf = g["rotulo"].to_numpy().astype(int), g["logit_corrigido"].to_numpy().astype(float)
            pf = zf >= lim["limiar_logit"]
            out["por_fonte"][f] = {"sensibilidade": float(pf[yf == 1].mean()), "fpr": float(pf[yf == 0].mean()),
                                   "ece_bruto": ece_igual_massa(yf, sig(zf)),
                                   "ece_temperatura": ece_igual_massa(yf, sig(zf / T))}
    json.dump(out, open(os.path.join(pasta, "pos_treino.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--treinos", default=TREINOS)
    a = ap.parse_args(argv)
    linhas, faltam = [], []
    for m in TAXA:
        for s in SEMENTES:
            pasta = pasta_final(m, s, a.treinos)
            if not os.path.exists(os.path.join(pasta, "resumo.json")):
                faltam.append(os.path.basename(pasta))
                continue
            r = processar(pasta)
            lim = r["limiar"]
            linha = {"modelo": m, "semente": s, "limiar_logit": lim["limiar_logit"], "limiar_prob": lim["limiar_prob"],
                     "sensibilidade": lim["sensibilidade"], "fpr": lim["fpr"], "temperatura": r["temperatura"],
                     "ece_bruto": r["ece_bruto"], "ece_temperatura": r["ece_temperatura"]}
            for f, v in r.get("por_fonte", {}).items():
                linha[f"sens_{f}"], linha[f"fpr_{f}"] = v["sensibilidade"], v["fpr"]
                linha[f"ece_bruto_{f}"], linha[f"ece_temp_{f}"] = v["ece_bruto"], v["ece_temperatura"]
            linhas.append(linha)
            print(f"{m} s{s} | limiar {lim['limiar_prob']:.4f} (logit {lim['limiar_logit']:+.3f}) | sens "
                  f"{lim['sensibilidade']:.3f} | FPR {lim['fpr']:.3f} | T {r['temperatura']:.3f} | ECE "
                  f"{r['ece_bruto']:.4f} → {r['ece_temperatura']:.4f}"
                  + "".join(f" | {f}: sens {v['sensibilidade']:.3f}, FPR {v['fpr']:.3f}, ECE {v['ece_bruto']:.4f}"
                            f" → {v['ece_temperatura']:.4f}" for f, v in r.get("por_fonte", {}).items()))
    df = pd.DataFrame(linhas)
    df.to_csv(os.path.join(a.treinos, "pos_treino_resumo.csv"), index=False)
    print(f"\n{len(linhas)} modelos processados" + (f"; faltam: {faltam}" if faltam else ""))
    print("RESULTADO:", "OK" if not faltam else "INCOMPLETO", f"(salvo em {a.treinos}/pos_treino_resumo.csv)")
    return 0 if not faltam else 1


if __name__ == "__main__":
    raise SystemExit(main())
