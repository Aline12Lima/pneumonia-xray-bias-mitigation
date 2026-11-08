"""Análise do eixo B: curvas, tendência e controle negativo."""
import json
import os

import numpy as np
import pandas as pd

from src.analises.eixo_b import _pesos_ece, rodar


def test_pesos_devolvem_prevalencia():
    y = np.array([1] * 100 + [0] * 400)
    w = _pesos_ece(y, 0.02)
    assert abs(np.average(y, weights=w) - 0.02) < 1e-12


def test_eixo_b_detecta_tendencia(tmp_path):
    rng = np.random.default_rng(0)
    n = 500
    y = (np.arange(n) < 100).astype(int)
    base_z = 2.0 * y - 1 + rng.normal(0, 1, n)
    linhas = []
    for s in range(2):
        ex = f"M1_s{s}"
        os.makedirs(tmp_path / ex)
        json.dump({"limiar": {"limiar_logit": 0.0}}, open(tmp_path / ex / "pos_treino.json", "w"))
        niveis = [("nenhuma", "0", 0)] + [("contraste", p, k + 1) for k, p in enumerate(["p25", "p10", "p5", "p1"])] \
            + [("ruido", p, 0) for p in ["p75", "p90", "p95"]]
        for deg, nv, sev in niveis:
            z = base_z * (1 - 0.2 * sev) + rng.normal(0, 0.3 * sev + 1e-9, n)   # contraste piora; ruído não
            linhas.append(pd.DataFrame({"id_imagem": [f"i{i}" for i in range(n)], "id_paciente": [f"p{i}" for i in range(n)],
                                        "fonte": "chexpert_plus", "modelo": "M1", "semente": s, "id_execucao": ex,
                                        "degradacao": deg, "nivel": nv, "rotulo": y, "logit": z}))
    res = rodar(pd.concat(linhas, ignore_index=True), str(tmp_path), 0.02, n_boot=40)
    c = res[res["degradacao"] == "contraste"].iloc[0]
    r = res[res["degradacao"] == "ruido"].iloc[0]
    assert c["delta_auroc_max"] < -0.05 and c["p_tendencia_auroc"] < 0.05 and c["familia"] == "primaria"
    assert abs(r["delta_auroc_max"]) < 0.02 and r["familia"] == "secundaria" and r["controle_auroc_ok"]
    assert c["p_familia2"] == c["p_tendencia_auroc"]                      # um lado só: sem multiplicar


def test_poucos_erros_nao_estimavel(tmp_path):
    rng = np.random.default_rng(1)
    n = 300
    y = (np.arange(n) < 60).astype(int)
    os.makedirs(tmp_path / "M3_s0")
    json.dump({"limiar": {"limiar_logit": -6.0}}, open(tmp_path / "M3_s0" / "pos_treino.json", "w"))
    linhas = []
    for deg, nv in [("nenhuma", "0"), ("brilho", "p75"), ("brilho", "p90"), ("brilho", "p95")]:
        z = 5.0 * y - 2 + rng.normal(0, 1, n)                              # quase nenhum FN no limiar −6
        linhas.append(pd.DataFrame({"id_imagem": [f"i{i}" for i in range(n)], "id_paciente": [f"p{i}" for i in range(n)],
                                    "fonte": "rsna", "modelo": "M3", "semente": 0, "id_execucao": "M3_s0",
                                    "degradacao": deg, "nivel": nv, "rotulo": y, "logit": z}))
    res = rodar(pd.concat(linhas, ignore_index=True), str(tmp_path), 0.2, n_boot=10)
    r = res.iloc[0]
    assert np.isnan(r["p_tendencia_fnr"]) and "fnr: poucos eventos" in r["nao_estimavel"]
