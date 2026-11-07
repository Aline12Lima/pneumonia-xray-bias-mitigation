"""E0 e métricas do protocolo (§8.1, §9)."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from src.analises.e0 import papel, por_execucao, resumir
from src.analises.metricas import calibracao, indices_bootstrap, metricas


def test_calibracao_recupera_intercepto_e_inclinacao():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 2, 50000)
    y = (rng.random(50000) < 1 / (1 + np.exp(-(0.5 + 0.7 * z)))).astype(float)
    _, incl = calibracao(y, z)
    assert incl == pytest.approx(0.7, abs=0.03)
    inter, incl = calibracao((rng.random(50000) < 1 / (1 + np.exp(-(z + 0.3)))).astype(float), z)
    assert inter == pytest.approx(0.3, abs=0.05) and incl == pytest.approx(1, abs=0.05)


def test_metricas_no_limiar():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    z = np.array([3, 2, 1, -1, 0.5, -2, -3, -4], float)
    m = metricas(y, z, limiar=0.0)
    assert m["fnr"] == 0.25 and m["fpr"] == 0.25 and m["auroc"] == pytest.approx(15 / 16)


def test_bootstrap_por_paciente_mantem_imagens_juntas():
    pac = np.array(["a", "a", "b", "c", "c", "c"])
    for idx in indices_bootstrap(pac, 20, 0):
        cont = pd.Series(pac[idx]).value_counts()
        assert cont.get("a", 0) % 2 == 0 and cont.get("c", 0) % 3 == 0


def _tabela(tmp_path):
    rng = np.random.default_rng(1)
    linhas = []
    for mod, sens in (("M1", 2.0), ("M2", 2.0)):
        for s in range(2):
            ex = f"{mod}_s{s}"
            os.makedirs(tmp_path / ex)
            json.dump({"limiar": {"limiar_logit": -1.0}}, open(tmp_path / ex / "pos_treino.json", "w"))
            for fonte, n, prev in (("chexpert_plus", 300, 0.1), ("rsna", 200, 0.2)):
                y = (rng.random(n) < prev).astype(int)
                forca = sens if (mod == "M1") == (fonte == "chexpert_plus") else 1.0
                z = rng.normal(0, 1, n) + forca * y - 2
                for i in range(n):
                    linhas.append({"id_imagem": f"{fonte}{i}", "id_paciente": f"{fonte}p{i // 2}", "fonte": fonte,
                                   "modelo": mod, "semente": s, "id_execucao": ex, "rotulo": y[i], "logit": z[i],
                                   "probabilidade_temperatura": np.nan, "degradacao": "nenhuma"})
    return pd.DataFrame(linhas)


def test_e0_pares_e_gaps(tmp_path):
    tab = _tabela(tmp_path)
    tab.loc[tab["fonte"] == "rsna", "rotulo"] = tab[tab["fonte"] == "rsna"].groupby("id_imagem")["rotulo"].transform("first")
    tab.loc[tab["fonte"] == "chexpert_plus", "rotulo"] = tab[tab["fonte"] == "chexpert_plus"].groupby("id_imagem")["rotulo"].transform("first")
    por = por_execucao(tab, str(tmp_path))
    assert len(por) == 8 and set(por["papel"]) == {"interno", "externo"}
    pares, gaps = resumir(tab, por, str(tmp_path), n_boot=30)
    assert len(pares) == 4 and len(gaps) == 2
    g1 = gaps[gaps["modelo"] == "M1"].iloc[0]
    assert g1["interno"] == "chexpert_plus" and g1["externo"] == "rsna"
    assert g1["gap_auroc"] > 0                                   # o externo é pior (sinal mais fraco)
    assert g1["gap_auroc_ic_inf"] <= g1["gap_auroc"] <= g1["gap_auroc_ic_sup"]
    assert papel("M3", "rsna") == "interno" and papel("M3", "padchest") == "externo"
