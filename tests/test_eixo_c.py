"""Eixo C (§8.5): pesos dentro da classe, D_bruto, D_S1, D_S2 e parcelas."""
import json
import os

import numpy as np
import pandas as pd

from src.analises.eixo_c import ece_ponderado, pesos, rodar
from src.analises.t1_t2 import COMPARAVEIS
from src.qualidade.medidas import PRIMARIAS, SECUNDARIAS


def _base(fonte, n, desloc_q, rng):
    d = pd.DataFrame({"id_imagem": [f"{fonte}{i}" for i in range(n)], "id_paciente": [f"{fonte}p{i}" for i in range(n)],
                      "fonte": fonte, "rotulo": (rng.random(n) < 0.3).astype(int),
                      "posicao": rng.choice(["AP", "PA"], n), "sexo": rng.choice(["M", "F"], n),
                      "idade": rng.integers(18, 90, n), "degradacao": "nenhuma"})
    for m in PRIMARIAS + SECUNDARIAS:
        d[m] = rng.normal(0, 1, n)
    d["q_contraste"] += desloc_q                                     # o externo tem contraste diferente
    return d


def test_eixo_c_qualidade_explica_o_gap(tmp_path):
    rng = np.random.default_rng(0)
    src, ext = _base("chexpert_plus", 1500, 0.0, rng), _base("rsna", 1500, 1.5, rng)
    linhas = []
    for s in range(2):
        ex = f"M1_s{s}"
        os.makedirs(tmp_path / ex)
        json.dump({"limiar": {"limiar_logit": 0.0}}, open(tmp_path / ex / "pos_treino.json", "w"))
        for d in (src, ext):
            ruido = 0.5 + 1.0 * np.clip(d["q_contraste"], 0, None)    # contraste alto → modelo pior
            z = 2.0 * d["rotulo"] - 1 + rng.normal(0, 1, len(d)) * ruido
            linhas.append(d.assign(id_execucao=ex, modelo="M1", semente=s, logit=z))
    tab = pd.concat(linhas, ignore_index=True)
    res = rodar(tab, str(tmp_path), n_boot=30)
    au = res[res["metrica"] == "auroc"].iloc[0]
    assert au["externo"] == "rsna" and au["d_bruto"] > 0.05
    assert abs(au["d_clinica"]) < 0.03 and au["d_qualidade"] > 0.03            # a queda vem da qualidade
    assert au["s2_smd_max"] < au["smd_max_antes"] and au["s2_smd_max"] < 0.3   # deslocamento de 1,5 DP; truncar no p99 limita o equilíbrio
    assert 0 <= au["p_ajustado"] <= 1 and au["bootstrap_validos"] == 30


def test_pesos_dentro_da_classe_preservam_prevalencia():
    rng = np.random.default_rng(1)
    ref, ext = _base("a", 800, 0, rng), _base("b", 800, 1.0, rng)
    ext = ext.assign(pa=0.0, masculino=0.0)
    ref = ref.assign(pa=0.0, masculino=0.0)
    w, d = pesos(ref, ext, COMPARAVEIS)
    y = ext["rotulo"].to_numpy()
    assert np.average(y, weights=w) == np.mean(y) or abs(np.average(y, weights=w) - np.mean(y)) < 1e-9
    assert d["tamanho_efetivo"] < len(ext)


def test_ece_ponderado_igual_ao_simples_com_pesos_1():
    from src.analises.metricas import ece
    rng = np.random.default_rng(2)
    p = rng.random(3000)
    y = (rng.random(3000) < p).astype(float)
    assert abs(ece_ponderado(y, p, np.ones(3000)) - ece(y, p)) < 1e-9


def test_smd_dentro_da_classe_e_suporte_comum():
    from src.analises.eixo_c import smd_maxima, suporte_comum
    rng = np.random.default_rng(3)
    ref, ext = _base("a", 1000, 0, rng), _base("b", 1000, 0, rng)
    ref["rotulo"] = (rng.random(1000) < 0.02).astype(int)                # prevalências diferentes
    ref["x"] = rng.normal(0, 1, 1000) + 2 * ref["rotulo"]
    ext["x"] = rng.normal(0, 1, 1000) + 2 * ext["rotulo"]               # mesma distribuição dentro da classe
    assert smd_maxima(ref, ext, ["x"], np.ones(1000)) < 0.25             # misturando classes daria ~0,7
    ext["q_contraste"] += 5                                              # quase sem sobreposição
    mr, me = suporte_comum(ref, ext, ["q_contraste"])
    assert me.mean() < 0.2 and mr.mean() < 0.2
