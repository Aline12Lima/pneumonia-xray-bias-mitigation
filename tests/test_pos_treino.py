"""Limiar por sensibilidade e temperatura (F6, passos 6.3 e 6.4)."""
import json

import numpy as np
import pandas as pd
import pytest

from src.predicoes.pos_treino import ajustar_temperatura, limiar_sensibilidade, processar


def test_limiar_maior_com_sensibilidade_090():
    y = np.array([1] * 10 + [0] * 10)
    z = np.concatenate([np.linspace(5, -4, 10), np.linspace(-5, 2, 10)])
    r = limiar_sensibilidade(y, z)
    assert r["sensibilidade"] >= 0.9
    assert ((z >= r["limiar_logit"] + 1e-9)[y == 1]).mean() < 0.9               # qualquer limiar acima perde a meta


def test_temperatura_recupera_escala():
    rng = np.random.default_rng(0)
    zv = rng.normal(0, 2, 20000)
    y = (rng.random(20000) < 1 / (1 + np.exp(-zv))).astype(int)
    assert ajustar_temperatura(y, zv * 3) == pytest.approx(3, rel=0.08)          # logits 3 vezes confiantes demais


def test_processar_m3_por_fonte(tmp_path):
    rng = np.random.default_rng(1)
    n = 400
    f = np.array(["plus"] * 300 + ["rsna"] * 100)
    y = (rng.random(n) < np.where(f == "plus", 0.05, 0.2)).astype(int)
    z = rng.normal(0, 1, n) + 2 * y - 2
    pd.DataFrame({"id": range(n), "fonte": f, "rotulo": y, "logit_corrigido": z}).to_csv(
        tmp_path / "predicoes_validacao.csv", index=False)
    json.dump({"modelo": "M3", "semente": 0, "taxa": 1e-4, "base": "plus+rsna", "bases": ["plus", "rsna"],
               "correcao_principal": "3c"}, open(tmp_path / "resumo.json", "w"))
    r = processar(str(tmp_path))
    assert r["validacao"] == "conjunta" and set(r["por_fonte"]) == {"plus", "rsna"}
    assert r["limiar"]["sensibilidade"] >= 0.9 and r["temperatura"] > 0
    assert (tmp_path / "pos_treino.json").exists()


def test_m3_limiar_por_fonte(tmp_path):
    test_processar_m3_por_fonte(tmp_path)
    r = json.load(open(tmp_path / "pos_treino.json"))
    for f in ("plus", "rsna"):
        assert r["limiar_por_fonte"][f]["sensibilidade"] >= 0.9
