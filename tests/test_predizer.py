"""Predições do ensaio geral (passo 6.5) e regra de correção por modelo e fonte."""
import json
import math

import numpy as np
import pandas as pd
import pytest
import yaml

from src.predicoes.predizer import correcao_da_imagem, predizer
from src.treino.carregador import ler_particao
from src.treino.treinar import correcoes_m3, prevalencias, treinar
from tests.test_treinar import _cache


def test_correcao_da_imagem():
    lo = lambda p: math.log(p / (1 - p))                                     # noqa: E731
    pn, pt = prevalencias("plus")
    m1 = {"base": "plus"}
    assert correcao_da_imagem(m1, "rsna") == pytest.approx(lo(pn) - lo(pt))    # a correção é do modelo
    assert correcao_da_imagem({"base": "rsna"}, "plus") == pytest.approx(0)
    c = correcoes_m3()
    m3 = {"base": "plus+rsna", "bases": ["plus", "rsna"], "correcao_principal": "3c"}
    assert correcao_da_imagem(m3, "plus") == pytest.approx(c["3c"]["plus"])
    assert correcao_da_imagem(m3, "rsna") == pytest.approx(0)
    assert correcao_da_imagem(m3, "padchest") == pytest.approx(c["externa"])
    assert correcao_da_imagem({**m3, "correcao_principal": "3a"}, "plus") == pytest.approx(c["3a"])


def test_predizer_reproduz_a_validacao_do_treino(tmp_path):
    cfg = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))
    cfg["treino"]["lote"] = 4
    raiz = _cache(tmp_path)
    pasta = str(tmp_path / "M1")
    treinar("M1", 0, 1e-4, cfg, pasta, raiz=raiz, pesos_imagenet=False, epocas_max=2, workers=0)
    imagens = ler_particao("plus", "validacao", raiz=raiz)
    pred = predizer(pasta, imagens, workers=0, lote=4)
    orig = pd.read_csv(f"{pasta}/predicoes_validacao.csv", dtype={"id": str})
    j = pred.merge(orig, on="id", suffixes=("", "_t"))
    assert len(j) == len(orig)
    assert np.abs(j["logit_corrigido"] - j["logit_corrigido_t"]).max() < 1e-4
