"""Determinismo (passo 5.6): comparação de duas execuções e repetição idêntica na CPU."""
import json

import pandas as pd
import pytest
import yaml

from src.treino.determinismo import comparar
from src.treino.treinar import main, treinar
from tests.test_treinar import _cache


def test_repeticao_na_cpu_da_predicoes_identicas(tmp_path):
    cfg = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))
    cfg["treino"]["lote"] = 4
    raiz = _cache(tmp_path)
    for nome in ("orig", "rep"):
        treinar("M1", 0, 1e-4, cfg, str(tmp_path / nome), raiz=raiz, pesos_imagenet=False, epocas_max=2, workers=0)
    r = comparar(str(tmp_path / "orig"), str(tmp_path / "rep"))
    assert r["predicoes_identicas"] and r["delta_auroc"] == 0 and r["aprovado"]


def _falso(pasta, auroc, semente=0, logit=0.0):
    pasta.mkdir()
    json.dump({"modelo": "M1", "semente": semente, "taxa": 1e-4, "base": "plus", "epoca_melhor": 1, "epocas_rodadas": 2,
               "validacao_com_correcao": {"auroc": auroc, "auprc": 0.1}, "maquina": {"gpu": "L4"}},
              open(pasta / "resumo.json", "w"))
    pd.DataFrame({"epoca": [0, 1], "auroc": [0.7, auroc]}).to_csv(pasta / "curvas.csv", index=False)
    pd.DataFrame({"id": ["a", "b"], "logit": [0.1, -0.2 + logit]}).to_csv(pasta / "predicoes_validacao.csv", index=False)


def test_tolerancia_e_configuracao(tmp_path):
    _falso(tmp_path / "a", 0.780)
    _falso(tmp_path / "b", 0.786, logit=0.05)
    r = comparar(str(tmp_path / "a"), str(tmp_path / "b"))
    assert r["aprovado"] and not r["predicoes_identicas"] and r["maior_delta_logit"] == pytest.approx(0.05)
    _falso(tmp_path / "c", 0.795)
    assert not comparar(str(tmp_path / "a"), str(tmp_path / "c"))["aprovado"]      # ΔAUROC 0,015
    _falso(tmp_path / "d", 0.780, semente=1)
    with pytest.raises(ValueError):
        comparar(str(tmp_path / "a"), str(tmp_path / "d"))


def test_repeticao_vai_para_outra_pasta(monkeypatch):
    pastas = []
    monkeypatch.setattr("src.treino.treinar.treinar", lambda *a, **k: pastas.append(a[4]))
    main(["--modelo", "M1", "--semente", "0", "--taxa", "1e-4", "--repeticao", "1"])
    main(["--modelo", "M1", "--semente", "0", "--taxa", "1e-4"])
    assert pastas == ["/out/treinos/M1_s0_lr0.0001_rep1", "/out/treinos/M1_s0_lr0.0001"]
