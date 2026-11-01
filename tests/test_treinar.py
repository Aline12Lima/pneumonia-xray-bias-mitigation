"""Treino de ponta a ponta num cache sintético (CPU, sem pesos ImageNet, 2 épocas)."""
import hashlib
import json
import os

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from src.treino.carregador import nome_cache
from src.treino.treinar import ece_igual_massa, prevalencias, treinar


def test_prevalencias_do_plus_e_do_rsna():
    pi_nat, pi_tr = prevalencias("plus")
    assert pi_nat == pytest.approx(2420 / (123055 - 10678)) and pi_tr == pytest.approx(0.2)
    pi_nat, pi_tr = prevalencias("rsna")
    assert pi_nat == pi_tr                                   # RSNA não é subamostrado: correção zero


def test_ece_igual_massa():
    y = np.array([0, 0, 1, 1] * 25, dtype=float)
    assert ece_igual_massa(y, np.full(100, 0.5), 5) == pytest.approx(0.0)
    assert ece_igual_massa(y, y.copy(), 5) < 0.2


def _cache(tmp_path):
    raiz = tmp_path / "dados"
    cache = raiz / "plus" / "passada" / "c320_8bits"
    listas = raiz / "plus" / "listas"
    cache.mkdir(parents=True); listas.mkdir(parents=True)
    rng = np.random.default_rng(0)
    linhas, hashes = [], []
    for k in range(20):
        pos = k % 2 == 0
        img = rng.integers(60, 120, (320, 320)).astype(np.uint8)
        if pos:
            img[100:220, 100:220] = 230                      # "opacidade" sintética
        ident = f"patient{k}/study1/view1_frontal.dcm"
        cv2.imwrite(str(cache / nome_cache(ident)), img)
        linhas.append({"id": ident, "deid_patient_id": f"p{k}", "rotulo": "positivo" if pos else "negativo"})
        hashes.append({"id": ident, "sha256": hashlib.sha256((cache / nome_cache(ident)).read_bytes()).hexdigest()})
    pd.DataFrame(linhas[:12]).to_csv(listas / "treino_subamostra.csv", index=False)
    pd.DataFrame(linhas[12:]).to_csv(listas / "validacao.csv", index=False)
    pd.DataFrame(hashes).to_csv(raiz / "plus" / "cache_hashes.csv", index=False)
    return str(raiz)


def test_treino_de_ponta_a_ponta_e_retomada(tmp_path):
    cfg = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))
    cfg["treino"]["lote"] = 4
    raiz, saida = _cache(tmp_path), str(tmp_path / "treino")
    res = treinar("M1", 0, 1e-4, cfg, saida, raiz=raiz, pesos_imagenet=False, epocas_max=1, workers=0)
    assert res["epocas_rodadas"] == 1
    res = treinar("M1", 0, 1e-4, cfg, saida, raiz=raiz, pesos_imagenet=False, epocas_max=2, workers=0, retomar=True)
    assert res["epocas_rodadas"] == 2                        # retomou da época 1, não recomeçou
    for arq in ["curvas.csv", "melhor.pt", "ultimo.pt", "predicoes_validacao.csv", "resumo.json"]:
        assert os.path.exists(os.path.join(saida, arq)), arq
    pred = pd.read_csv(os.path.join(saida, "predicoes_validacao.csv"))
    assert len(pred) == 8 and pred["prob"].between(0, 1).all()
    r = json.load(open(os.path.join(saida, "resumo.json"), encoding="utf-8"))
    # a correção de prior baixa as probabilidades (de 20% no treino para cerca de 2% natural)
    assert r["validacao_com_correcao"]["prob_media"] < r["validacao_sem_correcao"]["prob_media"]
