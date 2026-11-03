"""M3 (F6): época balanceada, parada pela média das fontes e correções de prior 3a e 3c."""
import hashlib
import json
import math
import os

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from src.treino.carregador import nome_cache
from src.treino.treinar import correcoes_m3, indices_balanceados, prevalencias, treinar


def _base(raiz, base, n, desloc):
    cache = raiz / base / "passada" / "c320_8bits"
    listas = raiz / base / "listas"
    cache.mkdir(parents=True); listas.mkdir(parents=True)
    rng = np.random.default_rng(desloc)
    col_pac = "deid_patient_id" if base == "plus" else "nih_paciente"
    linhas, hashes = [], []
    for k in range(n):
        pos = k % 2 == 0
        img = rng.integers(60, 120, (320, 320)).astype(np.uint8)
        if pos:
            img[100:220, 100:220] = 230
        ident = f"patient{desloc + k}/study1/view1_frontal.dcm" if base == "plus" else f"1.2.{desloc + k}"
        cv2.imwrite(str(cache / nome_cache(ident)), img)
        linhas.append({"id": ident, col_pac: f"p{desloc + k}", "rotulo": "positivo" if pos else "negativo"})
        hashes.append({"id": ident, "sha256": hashlib.sha256((cache / nome_cache(ident)).read_bytes()).hexdigest()})
    h = n * 2 // 3
    nome_tr = "treino_subamostra.csv" if base == "plus" else "treino.csv"
    pd.DataFrame(linhas[:h]).to_csv(listas / nome_tr, index=False)
    pd.DataFrame(linhas[h:]).to_csv(listas / "validacao.csv", index=False)
    pd.DataFrame(hashes).to_csv(raiz / base / "cache_hashes.csv", index=False)


def test_indices_balanceados():
    fontes = np.array(["plus"] * 6 + ["rsna"] * 10)
    a = indices_balanceados(fontes, 0, 0)
    assert len(a) == 12 and len(set(a)) == 12                       # sem reposição
    assert sum(fontes[a] == "plus") == 6 and sum(fontes[a] == "rsna") == 6
    assert set(np.flatnonzero(fontes == "plus")) <= set(a)          # todas as da menor fonte
    assert a == indices_balanceados(fontes, 0, 0)                   # fixado por (semente, época)
    assert a != indices_balanceados(fontes, 0, 1)                   # muda a cada época


def test_correcoes_m3():
    c = correcoes_m3()
    lo = lambda p: math.log(p / (1 - p))                             # noqa: E731
    (pn_p, pt_p), (pn_r, pt_r) = prevalencias("plus"), prevalencias("rsna")
    assert c["3c"]["plus"] == pytest.approx(lo(pn_p) - lo(pt_p)) and c["3c"]["rsna"] == pytest.approx(0)
    assert c["3a"] == pytest.approx(lo((pn_p + pn_r) / 2) - lo((pt_p + pt_r) / 2))
    assert c["externa"] == c["3a"]


def test_m3_de_ponta_a_ponta(tmp_path):
    raiz = tmp_path / "dados"
    _base(raiz, "plus", 12, 0)
    _base(raiz, "rsna", 18, 100)
    cfg = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))
    cfg["treino"]["lote"] = 4
    saida = str(tmp_path / "M3")
    res = treinar("M3", 0, 1e-4, cfg, saida, raiz=str(raiz), pesos_imagenet=False, epocas_max=2, workers=0)
    assert res["bases"] == ["plus", "rsna"] and res["correcao_principal"] in ("3a", "3c")
    cur = pd.read_csv(os.path.join(saida, "curvas.csv"))
    assert {"auprc_plus", "auprc_rsna"} <= set(cur.columns)
    assert cur["auprc"].tolist() == pytest.approx(((cur["auprc_plus"] + cur["auprc_rsna"]) / 2).tolist())
    pred = pd.read_csv(os.path.join(saida, "predicoes_validacao.csv"))
    assert set(pred["fonte"]) == {"plus", "rsna"} and len(pred) == 4 + 6
    c = res["correcao_prior"]
    d = pred["logit_corrigido_3c"] - pred["logit"]
    assert d[pred["fonte"] == "plus"].tolist() == pytest.approx([c["3c"]["plus"]] * 4, abs=1e-5)
    assert d[pred["fonte"] == "rsna"].tolist() == pytest.approx([0] * 6, abs=1e-5)
    escolhida = pred[f"logit_corrigido_{res['correcao_principal']}"]
    assert pred["logit_corrigido"].tolist() == pytest.approx(escolhida.tolist())
    e3a, e3c = res["validacao_3a"]["ece"], res["validacao_3c"]["ece"]
    assert res["correcao_principal"] == ("3c" if e3c <= e3a else "3a")


def test_log_em_arquivo(tmp_path, capsys):
    from src.treino.treinar import _Log
    arq = tmp_path / "log.txt"
    with _Log(str(arq)):
        print("época de teste")
    assert "época de teste" in arq.read_text(encoding="utf-8")
    assert "época de teste" in capsys.readouterr().out          # continua aparecendo na tela
