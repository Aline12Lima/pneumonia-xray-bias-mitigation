"""Testes das listas de treino, validação e teste."""
import json

import numpy as np
import pandas as pd
import pytest

from src.divisao.listas import dividir, gerar, sha256_ids, subamostrar


def _plus(n_pac=2000, semente=0):
    rng = np.random.default_rng(semente)
    linhas = []
    for p in range(n_pac):
        for k in range(rng.integers(1, 5)):
            cat = rng.choice(["positivo", "incerto", "negativo", None], p=[0.03, 0.08, 0.01, 0.88])
            linhas.append({"chave_dcm": f"patient{p}/study{k}/view1_frontal.dcm",
                           "deid_patient_id": f"pid{p}", "pn_cat": cat if cat else "não mencionado"})
    return pd.DataFrame(linhas)


def test_sha256_nao_depende_da_ordem():
    assert sha256_ids(["b", "a", "c"]) == sha256_ids(["c", "b", "a"])
    assert sha256_ids(["a"]) != sha256_ids(["a", "b"])


def test_divisao_por_paciente_reprodutivel_e_proporcional():
    df = _plus().assign(rotulo=lambda d: d["pn_cat"].map({"positivo": "positivo", "incerto": "incerto"})
                        .fillna("negativo"))
    p1, p2 = dividir(df, "deid_patient_id"), dividir(df, "deid_patient_id")
    assert (p1 == p2).all()
    assert (df.assign(p=p1).groupby("deid_patient_id")["p"].nunique() == 1).all()
    frac = df.assign(p=p1).drop_duplicates("deid_patient_id")["p"].value_counts(normalize=True)
    assert frac["treino"] == pytest.approx(0.7, abs=0.02) and frac["teste"] == pytest.approx(0.2, abs=0.02)
    pos = df.assign(p=p1)[df["rotulo"] == "positivo"].drop_duplicates("deid_patient_id")["p"]
    assert pos.value_counts(normalize=True)["teste"] == pytest.approx(0.2, abs=0.05)   # estratificada


def test_subamostra_4_por_positivo_sem_incertos():
    tr = pd.DataFrame({"id": [f"i{k}" for k in range(200)],
                       "rotulo": ["positivo"] * 10 + ["incerto"] * 20 + ["negativo"] * 170})
    s1, s2 = subamostrar(tr), subamostrar(tr.sample(frac=1, random_state=3))
    assert (s1["rotulo"] == "positivo").sum() == 10 and (s1["rotulo"] == "negativo").sum() == 40
    assert "incerto" not in set(s1["rotulo"])
    assert s1["id"].tolist() == s2["id"].tolist()                     # não depende da ordem de entrada


def test_gerar_plus(tmp_path):
    tab = tmp_path / "eleg.csv"
    _plus().to_csv(tab, index=False)
    r = gerar("plus", str(tab), str(tmp_path / "plus"))
    l = r["listas"]
    assert l["treino"]["imagens"] + l["validacao"]["imagens"] + l["teste"]["imagens"] == r["elegiveis"]
    assert l["treino_subamostra"]["imagens"] == 5 * l["treino"]["positivos"]
    assert (tmp_path / "plus" / "listas_teste_trancado" / "teste.csv").exists()
    assert not (tmp_path / "plus" / "listas" / "teste.csv").exists()
    teste = pd.read_csv(tmp_path / "plus" / "listas_teste_trancado" / "teste.csv")
    assert sha256_ids(teste["id"]) == l["teste"]["sha256"]
    salvo = json.loads((tmp_path / "plus" / "listas" / "resumo.json").read_text(encoding="utf-8"))
    assert salvo["listas"]["teste"]["sha256"] == l["teste"]["sha256"]
    assert gerar("plus", str(tab), str(tmp_path / "plus2"))["listas"]["teste"]["sha256"] == l["teste"]["sha256"]


def test_gerar_padchest_tudo_teste(tmp_path):
    tab = tmp_path / "rec.csv"
    pd.DataFrame({"ImageID": ["a.png", "b.png", "c.png"], "PatientID": [1, 2, 3], "pos": [True, False, False],
                  "PatientSex_DICOM": ["M", "F", "O"]}).to_csv(tab, index=False)
    r = gerar("padchest", str(tab), str(tmp_path / "pc"))
    assert list(r["listas"]) == ["teste"] and r["listas"]["teste"]["imagens"] == 2
