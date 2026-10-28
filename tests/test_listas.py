"""Testes das listas de treino, validação e teste."""
import json

import numpy as np
import pandas as pd
import pytest

from src.divisao.listas import dividir, excluir_duplicatas_pixel, gerar, sha256_ids, subamostrar


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


def test_gerar_rsna_tira_idade_acima_de_100(tmp_path):
    tab = tmp_path / "eleg.csv"
    pd.DataFrame({"SOPInstanceUID": [f"1.2.{i}" for i in range(40)], "nih_paciente": range(40),
                  "positivo": ["True", "False"] * 20, "idade": [40] * 38 + [150, 412]}).to_csv(tab, index=False)
    r = gerar("rsna", str(tab), str(tmp_path / "rsna"))
    assert r["elegiveis"] == 38


def test_duplicatas_pixel_mesmo_paciente_fica_a_mais_antiga_e_entre_pacientes_saem_todas():
    df = pd.DataFrame({"SOPInstanceUID": ["a", "b", "c", "d", "e"], "nih_paciente": [1, 1, 2, 3, 4],
                       "seguimento": [12, 3, 0, 0, 0]})
    md5 = pd.DataFrame({"id": ["a", "b", "c", "d", "e"], "md5_pixels": ["x", "x", "y", "y", "z"]})
    fica, sai = excluir_duplicatas_pixel(df, md5, "SOPInstanceUID", "nih_paciente", "seguimento")
    assert sorted(sai) == ["a", "c", "d"] and fica["SOPInstanceUID"].tolist() == ["b", "e"]


def test_gerar_rsna_usa_os_blocos_da_passada(tmp_path):
    tab = tmp_path / "eleg.csv"
    pd.DataFrame({"SOPInstanceUID": [f"s{i}" for i in range(20)], "nih_paciente": [0, 0] + list(range(1, 19)),
                  "positivo": ["False"] * 20, "idade": 50, "seguimento": [5, 1] + [0] * 18}).to_csv(tab, index=False)
    (tmp_path / "rsna" / "passada" / "blocos").mkdir(parents=True)
    pd.DataFrame({"id": [f"s{i}" for i in range(20)], "md5_pixels": ["m"] * 2 + [f"u{i}" for i in range(18)]}).to_csv(
        tmp_path / "rsna" / "passada" / "blocos" / "b.csv", index=False)
    r = gerar("rsna", str(tab), str(tmp_path / "rsna"))
    assert r["elegiveis"] == 19 and r["duplicatas_pixel_retiradas"] == 1
