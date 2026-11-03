"""Covariáveis (passo 5.7) e tabela de predições da validação no formato do schema."""
import json

import numpy as np
import pandas as pd
import pytest

from src.cache.covariaveis import MEDIDAS, montar as montar_cov
from src.predicoes.schema import validar
from src.predicoes.tabela import montar as montar_tabela


def _elegiveis_plus(n=6):
    return pd.DataFrame({"path_to_image": "x", "chave_dcm": [f"p{i}/s1/v1_frontal.dcm" for i in range(n)],
                         "deid_patient_id": [f"pac{i}" for i in range(n)], "estudo": [f"p{i}/s1" for i in range(n)],
                         "patient_report_date_order": 0, "ap_pa": ["AP", "PA"] * (n // 2),
                         "idade": [40.0, 71.0] * (n // 2), "sex": ["Male", "Female"] * (n // 2), "pn_cat": "negativo"})


def _passada(ids, fab=("Agfa-Gevaert AG", '"GE Healthcare"')):
    d = {"id": list(ids), "fabricante": [fab[i % 2] for i in range(len(ids))]}
    d.update({m: np.linspace(0.1, 0.9, len(ids)) for m in MEDIDAS})
    return pd.DataFrame(d)


def test_covariaveis_plus_mapeia_e_filtra():
    el = _elegiveis_plus()
    ids = el["chave_dcm"].iloc[:4]                            # duas imagens (o "teste") ficam fora
    cov = montar_cov("plus", el, _passada(el["chave_dcm"]), ids)
    assert len(cov) == 4 and set(cov["sexo"]) == {"M", "F"} and set(cov["fabricante"]) == {"Agfa", "GE"}
    assert str(cov["idade"].dtype) == "Int64" and list(cov.columns[:6]) == ["id", "id_estudo", "posicao", "idade",
                                                                            "sexo", "fabricante"]


def test_covariaveis_rsna_fabricante_vazio_e_falta_imagem():
    el = pd.DataFrame({"SOPInstanceUID": ["1.1", "1.2"], "StudyInstanceUID": ["9.1", "9.2"], "proj": ["AP", "PA"],
                       "idade": [30, 50], "sexo": ["F", "M"]})
    cov = montar_cov("rsna", el, _passada(["1.1", "1.2"], fab=(np.nan, np.nan)), pd.Series(["1.1", "1.2"]))
    assert cov["fabricante"].isna().all()
    with pytest.raises(RuntimeError):
        montar_cov("rsna", el, _passada(["1.1"]), pd.Series(["1.1", "1.2"]))   # sem linha na passada


def test_covariaveis_aceita_mascara_vazia():
    el = _elegiveis_plus()
    pa = _passada(el["chave_dcm"])
    pa.loc[0, [m for m in MEDIDAS if m != "q_enquadramento"]] = np.nan      # máscara vazia
    cov = montar_cov("plus", el, pa, el["chave_dcm"])
    assert len(cov) == 6 and cov["q_brilho"].isna().sum() == 1


def _execucao(tmp_path, base="plus", embaralhado=False):
    el = _elegiveis_plus()
    cov = montar_cov("plus", el, _passada(el["chave_dcm"]), el["chave_dcm"])
    (tmp_path / "dados" / base).mkdir(parents=True)
    cov.to_csv(tmp_path / "dados" / base / "covariaveis.csv", index=False)
    pasta = tmp_path / "M1_s0_lr0.0001"
    pasta.mkdir()
    json.dump({"modelo": "M1", "semente": 0, "base": base, "rotulos_embaralhados": embaralhado},
              open(pasta / "resumo.json", "w"))
    z = np.array([-3.0, 1.0, -2.0, 0.5, -1.0, 2.0])
    pd.DataFrame({"id": el["chave_dcm"], "paciente": el["deid_patient_id"], "rotulo": [0, 1] * 3, "logit": z + 2.5,
                  "logit_corrigido": z, "prob": 1 / (1 + np.exp(-z))}).to_csv(pasta / "predicoes_validacao.csv",
                                                                              index=False)
    return str(pasta), str(tmp_path / "dados")


def test_tabela_da_validacao_passa_no_schema(tmp_path):
    pasta, dados = _execucao(tmp_path)
    t = montar_tabela(pasta, dados)
    assert validar(t) == []
    assert (t["fonte"] == "chexpert_plus").all() and (t["logit"] < 2.5).all()   # logit com a correção de prior
    assert t["probabilidade_temperatura"].isna().all() and t["id_execucao"].iloc[0] == "M1_s0_lr0.0001"


def test_tabela_recusa_sanidade_e_sem_covariaveis(tmp_path):
    pasta, dados = _execucao(tmp_path, embaralhado=True)
    with pytest.raises(ValueError):
        montar_tabela(pasta, dados)
    pasta, _ = _execucao(tmp_path / "b")
    with pytest.raises(FileNotFoundError):
        montar_tabela(pasta, str(tmp_path / "vazio"))
