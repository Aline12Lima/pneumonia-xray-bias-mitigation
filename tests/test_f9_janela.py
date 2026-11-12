"""F9 (amostra da revisão qualitativa) e sensibilidade da conversão do PadChest (janela, só Philips)."""
import json
import os

import numpy as np
import pandas as pd

from src.analises.f9_revisao import CATEGORIAS, folha, resumo, sha256_amostra, sortear
from src.predicoes.pos_treino import SEMENTES, TAXA


def _tab(tmp_path):
    rng = np.random.default_rng(0)
    linhas = []
    for m in TAXA:
        for s in SEMENTES:
            ex = f"{m}_s{s}_lr{TAXA[m]:g}"
            os.makedirs(tmp_path / ex, exist_ok=True)
            json.dump({"limiar": {"limiar_logit": 0.0}}, open(tmp_path / ex / "pos_treino.json", "w"))
            for f in ("chexpert_plus", "rsna", "padchest"):
                n = 1000
                y = (rng.random(n) < 0.3).astype(int)
                z = rng.normal(0, 2, n)
                linhas.append(pd.DataFrame({"modelo": m, "semente": s, "id_execucao": ex, "fonte": f,
                                            "id_imagem": [f"{f}{i}" for i in range(n)], "id_paciente": [f"{f}p{i}" for i in range(n)],
                                            "rotulo": y, "logit": z, "probabilidade": 1 / (1 + np.exp(-z))}))
    return pd.concat(linhas, ignore_index=True)


def test_amostra_da_f9(tmp_path):
    tab = _tab(tmp_path)
    am = sortear(tab, str(tmp_path))
    pares = am.groupby(["modelo", "fonte"]).size()
    assert set(pares.index) == {("M1", "rsna"), ("M1", "padchest"), ("M2", "chexpert_plus"), ("M2", "padchest"), ("M3", "padchest")}
    assert (am.groupby(["modelo", "fonte", "tipo_erro", "criterio"]).size() == 25).all()
    fp = am[am["tipo_erro"] == "FP"]
    assert (fp["rotulo"] == 0).all() and (am[am["tipo_erro"] == "FN"]["rotulo"] == 1).all()
    conf = am[(am["modelo"] == "M1") & (am["fonte"] == "rsna") & (am["tipo_erro"] == "FP")]
    assert conf[conf["criterio"] == "maior_confianca"]["probabilidade"].min() >= conf[conf["criterio"] == "aleatorio"]["probabilidade"].max()
    assert not am.duplicated(["modelo", "fonte", "tipo_erro", "id_imagem"]).any()
    assert sha256_amostra(am) == sha256_amostra(sortear(tab, str(tmp_path)))          # reprodutível
    # mesma imagem em dois pares → mesmo código
    pad = am[am["fonte"] == "padchest"]
    assert pad.groupby("id_imagem")["codigo"].nunique().max() == 1
    f = folha(am, "padchest")
    assert "id_imagem" not in f.columns and "probabilidade" not in f.columns and f["codigo"].is_unique
    f[list(CATEGORIAS)] = 0
    f.loc[0, "achado_concorrente"] = 1
    r = resumo(am, f)
    assert r["n_codificadas"].sum() == len(am[am["fonte"] == "padchest"])


def test_janela_so_philips_com_janela():
    from src.padchest.janela import converter_janela, selecionar
    lista = pd.DataFrame({"id": ["a.png", "b.png", "c.png", "d.png"], "PatientID": [1, 2, 3, 4],
                          "rotulo": ["positivo", "negativo", "negativo", "negativo"]})
    rec = pd.DataFrame({"ImageID": ["a.png", "b.png", "c.png", "d.png"],
                        "Manufacturer_DICOM": ["PhilipsMedicalSystems", "ImagingDynamicsCompanyLtd", "PhilipsMedicalSystems",
                                               "PhilipsMedicalSystems"],
                        "WindowCenter_DICOM": ["2047", "2047", None, "[2047, 2047]"], "WindowWidth_DICOM": ["4095", "4095", "4095", "4095"],
                        "BitsStored_DICOM": [12, 12, 12, 12]})
    assert selecionar(lista, rec)["id"].tolist() == ["a.png", "d.png"]
    g = (np.linspace(0, 4095, 600)[None, :].repeat(500, 0) * 16).astype(np.uint16)
    img = converter_janela(g, rec.iloc[0])
    assert img.shape == (320, 320) and img.dtype == np.uint8 and img.max() == 255 and img.min() == 0
