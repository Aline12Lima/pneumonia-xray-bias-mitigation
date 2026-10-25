"""Testes das peças do teste "máscara × doença" que não dependem do Redivis nem do modelo."""
import json

import numpy as np
import pandas as pd
import pytest

from src.imagem.teste_mascara_doenca import amostrar, caixas_por_imagem, cobertura, resumir


def test_cobertura_total_parcial_e_vazia():
    roi = np.zeros((100, 100), dtype=np.uint8)
    roi[:, :50] = 1
    assert cobertura(roi, [(10, 10, 20, 20)]) == pytest.approx(1.0)
    assert cobertura(roi, [(40, 10, 20, 20)]) == pytest.approx(0.5)
    assert cobertura(roi, [(60, 10, 20, 20)]) == pytest.approx(0.0)
    assert np.isnan(cobertura(roi, []))


def test_cobertura_une_caixas_sobrepostas():
    roi = np.ones((100, 100), dtype=np.uint8)
    roi[:, 50:] = 0
    assert cobertura(roi, [(40, 0, 20, 10), (40, 0, 20, 10)]) == pytest.approx(0.5)


def test_caixas_so_do_grupo_final(tmp_path):
    anot = {"labelGroups": [
        {"name": "Team 1", "labels": [{"id": "L_a", "name": "Lung Opacity (High Prob)"}]},
        {"name": "Calculated", "labels": [{"id": "L_v8n", "name": "Lung Opacity"},
                                          {"id": "L_o8w", "name": "Normal"}]}],
        "datasets": [{"annotations": [
            {"SOPInstanceUID": "s1", "labelId": "L_v8n", "data": {"x": 1, "y": 2, "width": 3, "height": 4}},
            {"SOPInstanceUID": "s1", "labelId": "L_a", "data": {"x": 9, "y": 9, "width": 9, "height": 9}},
            {"SOPInstanceUID": "s2", "labelId": "L_o8w", "data": None}]}]}
    p = tmp_path / "anot.json"
    p.write_text(json.dumps(anot))
    assert caixas_por_imagem(p) == {"s1": [(1.0, 2.0, 3.0, 4.0)]}


def _cab(n=40):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "patientId": [f"s{i}" for i in range(n * 4)],
        "nih_paciente": [f"p{i // 2}" for i in range(n * 4)],
        "proj": ["AP"] * (2 * n) + ["PA"] * (2 * n),
        "classe_final": (["Lung Opacity"] * n + ["Normal"] * n) * 2,
        "idade": rng.integers(18, 90, n * 4),
        "kaggle_teste": [False] * (n * 4),
    })


def test_amostra_estratificada_e_reprodutivel():
    a1, a2 = amostrar(_cab(), 10), amostrar(_cab(), 10)
    assert a1.groupby(["proj", "positivo"]).size().tolist() == [10, 10, 10, 10]
    assert a1["patientId"].tolist() == a2["patientId"].tolist()


def test_regra_detecta_mascara_que_encolhe_nos_positivos():
    rng = np.random.default_rng(1)
    linhas = []
    for proj in ["AP", "PA"]:
        for pos in [True, False]:
            for i in range(60):
                linhas.append({"nih_paciente": f"{proj}{pos}{i}", "proj": proj, "positivo": pos,
                               "area": rng.normal(0.18 if pos else 0.25, 0.02),
                               "razao_pulmoes": 0.3 if (pos and i < 15) else 0.85,
                               "cobertura_caixas": rng.uniform(0.3, 0.7) if pos else np.nan})
    r = resumir(pd.DataFrame(linhas), n_boot=200)
    assert r["dependente_da_doenca"]
    assert r["criterios"]["1_cobertura_baixa"] and r["criterios"]["2_area_menor_nos_positivos"]


def test_regra_nao_dispara_quando_mascara_e_estavel():
    rng = np.random.default_rng(2)
    linhas = []
    for proj in ["AP", "PA"]:
        for pos in [True, False]:
            for i in range(60):
                linhas.append({"nih_paciente": f"{proj}{pos}{i}", "proj": proj, "positivo": pos,
                               "area": rng.normal(0.24, 0.02), "razao_pulmoes": 0.85,
                               "cobertura_caixas": rng.uniform(0.85, 1.0) if pos else np.nan})
    assert not resumir(pd.DataFrame(linhas), n_boot=200)["dependente_da_doenca"]


def test_amostra_avisa_colunas_faltando():
    with pytest.raises(KeyError, match="colunas disponíveis"):
        amostrar(_cab().drop(columns="proj"), 10)
