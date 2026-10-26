"""Testes das peças do teste "máscara × doença" que não dependem do Redivis nem do modelo."""
import json

import numpy as np
import pandas as pd
import pytest

from src.imagem.teste_mascara_doenca import (amostrar, caixas_por_imagem, cobertura, mapear_arquivos,
                                             parear, preencher_pulmoes, resumir)


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


def _tab(n=40):
    """Mesmo formato do rsna_elegiveis.csv (booleanos como texto, como saem do CSV)."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "SOPInstanceUID": [f"1.2.{i}" for i in range(n * 4)],
        "img_id": [f"{i:08d}_000.png" for i in range(n * 4)],
        "nih_paciente": [i // 2 for i in range(n * 4)],
        "proj": ["AP"] * (2 * n) + ["PA"] * (2 * n),
        "positivo": (["True"] * n + ["False"] * n) * 2,
        "exclude": ["False"] * (n * 4),
        "kaggle_teste": ["False"] * (n * 4),
        "idade": rng.integers(18, 90, n * 4),
    })


def test_amostra_estratificada_e_reprodutivel():
    a1, a2 = amostrar(_tab(), 10), amostrar(_tab(), 10)
    assert a1.groupby(["proj", "positivo"]).size().tolist() == [10, 10, 10, 10]
    assert a1["SOPInstanceUID"].tolist() == a2["SOPInstanceUID"].tolist()


def test_amostra_tira_exclude_e_teste_do_kaggle():
    t = _tab()
    t.loc[:5, "exclude"] = "True"
    t.loc[6:9, "kaggle_teste"] = "True"
    a = amostrar(t, 30)
    assert not set(t.loc[:9, "SOPInstanceUID"]) & set(a["SOPInstanceUID"])


def test_amostra_avisa_colunas_faltando():
    with pytest.raises(KeyError, match="colunas disponíveis"):
        amostrar(_tab().drop(columns="proj"), 10)


def test_pareamento_dentro_da_projecao_e_sem_repetir():
    a = amostrar(_tab(), 10)
    pares = parear(a)
    proj = dict(zip(a["SOPInstanceUID"], a["proj"]))
    pos = dict(zip(a["SOPInstanceUID"], a["positivo"]))
    assert len(pares) == 20 and len(set(pares.values())) == 20
    assert all(proj[p] == proj[n] and pos[p] and not pos[n] for p, n in pares.items())


def test_mapear_arquivos_descobre_a_coluna():
    a = amostrar(_tab(), 10)
    col, mapa = mapear_arquivos([f"{s}.dcm" for s in a["SOPInstanceUID"]], a)
    assert col == "SOPInstanceUID" and mapa[a["SOPInstanceUID"][0]] == f"{a['SOPInstanceUID'][0]}.dcm"
    col, _ = mapear_arquivos([f"{s.replace('.png', '')}.dcm" for s in a["img_id"]], a)
    assert col == "img_id"
    with pytest.raises(KeyError, match="exemplos"):
        mapear_arquivos(["outro.dcm"], a)


def test_preencher_devolve_o_buraco_da_opacidade():
    pulmoes = np.zeros((2, 100, 100), dtype=np.uint8)
    pulmoes[0, 10:90, 10:40] = 1
    pulmoes[0, 50:70, 25:40] = 0                       # opacidade deixada de fora, na borda interna
    pulmoes[1, 10:90, 60:90] = 1
    cheia = preencher_pulmoes(pulmoes)
    assert cheia[60, 30] == 1 and cheia[50, 50] == 0   # buraco devolvido; mediastino continua fora
    assert cheia.sum() > pulmoes.max(axis=0).sum()


def _linhas(cob_pos, cob_transpl, area_pos, raz_enc):
    rng = np.random.default_rng(1)
    linhas = []
    for proj in ["AP", "PA"]:
        for pos in [True, False]:
            for i in range(60):
                linhas.append({"nih_paciente": f"{proj}{pos}{i}", "proj": proj, "positivo": pos,
                               "area": rng.normal(area_pos if pos else 0.25, 0.02),
                               "razao_pulmoes": 0.3 if (pos and i < raz_enc) else 0.85,
                               "cobertura_caixas": cob_pos(rng) if pos else np.nan,
                               "cobertura_transplantada": cob_transpl(rng) if pos else np.nan,
                               "q_brilho_mascara": rng.normal(120 + 5 * pos, 10),
                               "q_brilho_preenchida": rng.normal(120 + 15 * pos, 10)})
    return pd.DataFrame(linhas)


def test_regra_detecta_mascara_que_encolhe_nos_positivos():
    res = _linhas(lambda r: r.uniform(0.3, 0.6), lambda r: r.uniform(0.75, 0.85), 0.18, 15)
    r = resumir(res, n_boot=200)
    assert r["dependente_da_doenca"]
    assert all(r["criterios"].values())
    assert r["por_projecao"]["AP"]["descritivo_brilho"]["mudanca"] > 0


def test_cobertura_baixa_mas_igual_a_referencia_nao_dispara():
    """Caixas que passam da borda do pulmão: cobertura baixa nos dois, sem perda relativa."""
    res = _linhas(lambda r: r.uniform(0.70, 0.80), lambda r: r.uniform(0.70, 0.80), 0.24, 0)
    r = resumir(res, n_boot=200)
    assert r["cobertura_mediana_positivos"] < 0.80
    assert not r["dependente_da_doenca"]
