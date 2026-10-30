"""Testes das medidas de qualidade (rodar com: python -m pytest tests/).

Cada teste usa uma imagem sintética em que a resposta certa é conhecida.
"""
import warnings

import cv2
import numpy as np
import pytest

from src.qualidade.medidas import COLUNAS, medidas_qualidade, margem_preta

H, W = 600, 800


def plana(valor: int = 120) -> np.ndarray:
    return np.full((H, W), valor, dtype=np.uint8)


def roi_central() -> np.ndarray:
    m = np.zeros((H, W), dtype=np.uint8)
    m[150:450, 200:600] = 1
    return m


def test_devolve_todas_as_colunas_com_prefixo():
    q = medidas_qualidade(plana(), roi_central())
    assert set(q) == set(COLUNAS)
    assert all(c.startswith("q_") for c in q)


def test_brilho_e_contraste_de_imagem_plana():
    q = medidas_qualidade(plana(120), roi_central())
    assert q["q_brilho"] == pytest.approx(120)
    assert q["q_contraste"] == pytest.approx(0)
    assert q["q_amplitude"] == pytest.approx(0)
    assert q["q_entropia"] == pytest.approx(0)
    assert q["q_saturacao"] == 0


def test_brilho_mede_so_dentro_da_roi():
    img = plana(40)
    img[150:450, 200:600] = 200
    assert medidas_qualidade(img, roi_central())["q_brilho"] == pytest.approx(200)


def test_contraste_e_amplitude_crescem_com_o_espalhamento():
    rng = np.random.default_rng(0)
    estreita = np.clip(rng.normal(128, 5, (H, W)), 0, 255).astype(np.uint8)
    larga = np.clip(rng.normal(128, 30, (H, W)), 0, 255).astype(np.uint8)
    qe, ql = medidas_qualidade(estreita, roi_central()), medidas_qualidade(larga, roi_central())
    assert ql["q_contraste"] > 4 * qe["q_contraste"]
    assert ql["q_amplitude"] > 4 * qe["q_amplitude"]
    assert ql["q_entropia"] > qe["q_entropia"]


def test_saturacao_conta_os_dois_extremos():
    img = plana(120)
    roi = roi_central()
    ys, xs = np.nonzero(roi)
    n = len(ys)
    img[ys[: n // 10], xs[: n // 10]] = 0                       # 10% pretos
    img[ys[n // 10: n // 10 + n // 20], xs[n // 10: n // 10 + n // 20]] = 255   # 5% brancos
    q = medidas_qualidade(img, roi)
    assert q["q_aux_saturacao_baixa"] == pytest.approx(0.10, abs=1e-3)
    assert q["q_aux_saturacao_alta"] == pytest.approx(0.05, abs=1e-3)
    assert q["q_saturacao"] == pytest.approx(0.15, abs=1e-3)


def test_enquadramento_mede_a_margem_preta_ligada_a_borda():
    img = plana(120)
    img[:, :80] = 0                                             # faixa preta à esquerda: 10% da imagem
    img[300:310, 400:410] = 0                                   # mancha preta isolada: não é margem
    assert margem_preta(img) == pytest.approx(0.10, abs=1e-3)
    assert medidas_qualidade(img, roi_central())["q_enquadramento"] == pytest.approx(0.10, abs=1e-3)


def test_enquadramento_zero_sem_margem():
    assert margem_preta(plana(120)) == 0.0


def test_nitidez_cai_com_desfoque():
    rng = np.random.default_rng(1)
    nitida = (rng.random((H, W)) > 0.5).astype(np.uint8) * 200 + 20
    borrada = cv2.GaussianBlur(nitida, (0, 0), 3)
    qn, qb = medidas_qualidade(nitida, roi_central()), medidas_qualidade(borrada, roi_central())
    assert qn["q_nitidez"] > 10 * qb["q_nitidez"]


def test_ruido_cresce_com_ruido_adicionado_e_ignora_bordas():
    rng = np.random.default_rng(2)
    base = plana(120).astype(np.float32)
    base[:, W // 2:] = 160                                      # uma borda forte não deve virar ruído
    pouco = np.clip(base + rng.normal(0, 2, (H, W)), 0, 255).astype(np.uint8)
    muito = np.clip(base + rng.normal(0, 8, (H, W)), 0, 255).astype(np.uint8)
    qp, qm = medidas_qualidade(pouco, roi_central()), medidas_qualidade(muito, roi_central())
    assert qm["q_ruido"] > 2.5 * qp["q_ruido"]


def test_roi_vazia_devolve_nan_mas_mantem_enquadramento():
    img = plana(120)
    img[:, :80] = 0
    q = medidas_qualidade(img, np.zeros((H, W), dtype=np.uint8))
    assert q["q_aux_roi_vazia"] == 1.0
    assert np.isnan(q["q_brilho"]) and np.isnan(q["q_ruido"])
    assert q["q_enquadramento"] == pytest.approx(0.10, abs=1e-3)


def test_saturacao_nativa_usa_a_imagem_original():
    bruta = np.full((H, W), 30000, dtype=np.uint16)
    roi = roi_central()
    ys, xs = np.nonzero(roi)
    bruta[ys[:1200], xs[:1200]] = 65519                          # teto do aparelho dentro do pulmão
    bruta[0, 0] = 0
    img8 = (bruta / 257).astype(np.uint8)
    q = medidas_qualidade(img8, roi, bruta)
    assert q["q_aux_saturacao_nativa"] == pytest.approx(1200 / roi.sum(), rel=1e-6)


def test_sem_bruta_saturacao_nativa_fica_vazia():
    assert np.isnan(medidas_qualidade(plana(), roi_central())["q_aux_saturacao_nativa"])


def test_rejeita_imagem_que_nao_e_8_bits():
    with pytest.raises(ValueError):
        medidas_qualidade(np.zeros((H, W), dtype=np.uint16), roi_central())


def test_rejeita_tamanhos_diferentes():
    with pytest.raises(ValueError):
        medidas_qualidade(plana(), np.ones((H // 2, W // 2), dtype=np.uint8))


def test_avisa_quando_a_imagem_parece_redimensionada():
    pequena = np.full((320, 320), 120, dtype=np.uint8)
    roi = np.zeros((320, 320), dtype=np.uint8)
    roi[60:260, 60:260] = 1
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        medidas_qualidade(pequena, roi)
    assert any("resolução original" in str(x.message) for x in w)


def test_fabricante_unificado():
    from src.qualidade.fabricante import normalizar_fabricante as n
    assert {n('"GE Healthcare"'), n("GE Healthcare"), n("GE MEDICAL SYSTEMS")} == {"GE"}
    assert {n("AGFA"), n("Agfa"), n("Agfa-Gevaert AG")} == {"Agfa"}
    assert n("PhilipsMedicalSystems") == "Philips" and n("ImagingDynamicsCompanyLtd") == "Imaging Dynamics"
    assert n(None) == "Desconhecido" and n("") == "Desconhecido" and n("Varian") == "Varian"
