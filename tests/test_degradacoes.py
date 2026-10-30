"""Testes das degradações do eixo B e das contas do piloto (sem DICOM, sem modelo)."""
import warnings

import numpy as np
import pandas as pd

from src.perturbations.degradacoes import (aplicar, brilho, contraste, desfoque, gama, interpolar_parametro,
                                           resolucao, ruido)
from src.qualidade.medidas import medidas_qualidade

rng = np.random.default_rng(0)
IMG = np.clip(rng.normal(120, 30, (600, 600)), 0, 255).astype(np.uint8)
ROI = np.zeros((600, 600), np.uint8)
ROI[100:500, 100:500] = 1


def q(img):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return medidas_qualidade(img, ROI)


def test_todas_devolvem_uint8_do_mesmo_tamanho():
    for tipo, p in [("gama", 1.5), ("contraste", 0.7), ("brilho", 20), ("desfoque", 2), ("ruido", 5), ("resolucao", 3)]:
        d = aplicar(IMG, tipo, p)
        assert d.dtype == np.uint8 and d.shape == IMG.shape


def test_parametro_neutro_nao_muda_a_imagem():
    assert np.array_equal(gama(IMG, 1.0), IMG)
    assert np.array_equal(brilho(IMG, 0), IMG)
    assert np.array_equal(desfoque(IMG, 0), IMG)
    assert np.array_equal(ruido(IMG, 0), IMG)
    assert np.array_equal(resolucao(IMG, 1), IMG)


def test_direcao_de_cada_degradacao():
    base = q(IMG)
    assert q(gama(IMG, 2.0))["q_brilho"] < base["q_brilho"] < q(gama(IMG, 0.5))["q_brilho"]
    assert q(contraste(IMG, 0.5))["q_contraste"] < base["q_contraste"]
    assert q(brilho(IMG, -30))["q_brilho"] < base["q_brilho"]
    assert q(desfoque(IMG, 2))["q_nitidez"] < base["q_nitidez"]
    assert q(resolucao(IMG, 4))["q_nitidez"] < base["q_nitidez"]
    plano = np.full((600, 600), 120, np.uint8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert medidas_qualidade(ruido(plano, 8), ROI)["q_ruido"] > medidas_qualidade(plano, ROI)["q_ruido"]


def test_desfoque_mais_forte_tira_mais_nitidez():
    n = [q(desfoque(IMG, s))["q_nitidez"] for s in (0.5, 1, 2, 4)]
    assert all(a > b for a, b in zip(n, n[1:]))


def test_ruido_reprodutivel_pela_semente():
    assert np.array_equal(ruido(IMG, 5, semente=7), ruido(IMG, 5, semente=7))
    assert not np.array_equal(ruido(IMG, 5, semente=7), ruido(IMG, 5, semente=8))


def test_interpolar_parametro():
    params, medianas = [1, 2, 4], [100.0, 60.0, 20.0]       # medida cai quando o parâmetro sobe
    assert interpolar_parametro(params, medianas, 60.0) == 2.0
    assert interpolar_parametro(params, medianas, 40.0) == 3.0
    assert np.isnan(interpolar_parametro(params, medianas, 5.0))     # a grade não alcança


def test_tabela_resumo_e_ancoragem_do_piloto():
    from src.perturbations.piloto import ancoragem, tabela_resumo
    linhas = []
    for i in range(10):
        nat = {"q_brilho": 100 + i, "q_contraste": 40 + i, "q_nitidez": 50 + 2 * i, "q_ruido": 2 + 0.1 * i}
        linhas.append({"id": f"a{i}", "tipo": "natural", "parametro": 0.0, "png_kb": 50,
                       **{f"{k}_orig": v for k, v in nat.items()}, **{f"{k}_320": v for k, v in nat.items()}})
        for s in (1.0, 2.0):
            d = dict(nat, q_nitidez=nat["q_nitidez"] / (2 * s))
            d320 = dict(nat, q_nitidez=nat["q_nitidez"] * (1 - 0.1 * s))      # sobrevive pouco
            linhas.append({"id": f"a{i}", "tipo": "desfoque", "parametro": s, "png_kb": 40,
                           **{f"{k}_orig": v for k, v in d.items()}, **{f"{k}_320": v for k, v in d320.items()}})
    df = pd.DataFrame(linhas)
    nat = df[df["tipo"] == "natural"].rename(columns=lambda c: c.removesuffix("_orig"))
    tab = tabela_resumo(df, nat)
    assert set(tab["degradacao"]) == {"desfoque"}
    assert (tab["efeito_orig_dp"] < 0).all() and (tab["sobrevive"] < 1).all()
    anc = ancoragem(df, nat)
    assert set(anc[anc["degradacao"] == "desfoque"]["percentil"]) == {"p25", "p10", "p5", "p1"}
