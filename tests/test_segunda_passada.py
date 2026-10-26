"""Testes da segunda passada do PadChest que não precisam de rede, do modelo de máscara nem do Redivis."""
import io
import os
import types
import zipfile

import cv2
import numpy as np
import pandas as pd
import pytest

import src.padchest.segunda_passada as sp
from src.padchest.segunda_passada import comparar, dhash, escala_png, janela_8bits, medir_imagem, numero


def _seg_falso(img8):
    p = np.zeros((2,) + img8.shape, dtype=np.uint8)
    h, w = img8.shape
    p[0, h // 6: 5 * h // 6, w // 8: 3 * w // 8] = 1
    p[1, h // 6: 5 * h // 6, 5 * w // 8: 7 * w // 8] = 1
    return p


def _linha(wc="2047", ww="4095", bits="12"):
    return types.SimpleNamespace(WindowCenter_DICOM=wc, WindowWidth_DICOM=ww, BitsStored_DICOM=bits)


def _png16(h=800, w=700, semente=0):
    rng = np.random.default_rng(semente)
    base = np.linspace(500, 3500, w)[None, :] + rng.normal(0, 60, (h, w))
    return (np.clip(base, 0, 4095) * escala_png(12)).astype(np.uint16)


def test_numero_le_listas_e_vazios():
    assert numero("[2047, 2047]") == 2047 and numero("-12.5") == -12.5 and np.isnan(numero(None))


def test_escala_bate_com_a_auditoria():
    assert escala_png(12) == pytest.approx(16.004, abs=1e-3)
    assert escala_png(10) == pytest.approx(64.06, abs=1e-2)


def test_janela_completa_equivale_ao_minimo_maximo_da_escala():
    g = np.array([[0, 65535 // 2, 65535]], dtype=np.uint16)
    j = janela_8bits(g, 2047.5, 4095, 12)
    assert j.tolist() == [[0, 127, 255]] or j.tolist() == [[0, 128, 255]]


def test_janela_estreita_satura_as_pontas():
    g = (np.array([[100, 2000, 4000]]) * escala_png(12)).astype(np.uint16)
    j = janela_8bits(g, 2000, 1000, 12)
    assert j[0, 0] == 0 and j[0, 2] == 255 and 120 <= j[0, 1] <= 135


def test_janela_sem_dado_devolve_none():
    g = np.zeros((4, 4), np.uint16)
    assert janela_8bits(g, np.nan, 100, 12) is None and janela_8bits(g, 100, 0, 12) is None


def test_dhash_estavel_e_sensivel():
    a = (np.tile(np.arange(256, dtype=np.uint8), (64, 1)))
    assert dhash(a) == dhash(a.copy()) and len(dhash(a)) == 16
    assert dhash(a) != dhash(a[:, ::-1].copy())


def test_medir_imagem_tem_as_duas_conversoes():
    reg = medir_imagem(_png16(), _linha(), _seg_falso)
    assert reg["janela_disponivel"] and reg["alt"] == 800 and reg["larg"] == 700
    assert "q_brilho" in reg and "q_brilho_janela" in reg and "q_ruido_janela" in reg
    assert reg["q_aux_area_roi"] == pytest.approx(reg["q_aux_area_roi_janela"])   # mesma máscara


def test_medir_imagem_sem_janela_so_tem_a_oficial():
    reg = medir_imagem(_png16(), _linha(wc="", ww=""), _seg_falso)
    assert not reg["janela_disponivel"] and "q_brilho_janela" not in reg and "q_brilho" in reg


def test_processar_pacote_com_retomada(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "SAIDA", str(tmp_path / "medidas"))
    monkeypatch.setattr(sp, "SALVAR_A_CADA", 2)
    os.makedirs(sp.SAIDA)
    ids = [f"img{i}.png" for i in range(5)]
    zipf = tmp_path / "3.zip"
    with zipfile.ZipFile(zipf, "w") as z:
        for k, i in enumerate(ids[:4]):                       # img4 não está no zip: deve virar abriu=False
            ok, buf = cv2.imencode(".png", _png16(600, 600, k))
            z.writestr(f"3/{i}", buf.tobytes())
    rec = pd.DataFrame({"ImageID": ids, "ImageDir": 3, "WindowCenter_DICOM": "2047",
                        "WindowWidth_DICOM": "4095", "BitsStored_DICOM": 12})
    os.makedirs(f"{sp.SAIDA}/parcial_pacote_3")
    pd.DataFrame([{"ImageID": "img0.png", "ImageDir": 3, "abriu": True}]).to_csv(
        f"{sp.SAIDA}/parcial_pacote_3/bloco_0.csv", index=False)   # sessão anterior parou depois da img0
    sp.processar_pacote(3, str(zipf), rec, _seg_falso)
    out = pd.read_csv(f"{sp.SAIDA}/medidas_pacote_3.csv")
    assert sorted(out["ImageID"]) == ids
    assert out.set_index("ImageID").loc["img4.png", "abriu"] == False   # noqa: E712
    assert out["q_brilho"].notna().sum() == 3                  # img1..img3 medidas agora
    assert not zipf.exists() and not os.path.exists(f"{sp.SAIDA}/parcial_pacote_3")


def test_comparar_conversoes():
    rng = np.random.default_rng(0)
    a = rng.normal(120, 10, 200)
    df = pd.DataFrame({"q_brilho": a, "q_brilho_janela": a + 5 + rng.normal(0, 1, 200)})
    r = comparar(df).set_index("medida")
    assert r.loc["q_brilho", "spearman"] > 0.9
    assert r.loc["q_brilho", "diferenca_mediana"] == pytest.approx(5, abs=0.5)


def test_link_nao_esta_no_codigo():
    texto = open(sp.__file__, encoding="utf-8").read()
    assert "http" not in texto            # nenhum endereço escrito no código: o link vem de PADCHEST_LINK
