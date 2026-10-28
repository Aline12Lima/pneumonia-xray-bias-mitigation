"""Testes da passada completa do RSNA e do Plus que não precisam do Redivis nem do modelo de máscara."""
import os

import numpy as np
import pandas as pd
import pytest

pydicom = pytest.importorskip("pydicom")
from pydicom.dataset import FileDataset, FileMetaDataset  # noqa: E402
from pydicom.uid import ExplicitVRLittleEndian  # noqa: E402

import src.passada.passada_completa as pc  # noqa: E402


def _seg(img8):
    p = np.zeros((2,) + img8.shape, dtype=np.uint8)
    h, w = img8.shape
    p[0, h // 6: 5 * h // 6, w // 8: 3 * w // 8] = 1
    p[1, h // 6: 5 * h // 6, 5 * w // 8: 7 * w // 8] = 1
    return p


def _dicom(caminho, px, fotometria="MONOCHROME2", janela=None, fabricante="GE"):
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1"
    meta.MediaStorageSOPInstanceUID = "1.2.3"
    ds = FileDataset(str(caminho), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.Rows, ds.Columns = px.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, fotometria
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
    ds.Manufacturer, ds.PixelSpacing = fabricante, [0.15, 0.15]
    if janela:
        ds.WindowCenter, ds.WindowWidth = janela
    ds.PixelData = px.astype(np.uint16).tobytes()
    ds.save_as(str(caminho), enforce_file_format=True)
    return str(caminho)


def _px(h=700, w=600, semente=0):
    rng = np.random.default_rng(semente)
    return np.clip(np.linspace(300, 3800, w)[None, :] + rng.normal(0, 50, (h, w)), 0, 4095)


def test_nome_cache_seguro():
    assert pc.nome_cache("patient1/study2/view1_frontal.dcm") == "patient1__study2__view1_frontal.png"
    assert pc.nome_cache("1.2.840.1") == "1.2.840.1.png"


def test_processar_dicom_sem_janela(tmp_path):
    reg, c320 = pc.processar_dicom(_dicom(tmp_path / "a.dcm", _px(), fabricante="AGFA"), _seg)
    assert c320.shape == (320, 320) and c320.dtype == np.uint8
    assert reg["linhas"] == 700 and reg["colunas"] == 600 and reg["espacamento_mm"] == pytest.approx(0.15)
    assert not reg["janela_disponivel"] and "q_brilho_janela" not in reg
    assert len(reg["md5_pixels"]) == 32 and len(reg["dhash"]) == 16 and "q_ruido" in reg


def test_processar_dicom_com_janela_ge(tmp_path):
    reg, _ = pc.processar_dicom(_dicom(tmp_path / "b.dcm", _px(), janela=(2048, 1500)), _seg)
    assert reg["janela_disponivel"] and "q_brilho_janela" in reg
    assert reg["q_aux_area_roi"] == pytest.approx(reg["q_aux_area_roi_janela"])     # mesma máscara


def test_monochrome1_invertido(tmp_path):
    px = _px()
    r2, c2 = pc.processar_dicom(_dicom(tmp_path / "m2.dcm", px), _seg)
    r1, c1 = pc.processar_dicom(_dicom(tmp_path / "m1.dcm", px, "MONOCHROME1"), _seg)
    assert abs(int(c1.mean()) - (255 - int(c2.mean()))) <= 2
    assert r1["md5_pixels"] == r2["md5_pixels"]                    # MD5 é dos pixels gravados, não da conversão


def test_rodar_com_retomada_e_resumo(tmp_path, monkeypatch, capsys):
    tab = tmp_path / "eleg.csv"
    ids = [f"1.2.{i}" for i in range(5)]
    pd.DataFrame({"SOPInstanceUID": ids}).to_csv(tab, index=False)
    monkeypatch.setitem(pc.BASES, "rsna", {"tabela": str(tab), "id": "SOPInstanceUID",
                                           "saida": str(tmp_path / "out"), "tmp": str(tmp_path / "tmp")})
    monkeypatch.setattr(pc, "SALVAR_A_CADA", 2)

    def baixador_falso(base, alvo):
        def baixar(ident):
            if ident == "1.2.4":
                raise RuntimeError("sem arquivo")
            d = tmp_path / "tmp" / ident
            d.mkdir(parents=True, exist_ok=True)
            return _dicom(d / "x.dcm", _px(semente=int(ident[-1])))
        return baixar

    monkeypatch.setattr(pc, "baixador", baixador_falso)
    import src.imagem.mascara as mascara
    monkeypatch.setattr(mascara, "segmentar_pulmoes", _seg)

    pc.rodar("rsna", limite=3)                                      # sessão que "parou" depois de 3
    pc.rodar("rsna")                                                # retoma: só as 2 que faltam
    df = pc.juntar("rsna")
    assert sorted(df["id"]) == ids and df["abriu"].sum() == 4
    assert len(os.listdir(tmp_path / "out" / "c320_8bits")) == 4
    assert not any((tmp_path / "tmp").rglob("*.dcm"))               # DICOMs apagados
    pc.resumo("rsna")
    assert "5 de 5 processadas" in capsys.readouterr().out


def test_rodar_so_as_listas_na_ordem(tmp_path, monkeypatch):
    tab = tmp_path / "eleg.csv"
    pd.DataFrame({"SOPInstanceUID": [f"1.2.{i}" for i in range(6)]}).to_csv(tab, index=False)
    lista = tmp_path / "lista.csv"
    pd.DataFrame({"id": ["1.2.5", "1.2.1"]}).to_csv(lista, index=False)
    monkeypatch.setitem(pc.BASES, "rsna", {"tabela": str(tab), "id": "SOPInstanceUID",
                                           "saida": str(tmp_path / "out"), "tmp": str(tmp_path / "tmp")})
    vistos = []

    def baixador_falso(base, alvo):
        def baixar(ident):
            vistos.append(ident)
            d = tmp_path / "tmp" / ident
            d.mkdir(parents=True, exist_ok=True)
            return _dicom(d / "x.dcm", _px())
        return baixar

    monkeypatch.setattr(pc, "baixador", baixador_falso)
    import src.imagem.mascara as mascara
    monkeypatch.setattr(mascara, "segmentar_pulmoes", _seg)
    pc.rodar("rsna", listas=[str(lista)])
    assert sorted(vistos) == ["1.2.1", "1.2.5"] and sorted(pc.juntar("rsna")["id"]) == ["1.2.1", "1.2.5"]
