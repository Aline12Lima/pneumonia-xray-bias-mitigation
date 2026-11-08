"""Eixo B (§8.4): amostra por paciente e geração das versões degradadas."""
import numpy as np
import pandas as pd
import pytest

from src.perturbations.eixo_b import sortear, uma_imagem


def _tab():
    rng = np.random.default_rng(0)
    n = 3000
    return pd.DataFrame({"id": [f"i{i}" for i in range(n)], "paciente": [f"p{i // 3}" for i in range(n)],
                         "rotulo": (rng.random(n) < 0.1).astype(int), "caminho": "x", "fonte": "plus"})


def test_amostra_por_paciente():
    a = sortear(_tab(), 100)
    pos, neg = a[a["rotulo"] == 1], a[a["rotulo"] == 0]
    assert len(pos) == 100 and len(neg) == 400
    assert pos["paciente"].is_unique and neg["paciente"].is_unique
    assert a.equals(sortear(_tab(), 100))                           # semente fixa
    todos = sortear(_tab(), None)
    assert (todos["rotulo"] == 1).sum() == (_tab()["rotulo"] == 1).sum()


def test_uma_imagem_gera_todas_as_versoes(tmp_path, monkeypatch):
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian
    rng = np.random.default_rng(1)
    px = (rng.random((640, 600)) * 4000).astype(np.uint16)
    meta = FileMetaDataset(); meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID = "1.2.840.10008.5.1.4.1.1.1", "1.2.3"
    ds = FileDataset(str(tmp_path / "a.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.Rows, ds.Columns, ds.BitsAllocated, ds.BitsStored, ds.HighBit = 640, 600, 16, 12, 11
    ds.SamplesPerPixel, ds.PixelRepresentation, ds.PhotometricInterpretation = 1, 0, "MONOCHROME2"
    ds.PixelData = px.tobytes()
    ds.save_as(tmp_path / "a.dcm", write_like_original=False)
    niv = pd.DataFrame({"degradacao": ["gama", "desfoque", "ruido"], "percentil": ["p25", "p10", "p90"],
                        "parametro": [1.25, 2.0, 6.0], "alvo": [100.0, 5.0, 3.0]})
    seg = lambda img: np.pad(np.ones((img.shape[0] - 100, img.shape[1] - 100), np.uint8), 50)   # noqa: E731
    linhas = uma_imagem(str(tmp_path / "a.dcm"), "pac/est/v1.dcm", niv, str(tmp_path), seg)
    assert [l["degradacao"] for l in linhas] == ["nenhuma", "gama", "desfoque", "ruido"]
    import cv2
    for l in linhas:
        assert cv2.imread(l["caminho"], cv2.IMREAD_GRAYSCALE).shape == (320, 320)
    assert np.isfinite([l["valor_medido"] for l in linhas[1:]]).all()
