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


def test_ancoragem_inclui_a_imagem_natural():
    from src.perturbations.piloto import ancoragem
    linhas = []
    for i in range(5):
        linhas.append({"id": i, "tipo": "natural", "parametro": 0.0, "q_nitidez_orig": 300.0})
        for p, v in ((0.5, 100.0), (1, 50.0)):
            linhas.append({"id": i, "tipo": "desfoque", "parametro": p, "q_nitidez_orig": v})
    nat = pd.DataFrame({"q_nitidez": np.linspace(150, 400, 100)})
    a = ancoragem(pd.DataFrame(linhas), nat)
    assert a["parametro"].notna().all() and (a["parametro"] < 0.5).all()   # entre a natural e o 1º passo


def test_uma_imagem_aceita_matriz_do_padchest(tmp_path):
    import cv2
    rng = np.random.default_rng(2)
    g = (np.linspace(500, 3500, 500)[None, :] + rng.normal(0, 50, (600, 500))).clip(0, 4095).astype(np.uint16) * 16
    niv = pd.DataFrame({"degradacao": ["brilho", "resolucao"], "percentil": ["p75", "p10"],
                        "parametro": [10.0, 1.2], "alvo": [120.0, 50.0]})
    seg = lambda img: np.pad(np.ones((img.shape[0] - 100, img.shape[1] - 100), np.uint8), 50)   # noqa: E731
    linhas = uma_imagem(g, "123_abc.png", niv, str(tmp_path), seg)
    assert [l["nivel"] for l in linhas] == ["0", "p75", "p10"]
    assert all(cv2.imread(l["caminho"], cv2.IMREAD_GRAYSCALE).shape == (320, 320) for l in linhas)
    assert linhas[0]["caminho"].endswith("123_abc__nenhuma__0.png")


def test_teste_do_eixo_b_exige_destrancamento(tmp_path, monkeypatch):
    import src.predicoes.teste as T
    from src.perturbations.eixo_b import tabela_da_particao
    arq = tmp_path / "teste.csv"
    pd.DataFrame({"id": ["a", "b"], "patientId": ["1", "2"], "rotulo": ["positivo", "negativo"]}).to_csv(arq, index=False)
    monkeypatch.setattr(T, "lista", lambda b, simulado: (str(arq), "teste"))
    with pytest.raises(PermissionError, match="trancado"):
        tabela_da_particao("rsna", "teste")


def test_ancoragem_da_f8_vai_para_outra_pasta():
    from src.perturbations.eixo_b import arquivo_ancoragem
    from src.perturbations.piloto import GRADE, GRADE_F8
    assert arquivo_ancoragem("plus", "validacao") == "/out/plus/eixo_b_piloto/ancoragem.csv"
    assert arquivo_ancoragem("padchest", "teste") == "/out/padchest/eixo_b_ancoragem_f8/ancoragem.csv"
    for tipo in GRADE:                                  # a grade da F8 contém a do ensaio e passos menores
        assert set(GRADE[tipo]) <= set(GRADE_F8[tipo])
    assert min(GRADE_F8["resolucao"]) < 1.5 and min(GRADE_F8["desfoque"]) < 0.5


def test_piloto_sorteia_dois_fabricantes_mais_frequentes():
    from src.perturbations.piloto import sortear as sortear_piloto
    nat = pd.DataFrame({"id": [str(i) for i in range(300)],
                        "fabricante": ["Philips"] * 200 + ["ImagingDynamics"] * 80 + ["Outro"] * 20})
    a = sortear_piloto(nat)
    assert a["fabricante"].value_counts().to_dict() == {"Philips": 25, "ImagingDynamics": 25}


def test_prevalencia_do_teste_pelas_contagens_congeladas():
    from src.analises.eixo_b import prevalencia
    assert abs(prevalencia("padchest", "teste") - 796 / 24770) < 1e-12
    assert abs(prevalencia("plus", "teste") - 690 / (35457 - 3065)) < 1e-12
