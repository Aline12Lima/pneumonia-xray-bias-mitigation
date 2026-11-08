"""Testes do cache de 320 px do PadChest (sem rede e sem Redivis)."""
import zipfile

import cv2
import numpy as np
import pandas as pd

from src.cache.hashes import calcular, nome_cache
from src.imagem.conversao import para_8bits, redimensionar
from src.padchest.cache320 import alvo, converter, faltando, processar_pacote


def _png16(semente, h=600, w=500):
    rng = np.random.default_rng(semente)
    return (np.linspace(500, 3500, w)[None, :] + rng.normal(0, 60, (h, w))).clip(0, 4095).astype(np.uint16) * 16


def _zip(tmp_path, ids):
    zf = tmp_path / "0.zip"
    with zipfile.ZipFile(zf, "w") as z:
        for k, i in enumerate(ids):
            z.writestr(f"0/{i}", cv2.imencode(".png", _png16(k))[1].tobytes())
    return str(zf)


def test_nome_do_padchest_sem_extensao_dupla():
    assert nome_cache("123_abc.png") == "123_abc.png"
    assert nome_cache("patient1/study1/view1_frontal.dcm") == "patient1__study1__view1_frontal.png"


def test_conversao_igual_a_das_medidas():
    g = _png16(0)
    c = converter(g)
    assert c.shape == (320, 320) and c.dtype == np.uint8
    assert np.array_equal(c, redimensionar(para_8bits(g), 320))


def test_grava_e_retoma(tmp_path):
    ids = ["a.png", "b.png", "c.png"]
    zf = _zip(tmp_path, ids)
    cache = tmp_path / "c320"
    falhas = processar_pacote(0, zf, ids + ["fora_do_zip.png"], str(cache))
    assert [f["id"] for f in falhas] == ["fora_do_zip.png"]
    assert sorted(p.name for p in cache.iterdir()) == ids              # nenhum .tmp sobrando
    lido = cv2.imread(str(cache / "a.png"), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(lido, converter(_png16(0)))
    # retomada: só o que falta
    df = pd.DataFrame({"id": ids + ["d.png"], "ImageDir": 0})
    assert faltando(df, str(cache))["id"].tolist() == ["d.png"]
    # e os hashes do conjunto saem completos
    lista = tmp_path / "teste.csv"
    pd.DataFrame({"id": ids, "PatientID": 1, "rotulo": "negativo"}).to_csv(lista, index=False)
    _, resumo = calcular(str(cache), {"teste": str(lista)})
    assert resumo["teste"]["faltando"] == 0


def test_alvo_le_so_ids_e_pacote(tmp_path):
    lista, rec = tmp_path / "teste.csv", tmp_path / "recorte.csv"
    pd.DataFrame({"id": ["a.png", "b.png"], "PatientID": [1, 2], "rotulo": ["positivo", "negativo"]}).to_csv(lista, index=False)
    pd.DataFrame({"ImageID": ["a.png", "b.png", "z.png"], "ImageDir": [3, 54, 1]}).to_csv(rec, index=False)
    df = alvo(str(lista), str(rec))
    assert list(df.columns) == ["id", "ImageDir"] and df["ImageDir"].tolist() == [3, 54]
