"""Testes do recorte do PadChest refeito a partir do CSV (mesma regra do Bloco P0)."""
import pandas as pd

from src.padchest.recorte import hash_congelado, recortar


def _csv():
    base = {"StudyDate_DICOM": 20150101, "PatientBirth": 1950.0, "PatientSex_DICOM": "M", "MethodLabel": "Physician"}
    linhas = [
        # estudo 1: PA e AP → fica a PA
        {"ImageID": "1b.png", "StudyID": "s1", "PatientID": 1, "ImageDir": 0, "Projection": "AP", "Labels": "['pneumonia']"},
        {"ImageID": "1a.png", "StudyID": "s1", "PatientID": 1, "ImageDir": 0, "Projection": "PA", "Labels": "['pneumonia']"},
        # fora do pacote
        {"ImageID": "2.png", "StudyID": "s2", "PatientID": 2, "ImageDir": 20, "Projection": "PA", "Labels": "['normal']"},
        # lateral
        {"ImageID": "3.png", "StudyID": "s3", "PatientID": 3, "ImageDir": 54, "Projection": "L", "Labels": "['normal']"},
        # exclude
        {"ImageID": "4.png", "StudyID": "s4", "PatientID": 4, "ImageDir": 1, "Projection": "PA", "Labels": "['exclude']"},
        # criança
        {"ImageID": "5.png", "StudyID": "s5", "PatientID": 5, "ImageDir": 1, "Projection": "PA", "Labels": "['normal']",
         "PatientBirth": 2010.0},
        # atípica, AP deitado, pacote 54
        {"ImageID": "6.png", "StudyID": "s6", "PatientID": 6, "ImageDir": 54, "Projection": "AP_horizontal",
         "Labels": "['atypical pneumonia']"},
    ]
    return pd.DataFrame([{**base, **l} for l in linhas])


def test_regra_do_bloco_p0():
    rec, etapas = recortar(_csv())
    assert sorted(rec["ImageID"]) == ["1a.png", "6.png"]
    r = rec.set_index("ImageID")
    assert bool(r.loc["1a.png", "pos"]) and not bool(r.loc["6.png", "pos"]) and bool(r.loc["6.png", "pos_atipica"])
    assert r.loc["1a.png", "idade"] == 65
    assert [n for _, n in etapas] == [7, 6, 5, 4, 3, 2]


def test_hash_congelado_registrado():
    ref = hash_congelado()
    assert ref["imagens"] == 24770 and ref["positivos"] == 796 and len(ref["sha256"]) == 64
