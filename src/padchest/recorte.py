"""Recorte e lista do PadChest refeitos a partir do CSV original (F6 › PadChest › Reconstrução).

Em 8/11/2026 o notebook ``00_padchest_auditoria`` foi apagado junto com o seu ``/out`` (recorte, lista
trancada e medidas). Este script refaz o recorte com a **mesma regra** do Bloco P0
(``notebooks/padchest/00_padchest_auditoria.ipynb``), agora sobre o CSV baixado do próprio link da
BIMCV, e gera a lista com ``src.divisao.listas``. A lista só é aceita se o SHA-256 dos IDs for igual ao
registrado no congelamento (``data/splits/listas_sha256.json``): isso prova que é a mesma lista.

Regra do recorte (D-025, F3 › PadChest › Recorte e Exclusões):
pacotes 0–11 e 54 → frontais (PA, AP, AP_horizontal) → idade 18 a 100 (ano do exame − ano de
nascimento) → sem rótulo "exclude" → uma frontal por estudo (PA > AP > AP deitado, depois ImageID).
A lista tira ainda o sexo diferente de M e F (``listas.gerar``).

Nada aqui olha predições; o rótulo é o do CSV, como no congelamento.

Uso (notebook do workflow ``padchest_teste``, CPU):
    python -m src.padchest.recorte
"""
from __future__ import annotations

import argparse
import ast
import json
import os

import pandas as pd

CSV_NOME = "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz"
CSV_LOCAL = f"/out/padchest/{CSV_NOME}"
RECORTE = "/out/padchest_recorte.csv"
HASHES = os.path.join(os.path.dirname(__file__), "..", "..", "data", "splits", "listas_sha256.json")
PACOTES = list(range(12)) + [54]
FRONTAIS = {"PA": 0, "AP": 1, "AP_horizontal": 2}           # também a prioridade dentro do estudo
COLUNAS = ["ImageID", "ImageDir", "StudyID", "PatientID", "idade", "PatientSex_DICOM", "Projection",
           "MethodLabel", "Manufacturer_DICOM", "BitsStored_DICOM", "Rows_DICOM", "Columns_DICOM",
           "WindowCenter_DICOM", "WindowWidth_DICOM", "PhotometricInterpretation_DICOM", "pos", "pos_atipica"]


def _lista(s) -> list[str]:
    try:
        return [str(x).strip().lower() for x in ast.literal_eval(s)]
    except Exception:
        return []


def recortar(df: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Mesma regra do Bloco P0. Devolve o recorte e as contagens de cada etapa."""
    df = df.copy()
    rot = df["Labels"].apply(_lista)
    df["pos"] = rot.apply(lambda r: "pneumonia" in r)
    df["pos_atipica"] = rot.apply(lambda r: "pneumonia" in r or "atypical pneumonia" in r)
    df["rot_exclude"] = rot.apply(lambda r: "exclude" in r)
    df["ImageDir"] = pd.to_numeric(df["ImageDir"], errors="coerce")
    ano = pd.to_numeric(df["StudyDate_DICOM"].astype(str).str[:4], errors="coerce")
    df["idade"] = ano - pd.to_numeric(df["PatientBirth"], errors="coerce")
    etapas = [("Todas as imagens", len(df))]
    x = df[df["ImageDir"].isin(PACOTES)];                  etapas.append(("Pacotes 0–11 e 54", len(x)))
    x = x[x["Projection"].isin(list(FRONTAIS))];           etapas.append(("Frontais", len(x)))
    x = x[x["idade"].between(18, 100)];                    etapas.append(("Idade de 18 a 100", len(x)))
    x = x[~x["rot_exclude"]];                              etapas.append(("Sem 'exclude'", len(x)))
    rec = (x.assign(p=x["Projection"].map(FRONTAIS)).sort_values(["StudyID", "p", "ImageID"])
           .drop_duplicates("StudyID", keep="first"))
    etapas.append(("Uma frontal por estudo", len(rec)))
    return rec[[c for c in COLUNAS if c in rec.columns]], etapas


def baixar_csv(link: str, destino: str = CSV_LOCAL) -> str:
    from src.padchest.segunda_passada import baixar, url_e_tamanho
    url, total = url_e_tamanho(link, CSV_NOME)
    if os.path.exists(destino) and os.path.getsize(destino) == total:
        return destino
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    baixar(url, total, destino, partes=1)
    return destino


def hash_congelado(hashes: str = HASHES) -> dict:
    return json.load(open(hashes, encoding="utf-8"))["bases"]["padchest"]["listas"]["teste"]


def main(argv=None) -> int:
    from src.divisao.listas import gerar
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    link = os.environ.get("PADCHEST_LINK")
    if not link:
        raise SystemExit("Defina PADCHEST_LINK no notebook (o link não fica no repositório).")
    arq = baixar_csv(link)
    df = pd.read_csv(arq, low_memory=False)
    print(f"CSV: {len(df)} linhas, {len(df.columns)} colunas")
    rec, etapas = recortar(df)
    for nome, n in etapas:
        print(f"  {nome:24s} {n}")
    rec.to_csv(RECORTE, index=False)
    print(f"Recorte: {len(rec)} imagens, {rec['PatientID'].nunique()} pacientes, {int(rec['pos'].sum())} positivos "
          f"→ {RECORTE}")
    r = gerar("padchest")["listas"]["teste"]
    ref = hash_congelado()
    print(f"Lista refeita:    {r['imagens']} imagens, {r['pacientes']} pacientes, {r['positivos']} positivos | "
          f"SHA-256 {r['sha256']}")
    print(f"Lista congelada:  {ref['imagens']} imagens, {ref['pacientes']} pacientes, {ref['positivos']} positivos | "
          f"SHA-256 {ref['sha256']}")
    igual = r["sha256"] == ref["sha256"]
    print("RESULTADO:", "LISTA IDÊNTICA À CONGELADA" if igual else "DIFERENTE — não seguir; me mande esta saída")
    return 0 if igual else 1


if __name__ == "__main__":
    raise SystemExit(main())
