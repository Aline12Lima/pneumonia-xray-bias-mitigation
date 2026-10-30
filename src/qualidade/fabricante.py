"""Nome do fabricante unificado (as análises de qualidade são feitas dentro do fabricante).

O DICOM grava o mesmo fabricante de vários jeitos ("GE Healthcare" com aspas, "GE MEDICAL SYSTEMS",
"AGFA", "Agfa-Gevaert AG", ...). Esta função leva todos para um nome só por empresa.
"""
from __future__ import annotations

import pandas as pd

_GRUPOS = [("agfa", "Agfa"), ("imaging dynamics", "Imaging Dynamics"), ("imagingdynamics", "Imaging Dynamics"),
           ("philips", "Philips"), ("varian", "Varian"), ("siemens", "Siemens"), ("carestream", "Carestream"),
           ("fuji", "Fujifilm"), ("kodak", "Kodak"), ("canon", "Canon"), ("toshiba", "Toshiba")]


def normalizar_fabricante(nome) -> str:
    """'"GE Healthcare"' -> 'GE'; 'Agfa-Gevaert AG' -> 'Agfa'; vazio -> 'Desconhecido'."""
    if nome is None or (isinstance(nome, float) and pd.isna(nome)):
        return "Desconhecido"
    t = str(nome).strip().strip('"').strip("'").strip().lower()
    if not t or t == "nan":
        return "Desconhecido"
    if t == "ge" or t.startswith("ge ") or "general electric" in t:
        return "GE"
    for chave, grupo in _GRUPOS:
        if chave in t:
            return grupo
    return str(nome).strip().strip('"').strip()
