"""Validação da tabela de predições contra configs/schema_predicoes.yaml (protocolo v1.0 §7.4).

Uso:
    from src.predicoes.schema import validar
    erros = validar(df)            # lista vazia = tabela válida
    validar(df, levantar=True)     # levanta ValueError se houver erro
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

SCHEMA_PADRAO = Path(__file__).resolve().parents[2] / "configs" / "schema_predicoes.yaml"

_TIPOS = {
    "texto": pd.api.types.is_string_dtype,
    "inteiro": pd.api.types.is_integer_dtype,
    "real": pd.api.types.is_float_dtype,
    "booleano": pd.api.types.is_bool_dtype,
}


def carregar_schema(caminho: str | Path = SCHEMA_PADRAO) -> dict:
    with open(caminho, encoding="utf-8") as f:
        return yaml.safe_load(f)


def validar(df: pd.DataFrame, schema: dict | None = None, levantar: bool = False) -> list[str]:
    schema = schema or carregar_schema()
    erros: list[str] = []
    cols = schema["colunas"]

    faltam = [c for c in cols if c not in df.columns]
    if faltam:
        erros.append(f"colunas faltando: {faltam}")

    prefixo = schema.get("prefixo_qualidade", "q_")
    extras = [c for c in df.columns if c not in cols and not c.startswith(prefixo)]
    if extras:
        erros.append(f"colunas fora do schema (medidas de qualidade usam o prefixo '{prefixo}'): {extras}")

    for nome, regra in cols.items():
        if nome not in df.columns:
            continue
        s = df[nome]
        nulos = int(s.isna().sum())
        if nulos and not regra.get("nulo", False):
            erros.append(f"{nome}: {nulos} valores vazios (não permitido)")
        v = s.dropna()
        if v.empty:
            continue
        checa_tipo = _TIPOS[regra["tipo"]]
        if not checa_tipo(v):
            if regra["tipo"] == "real" and pd.api.types.is_integer_dtype(v):
                pass  # inteiro também serve como real
            else:
                erros.append(f"{nome}: tipo {v.dtype}, esperado {regra['tipo']}")
                continue
        if "valores" in regra:
            fora = sorted(set(v.unique()) - set(regra["valores"]), key=str)
            if fora:
                erros.append(f"{nome}: valores fora da lista {regra['valores']}: {fora[:5]}")
        if "min" in regra and (v < regra["min"]).any():
            erros.append(f"{nome}: {int((v < regra['min']).sum())} valores abaixo de {regra['min']}")
        if "max" in regra and (v > regra["max"]).any():
            erros.append(f"{nome}: {int((v > regra['max']).sum())} valores acima de {regra['max']}")

    for c in df.columns:
        if c.startswith(prefixo) and not pd.api.types.is_numeric_dtype(df[c]):
            erros.append(f"{c}: medida de qualidade precisa ser numérica (tipo {df[c].dtype})")

    chave = [c for c in schema.get("unicidade", []) if c in df.columns]
    if chave and len(chave) == len(schema["unicidade"]):
        dup = int(df.duplicated(chave).sum())
        if dup:
            erros.append(f"{dup} linhas repetidas para a chave {chave}")

    if levantar and erros:
        raise ValueError("tabela de predições inválida:\n- " + "\n- ".join(erros))
    return erros
