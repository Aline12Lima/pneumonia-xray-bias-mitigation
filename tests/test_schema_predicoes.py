"""Testes do schema da tabela de predições (rodar com: python -m pytest tests/)."""
import pandas as pd

from src.predicoes.schema import validar


def tabela_valida(n=4):
    return pd.DataFrame({
        "id_imagem": [f"img{i}" for i in range(n)],
        "id_paciente": [f"pac{i // 2}" for i in range(n)],
        "id_estudo": [f"est{i}" for i in range(n)],
        "fonte": ["rsna"] * n,
        "particao": ["validacao"] * n,
        "modelo": ["M2"] * n,
        "semente": [0] * n,
        "id_execucao": ["M2_s0_lr1e-4"] * n,
        "degradacao": ["nenhuma"] * n,
        "nivel": ["0"] * n,
        "rotulo": [0, 1] * (n // 2),
        "logit": [-2.0, 1.5] * (n // 2),
        "probabilidade": [0.12, 0.82] * (n // 2),
        "probabilidade_temperatura": [None] * n,
        "posicao": ["AP", "PA"] * (n // 2),
        "idade": [40, 65] * (n // 2),
        "sexo": ["F", "M"] * (n // 2),
        "fabricante": [None] * n,
        "q_brilho": [0.4, 0.5] * (n // 2),
    }).astype({"probabilidade_temperatura": "float64", "fabricante": "object"})


def test_tabela_valida_passa():
    assert validar(tabela_valida()) == []


def test_coluna_faltando():
    erros = validar(tabela_valida().drop(columns=["sexo"]))
    assert any("faltando" in e for e in erros)


def test_probabilidade_fora_de_0_1():
    df = tabela_valida()
    df.loc[0, "probabilidade"] = 1.2
    assert any("probabilidade" in e and "acima" in e for e in validar(df))


def test_valor_fora_da_lista():
    df = tabela_valida()
    df.loc[0, "fonte"] = "nih"
    assert any(e.startswith("fonte") for e in validar(df))


def test_linha_repetida():
    df = pd.concat([tabela_valida(), tabela_valida().iloc[[0]]], ignore_index=True)
    assert any("repetidas" in e for e in validar(df))


def test_coluna_extra_sem_prefixo():
    df = tabela_valida()
    df["brilho"] = 0.3
    assert any("fora do schema" in e for e in validar(df))
