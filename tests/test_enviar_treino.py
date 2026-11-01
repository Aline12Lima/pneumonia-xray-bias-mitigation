"""O envio para o dataset de treino nunca inclui imagens nem listas de teste."""
import os

import pandas as pd
import pytest

from src.cache.enviar_treino import selecionar


def _out(tmp_path):
    raiz = tmp_path / "out"
    (raiz / "plus" / "listas").mkdir(parents=True)
    (raiz / "plus" / "listas_teste_trancado").mkdir(parents=True)
    cache = raiz / "plus" / "passada" / "c320_8bits"
    cache.mkdir(parents=True)
    linhas = []
    for k, part in enumerate(["treino_subamostra", "treino_subamostra", "validacao", "teste", "teste"]):
        arq = f"patient{k}__study1__view1_frontal.png"
        (cache / arq).write_bytes(b"x" * (k + 1))
        linhas.append({"id": f"patient{k}/study1/view1_frontal.dcm", "particao": part, "arquivo": arq,
                       "bytes": k + 1, "sha256": f"h{k}"})
    pd.DataFrame(linhas).to_csv(raiz / "plus" / "cache_hashes.csv", index=False)
    for nome in ["treino_subamostra.csv", "validacao.csv"]:
        pd.DataFrame({"id": ["a"], "deid_patient_id": ["p"], "rotulo": ["negativo"]}).to_csv(raiz / "plus" / "listas" / nome, index=False)
    pd.DataFrame({"id": ["t"]}).to_csv(raiz / "plus" / "listas_teste_trancado" / "teste.csv", index=False)
    return str(raiz)


def test_so_treino_e_validacao(tmp_path):
    raiz = _out(tmp_path)
    pngs, listas = selecionar("plus", raiz=raiz, pasta_listas=str(tmp_path / "envio"))
    assert sorted(os.path.basename(p) for p in pngs) == [f"patient{k}__study1__view1_frontal.png" for k in range(3)]
    assert sorted(os.path.basename(p) for p in listas) == ["cache_hashes.csv", "treino_subamostra.csv", "validacao.csv"]
    h = pd.read_csv(tmp_path / "envio" / "cache_hashes.csv")
    assert set(h["particao"]) == {"treino_subamostra", "validacao"}
    assert not any("teste" in os.path.basename(p) for p in pngs + listas)


def test_para_se_faltar_arquivo(tmp_path):
    raiz = _out(tmp_path)
    h = pd.read_csv(os.path.join(raiz, "plus", "cache_hashes.csv"))
    h.loc[0, "sha256"] = None
    h.to_csv(os.path.join(raiz, "plus", "cache_hashes.csv"), index=False)
    with pytest.raises(RuntimeError):
        selecionar("plus", raiz=raiz, pasta_listas=str(tmp_path / "envio"))
