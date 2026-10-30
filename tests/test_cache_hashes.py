"""Testes do hash do cache de 320 px (F5, passo 5.1)."""
import pandas as pd

from src.cache.hashes import calcular, nome_cache


def _montar(tmp_path):
    cache = tmp_path / "c320"
    cache.mkdir()
    ids = {"treino": ["patient1/study1/view1_frontal.dcm", "patient2/study1/view1_frontal.dcm"],
           "teste": ["patient3/study1/view1_frontal.dcm"]}
    listas = {}
    for part, lst in ids.items():
        for i in lst:
            (cache / nome_cache(i)).write_bytes(f"png de {i}".encode())
        listas[part] = str(tmp_path / f"{part}.csv")
        pd.DataFrame({"id": lst}).to_csv(listas[part], index=False)
    return cache, listas


def test_nome_igual_ao_da_passada():
    from src.passada.passada_completa import nome_cache as da_passada
    for i in ["patient1/study1/view1_frontal.dcm", "1.2.840.10008.1"]:
        assert nome_cache(i) == da_passada(i)


def test_hash_por_arquivo_e_por_particao(tmp_path):
    cache, listas = _montar(tmp_path)
    df, resumo = calcular(str(cache), listas)
    assert len(df) == 3 and (df["sha256"].str.len() == 64).all()
    assert resumo["treino"]["arquivos_no_cache"] == 2 and resumo["teste"]["faltando"] == 0
    # mudar um byte de um arquivo muda o hash da partição dele, e só dela
    (cache / nome_cache("patient1/study1/view1_frontal.dcm")).write_bytes(b"outro conteudo")
    _, depois = calcular(str(cache), listas)
    assert depois["treino"]["sha256_conjunto"] != resumo["treino"]["sha256_conjunto"]
    assert depois["teste"]["sha256_conjunto"] == resumo["teste"]["sha256_conjunto"]


def test_aponta_arquivo_faltando(tmp_path):
    cache, listas = _montar(tmp_path)
    (cache / nome_cache("patient3/study1/view1_frontal.dcm")).unlink()
    df, resumo = calcular(str(cache), listas)
    assert resumo["teste"]["faltando"] == 1
    assert (df["sha256"] == "").sum() == 1
