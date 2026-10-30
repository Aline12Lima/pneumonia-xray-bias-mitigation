"""Testes do carregador do cache (F5, passo 5.2), com um cache sintético numa pasta temporária."""
import hashlib

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.treino.carregador import ImagensCache, ler_particao, nome_cache, resumo

AUM = {"escala": [0.9, 1.0], "rotacao_graus": 5}


def _cache(tmp_path):
    raiz = tmp_path / "out"
    cache = raiz / "plus" / "passada" / "c320_8bits"
    listas = raiz / "plus" / "listas"
    cache.mkdir(parents=True); listas.mkdir(parents=True)
    rng = np.random.default_rng(0)
    linhas, hashes = [], []
    for k, rot in enumerate(["positivo", "negativo", "negativo", "incerto", "positivo", "negativo"]):
        ident = f"patient{k}/study1/view1_frontal.dcm"
        arq = cache / nome_cache(ident)
        cv2.imwrite(str(arq), rng.integers(0, 256, (320, 320), dtype=np.uint8))
        linhas.append({"id": ident, "deid_patient_id": f"patient{k}", "rotulo": rot})
        hashes.append({"id": ident, "sha256": hashlib.sha256(arq.read_bytes()).hexdigest()})
    pd.DataFrame(linhas).to_csv(listas / "treino_subamostra.csv", index=False)
    pd.DataFrame(linhas[:3]).to_csv(listas / "validacao.csv", index=False)
    pd.DataFrame(hashes).to_csv(raiz / "plus" / "cache_hashes.csv", index=False)
    return str(raiz), cache


def test_rotulo_principal_tira_incertos(tmp_path):
    raiz, _ = _cache(tmp_path)
    tr = ler_particao("plus", "treino", raiz=raiz)
    assert len(tr) == 5 and set(tr["rotulo"]) == {0, 1}
    assert resumo(tr) == {"imagens": 5, "pacientes": 5, "positivos": 2, "prevalencia": 0.4}


def test_teste_e_recusado():
    with pytest.raises(ValueError):
        ler_particao("plus", "teste")


def test_hash_errado_para_o_carregador(tmp_path):
    raiz, cache = _cache(tmp_path)
    arq = sorted(cache.iterdir())[0]
    cv2.imwrite(str(arq), np.zeros((320, 320), np.uint8))
    with pytest.raises(RuntimeError):
        ler_particao("plus", "treino", raiz=raiz)


def test_tensor_e_determinismo(tmp_path):
    raiz, _ = _cache(tmp_path)
    tr = ler_particao("plus", "treino", raiz=raiz)
    ds = ImagensCache(tr, AUM, semente=3)
    x, y = ds[1]
    from src.treino.dados import DP_IMAGENET, MEDIA_IMAGENET
    cru = x * DP_IMAGENET + MEDIA_IMAGENET                                     # desfaz a normalização
    assert x.shape == (3, 320, 320) and y.shape == (1,)
    assert torch.allclose(cru[0], cru[1], atol=1e-5) and torch.allclose(cru[1], cru[2], atol=1e-5)
    assert torch.equal(ds[1][0], ImagensCache(tr, AUM, semente=3)[1][0])        # mesma semente e época
    ds.definir_epoca(1)
    assert not torch.equal(ds[1][0], ImagensCache(tr, AUM, semente=3)[1][0])    # época muda o aumento
    ds.definir_epoca(0)
    a = next(iter(DataLoader(ds, batch_size=5, num_workers=0)))[0]
    b = next(iter(DataLoader(ds, batch_size=5, num_workers=2)))[0]
    assert torch.equal(a, b)                                                    # workers não mudam nada


def test_sem_aumento_e_identico_a_imagem(tmp_path):
    from src.treino.dados import DP_IMAGENET, MEDIA_IMAGENET
    raiz, _ = _cache(tmp_path)
    tr = ler_particao("plus", "validacao", raiz=raiz)
    x, _ = ImagensCache(tr, None)[0]
    img = cv2.imread(tr["caminho"][0], cv2.IMREAD_GRAYSCALE)
    volta = ((x * DP_IMAGENET + MEDIA_IMAGENET)[0] * 255).round().numpy().astype(np.uint8)
    assert np.array_equal(volta, img)


def test_codigo_de_treino_nao_le_teste_trancado():
    import pathlib
    for arq in pathlib.Path("src/treino").glob("*.py"):
        assert "teste_trancado" not in arq.read_text(encoding="utf-8"), arq
