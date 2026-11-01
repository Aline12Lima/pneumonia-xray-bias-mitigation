"""Sanidade do treino (passo 5.5) num cache sintético (CPU, sem pesos ImageNet)."""
import hashlib
import json

import cv2
import numpy as np
import pandas as pd
import yaml

from src.divisao.listas import sha256_ids
from src.treino.carregador import nome_cache
from src.treino.sanidade import conferir_pacientes, sobreajustar_lote
from src.treino.treinar import treinar


def _cache(tmp_path, paciente_repetido=False):
    raiz = tmp_path / "dados"
    cache = raiz / "plus" / "passada" / "c320_8bits"
    listas = raiz / "plus" / "listas"
    cache.mkdir(parents=True); listas.mkdir(parents=True)
    rng = np.random.default_rng(0)
    linhas, hashes = [], []
    for k in range(20):
        pos = k % 2 == 0
        img = rng.integers(60, 120, (320, 320)).astype(np.uint8)
        if pos:
            img[100:220, 100:220] = 230
        ident = f"patient{k}/study1/view1_frontal.dcm"
        cv2.imwrite(str(cache / nome_cache(ident)), img)
        linhas.append({"id": ident, "deid_patient_id": f"p{k}", "rotulo": "positivo" if pos else "negativo"})
        hashes.append({"id": ident, "sha256": hashlib.sha256((cache / nome_cache(ident)).read_bytes()).hexdigest()})
    if paciente_repetido:
        linhas[12]["deid_patient_id"] = "p0"                 # paciente do treino aparece na validação
    tr, va = pd.DataFrame(linhas[:12]), pd.DataFrame(linhas[12:])
    tr.to_csv(listas / "treino_subamostra.csv", index=False)
    va.to_csv(listas / "validacao.csv", index=False)
    pd.DataFrame(hashes).to_csv(raiz / "plus" / "cache_hashes.csv", index=False)
    ref = {"bases": {"plus": {"listas": {"treino_subamostra": {"sha256": sha256_ids(tr["id"])},
                                         "validacao": {"sha256": sha256_ids(va["id"])}}}}}
    (tmp_path / "listas.json").write_text(json.dumps(ref))
    return str(raiz), str(tmp_path / "listas.json")


def test_pacientes_aprovado_e_hash_das_listas(tmp_path):
    raiz, ref = _cache(tmp_path)
    r = conferir_pacientes(("plus",), raiz=raiz, listas_json=ref)
    assert r["aprovado"] and r["bases"]["plus"]["treino"]["sha256_confere"]


def test_pacientes_detecta_paciente_em_duas_particoes(tmp_path):
    raiz, ref = _cache(tmp_path, paciente_repetido=True)
    r = conferir_pacientes(("plus",), raiz=raiz, listas_json=ref)
    assert not r["aprovado"] and r["bases"]["plus"]["pacientes_em_duas_particoes"] == 1


def test_pacientes_detecta_lista_alterada_e_arquivo_de_teste(tmp_path):
    raiz, ref = _cache(tmp_path)
    d = json.load(open(ref)); d["bases"]["plus"]["listas"]["validacao"]["sha256"] = "0" * 64
    open(ref, "w").write(json.dumps(d))
    assert not conferir_pacientes(("plus",), raiz=raiz, listas_json=ref)["aprovado"]
    raiz, ref = _cache(tmp_path / "b")
    open(f"{raiz}/plus/listas/teste.csv", "w").write("id\n")
    assert not conferir_pacientes(("plus",), raiz=raiz, listas_json=ref)["aprovado"]


def test_sobreajuste_reduz_a_perda(tmp_path):
    raiz, _ = _cache(tmp_path)
    cfg = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))
    r = sobreajustar_lote(cfg, n=4, passos=8, taxa=1e-3, raiz=raiz, pesos_imagenet=False)
    assert r["positivos"] == 2 and r["perda_final"] < r["perda_inicial"]


def test_rotulos_embaralhados_mantem_prevalencia_e_validacao_real(tmp_path):
    raiz, _ = _cache(tmp_path)
    cfg = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))
    cfg["treino"]["lote"] = 4
    res = treinar("M1", 0, 1e-4, cfg, str(tmp_path / "t"), raiz=raiz, pesos_imagenet=False, epocas_max=1,
                  workers=0, rotulos_embaralhados=True)
    assert res["rotulos_embaralhados"] is True
    pred = pd.read_csv(tmp_path / "t" / "predicoes_validacao.csv")
    assert pred["rotulo"].tolist() == [1, 0] * 4             # a validação usa os rótulos reais
