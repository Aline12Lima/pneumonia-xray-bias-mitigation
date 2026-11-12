"""Testes das predições do teste: trava, imagens, covariáveis do PadChest e modo simulado."""
import json
import os
import subprocess

import cv2
import numpy as np
import pandas as pd
import pytest

import src.predicoes.teste as T
from src.cache.covariaveis import MEDIDAS
from src.divisao.listas import sha256_ids


def _git(repo, *a):
    subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True,
                   env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t"})


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    (r / "a.txt").write_text("a")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "c1")
    _git(r, "tag", "analise-v1")
    return r


def _arquivos(tmp_path, ids, sha=None, tag="analise-v1"):
    trava, hl = tmp_path / "destrancamento.json", tmp_path / "listas.json"
    h = sha or sha256_ids(ids)
    trava.write_text(json.dumps({"tag_codigo": tag, "listas_sha256": {"rsna": h}}))
    hl.write_text(json.dumps({"bases": {"rsna": {"listas": {"teste": {"sha256": sha256_ids(ids)}}}}}))
    return str(trava), str(hl)


def test_trava_sem_arquivo(tmp_path, repo):
    with pytest.raises(PermissionError, match="trancado"):
        T.conferir_trava("rsna", ["a"], trava=str(tmp_path / "nao.json"), repo=str(repo))


def test_trava_aberta_com_tag_e_hash(tmp_path, repo):
    trava, hl = _arquivos(tmp_path, ["a", "b"])
    d = T.conferir_trava("rsna", ["b", "a"], trava=trava, hashes_listas=hl, repo=str(repo))
    assert d["tag_codigo"] == "analise-v1"


def test_trava_recusa_hash_diferente(tmp_path, repo):
    trava, hl = _arquivos(tmp_path, ["a", "b"])
    with pytest.raises(PermissionError, match="SHA-256"):
        T.conferir_trava("rsna", ["a", "c"], trava=trava, hashes_listas=hl, repo=str(repo))


def test_trava_recusa_tag_inexistente_ou_fora_da_historia(tmp_path, repo):
    trava, hl = _arquivos(tmp_path, ["a"], tag="nao-existe")
    with pytest.raises(PermissionError, match="não existe"):
        T.conferir_trava("rsna", ["a"], trava=trava, hashes_listas=hl, repo=str(repo))
    _git(repo, "checkout", "-q", "--orphan", "outro")
    _git(repo, "commit", "-q", "-m", "c2")
    trava, hl = _arquivos(tmp_path, ["a"])
    with pytest.raises(PermissionError, match="não descende"):
        T.conferir_trava("rsna", ["a"], trava=trava, hashes_listas=hl, repo=str(repo))


def test_imagens_separa_incertos(tmp_path):
    arq = tmp_path / "teste.csv"
    pd.DataFrame({"id": ["p1/s1/v.dcm", "x", "y"], "deid_patient_id": ["1", "2", "3"],
                  "rotulo": ["positivo", "negativo", "incerto"]}).to_csv(arq, index=False)
    pr, inc = T.imagens("plus", str(arq), cache="/c")
    assert pr["rotulo"].tolist() == [1, 0] and inc["rotulo"].tolist() == [1]
    assert pr["caminho"].iloc[0] == "/c/p1__s1__v.png" and set(pr["fonte"]) == {"plus"}


def test_covariaveis_padchest():
    rec = pd.DataFrame({"ImageID": ["a.png", "b.png"], "StudyID": ["s1", "s2"], "Projection": ["AP_horizontal", "PA"],
                        "idade": [70.0, 33.0], "PatientSex_DICOM": ["F", "M"], "Manufacturer_DICOM": ["PhilipsMedicalSystems", None]})
    med = pd.DataFrame({"ImageID": ["b.png", "a.png"], **{c: [1.0, 2.0] for c in MEDIDAS}, "q_brilho_janela": [9, 9]})
    df = T.covariaveis_padchest(rec, med, ["b.png", "a.png"]).set_index("id")
    assert df.loc["a.png", "posicao"] == "AP" and df.loc["b.png", "posicao"] == "PA"
    assert df.loc["a.png", "q_brilho"] == 2.0 and "q_brilho_janela" not in df.columns
    assert pd.isna(df.loc["b.png", "fabricante"]) and df.loc["a.png", "idade"] == 70
    with pytest.raises(RuntimeError, match="sem medidas"):
        T.covariaveis_padchest(rec, med.iloc[:1], ["a.png", "b.png"])


def _base_falsa(tmp_path, base="rsna", n=6):
    cache = tmp_path / "cache"
    cache.mkdir()
    ids = [f"img{i}" for i in range(n)]
    regs = []
    for i in ids:
        cv2.imwrite(str(cache / f"{i}.png"), np.full((320, 320), 100, np.uint8))
        regs.append({"id": i, "sha256": __import__("hashlib").sha256(open(cache / f"{i}.png", "rb").read()).hexdigest()})
    pd.DataFrame(regs).to_csv(tmp_path / "hashes.csv", index=False)
    lista = tmp_path / "validacao.csv"
    pd.DataFrame({"id": ids, "patientId": [f"p{k // 2}" for k in range(n)],
                  "rotulo": ["positivo", "negativo"] * (n // 2)}).to_csv(lista, index=False)
    cov = pd.DataFrame({"id": ids, "id_estudo": ids, "posicao": "PA", "idade": 50, "sexo": "F", "fabricante": None,
                        **{c: 0.5 for c in MEDIDAS}})
    return cache, lista, cov


def test_modo_simulado_de_ponta_a_ponta(tmp_path, monkeypatch):
    cache, lista, cov = _base_falsa(tmp_path)
    monkeypatch.setattr(T, "lista", lambda b, s: (str(lista), "validacao"))
    monkeypatch.setattr(T, "CACHE", {"rsna": str(cache)})
    monkeypatch.setattr(T, "HASHES_CACHE", str(tmp_path / "hashes.csv"))
    modelos = tmp_path / "modelos"
    from src.predicoes.pos_treino import SEMENTES, TAXA
    for m in TAXA:
        for s in SEMENTES:
            d = modelos / f"{m}_s{s}_lr{TAXA[m]:g}"
            d.mkdir(parents=True)
            (d / "resumo.json").write_text(json.dumps({"modelo": m, "semente": s, "base": "rsna"}))

    def pred_falso(pasta, img):
        z = np.linspace(-2, 2, len(img))
        return pd.DataFrame({"id": img["id"], "paciente": img["paciente"], "fonte": img["fonte"], "rotulo": img["rotulo"],
                             "logit_corrigido": z, "prob": 1 / (1 + np.exp(-z))})
    r = T.rodar("rsna", True, str(modelos), str(tmp_path / "saida"), predizer_fn=pred_falso, cov_fn=lambda b, i: cov)
    assert r["execucoes"] == 15 and not r["problemas_schema"] and r["particao"] == "validacao"
    tab = pd.read_csv(tmp_path / "saida" / "rsna" / "tabela_teste.csv.gz")
    assert len(tab) == 15 * 6 and set(tab["particao"]) == {"validacao"}
    assert not (tmp_path / "saida" / "rsna" / "tabela_teste_incertos.csv.gz").exists()


def test_padchest_simulado_nao_usa_modelo(tmp_path, monkeypatch):
    cache, lista, cov = _base_falsa(tmp_path, "padchest")
    monkeypatch.setattr(T, "lista", lambda b, s: (str(lista), "teste"))
    monkeypatch.setattr(T, "CACHE", {"padchest": str(cache)})
    monkeypatch.setattr(T, "HASHES_CACHE", str(tmp_path / "hashes.csv"))

    def proibido(*a):
        raise AssertionError("modelo aplicado ao teste antes do destrancamento")
    r = T.rodar("padchest", True, str(tmp_path / "sem_modelos"), str(tmp_path / "saida"), predizer_fn=proibido,
                cov_fn=lambda b, i: cov)
    assert r["sem_modelo"] and r["execucoes"] == 1 and not r["problemas_schema"]


def test_sem_simulado_exige_trava(tmp_path, monkeypatch):
    cache, lista, cov = _base_falsa(tmp_path)
    monkeypatch.setattr(T, "lista", lambda b, s: (str(lista), "teste"))
    monkeypatch.setattr(T, "CACHE", {"rsna": str(cache)})
    monkeypatch.setattr(T, "TRAVA", str(tmp_path / "nao_existe.json"))
    monkeypatch.setattr(T.conferir_trava, "__defaults__", (str(tmp_path / "nao_existe.json"), T.HASHES_LISTAS, T.RAIZ_REPO))
    with pytest.raises(PermissionError):
        T.rodar("rsna", False, str(tmp_path), str(tmp_path / "saida"), cov_fn=lambda b, i: cov)


def test_conversao_janela_usa_cache_proprio(tmp_path, monkeypatch):
    cache, lista, cov = _base_falsa(tmp_path, "padchest")
    vistos = []
    monkeypatch.setattr(T, "lista", lambda b, s: (vistos.append(b), (str(lista), "teste"))[1])
    monkeypatch.setattr(T, "CACHE", {"padchest": "/nao/usar", "padchest_janela": str(cache)})
    monkeypatch.setattr(T, "HASHES_CACHE", str(tmp_path / "hashes.csv"))
    r = T.rodar("padchest", True, str(tmp_path), str(tmp_path / "saida"), predizer_fn=None,
                cov_fn=lambda b, i: cov, conversao="janela")
    assert vistos == ["padchest_janela"] and r["conversao"] == "janela" and not r["problemas_schema"]
    assert (tmp_path / "saida" / "padchest_janela" / "tabela_teste.csv.gz").exists()
    with pytest.raises(ValueError):
        T.rodar("rsna", True, str(tmp_path), str(tmp_path / "s2"), cov_fn=lambda b, i: cov, conversao="janela")
