"""Predições do teste (F6, passo 6.8; protocolo §5, §7.4), com trava até o destrancamento (6.7).

Roda no notebook **de cada base** (Plus, RSNA, PadChest), porque o cache de 320 px do teste nunca
sai de lá. Precisa da GPU e dos 15 modelos baixados do dataset ``modelos``
(``python -m src.cache.modelos --baixar`` → ``/out/modelos/<execução>/``).

Para cada uma das 15 execuções (M1, M2, M3 × sementes 0 a 4), aplica o modelo às imagens da lista
trancada da base, com a mesma ``predizer`` da validação (correção de prior do modelo; M3: 3c nas fontes
de treino e a correção da mistura no PadChest; temperatura de ``pos_treino.json``), junta as
covariáveis da base e valida contra o schema. Grava, em ``/out/teste/<base>/``:

- ``tabela_teste.csv.gz``: regra primária de rótulo (positivo = 1, negativo = 0; incerto fora);
- ``tabela_teste_incertos.csv.gz`` (só Plus): as imagens "incerto", com rótulo 1, para a análise de
  sensibilidade de D-018 (incerto = 1). Juntar as duas dá a tabela da sensibilidade;
- ``resumo_teste.json``: contagens, commit, tag e hora (sem IDs).

Nada de métrica é calculado aqui: as métricas saem só do código de análise congelado na tag (6.6).

**Trava.** Sem ``--simulado``, o script só roda se ``docs/destrancamento.json`` existir no repositório
(criado no passo 6.7), se a tag registrada nele existir e for ancestral do commit atual, e se o SHA-256
da lista trancada da base for igual ao do arquivo e ao de ``data/splits/listas_sha256.json``.

**Modo simulado** (antes da tag, passo 6.5): o mesmo caminho, sem trava, sobre a **validação** do Plus
e do RSNA, gravando em ``/out/teste_simulado/<base>/`` e nunca enviando nada. No PadChest, que não tem
validação, o modo simulado roda **sem modelo**: monta a tabela de imagens, confere os hashes do cache,
junta as covariáveis e valida o schema com logit 0. Nenhum modelo toca o teste antes de 6.7.

Uso (notebook da base):
    python -m src.predicoes.teste --base plus --simulado        # antes da tag (GPU)
    python -m src.predicoes.teste --base padchest --simulado    # antes da tag (CPU basta)
    python -m src.predicoes.teste --base plus                    # depois de 6.7 (GPU)
    python -m src.predicoes.teste --base plus --enviar           # idem, e envia ao dataset resultados
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import subprocess

import numpy as np
import pandas as pd

RAIZ_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRAVA = os.path.join(RAIZ_REPO, "docs", "destrancamento.json")
HASHES_LISTAS = os.path.join(RAIZ_REPO, "data", "splits", "listas_sha256.json")
FONTE = {"plus": "chexpert_plus", "rsna": "rsna", "padchest": "padchest"}
CACHE = {"plus": "/out/plus/passada/c320_8bits", "rsna": "/out/rsna/passada/c320_8bits",
         "padchest": "/out/padchest/passada/c320_8bits", "padchest_janela": "/out/padchest/passada/c320_janela"}
HASHES_CACHE = "/out/{base}/cache_hashes.csv"
ROTULOS = {"positivo": 1, "negativo": 0}
POSICAO_PADCHEST = {"PA": "PA", "AP": "AP", "AP_horizontal": "AP"}


def lista(base: str, simulado: bool) -> tuple[str, str]:
    """(arquivo da lista, partição). No modo simulado, a validação; no PadChest simulado, o próprio teste.
    ``padchest_janela``: o subconjunto Philips da lista (``src.padchest.janela``)."""
    if base == "padchest_janela":
        return "/out/padchest/lista_janela.csv", "teste"
    if simulado and base != "padchest":
        return f"/out/{base}/listas/validacao.csv", "validacao"
    return f"/out/{base}/listas_teste_trancado/teste.csv", "teste"


# ------------------------------------------------------------------ trava
def _git(*args: str, repo: str = RAIZ_REPO) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)


def conferir_trava(base: str, ids, trava: str = TRAVA, hashes_listas: str = HASHES_LISTAS,
                   repo: str = RAIZ_REPO) -> dict:
    """Levanta erro se o teste não foi destrancado pelo procedimento do passo 6.7."""
    from src.divisao.listas import sha256_ids
    if not os.path.exists(trava):
        raise PermissionError("teste trancado: docs/destrancamento.json não existe (passo 6.7). "
                              "Antes disso, só --simulado.")
    d = json.load(open(trava, encoding="utf-8"))
    tag = d["tag_codigo"]
    if _git("rev-parse", "-q", "--verify", f"refs/tags/{tag}", repo=repo).returncode != 0:
        raise PermissionError(f"a tag {tag} registrada no destrancamento não existe neste clone (git fetch --tags)")
    if _git("merge-base", "--is-ancestor", tag, "HEAD", repo=repo).returncode != 0:
        raise PermissionError(f"o commit atual não descende da tag {tag}")
    congelado = json.load(open(hashes_listas, encoding="utf-8"))["bases"][base]["listas"]["teste"]["sha256"]
    agora = sha256_ids(ids)
    if not (agora == congelado == d["listas_sha256"][base]):
        raise PermissionError(f"{base}: o SHA-256 da lista de teste não confere com o congelado")
    return d


# ------------------------------------------------------------------ imagens
def imagens(base: str, arq_lista: str, cache: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(regra primária, incertos) com id, paciente, rotulo, caminho, fonte. Incertos com rótulo 1."""
    from src.cache.hashes import nome_cache
    cache = cache or CACHE[base]
    df = pd.read_csv(arq_lista, dtype={"id": str})
    col_pac = [c for c in df.columns if c not in ("id", "rotulo")][0]
    df = df.rename(columns={col_pac: "paciente"})
    df["paciente"] = df["paciente"].astype(str)
    df["caminho"] = [os.path.join(cache, nome_cache(i)) for i in df["id"]]
    df["fonte"] = base
    principal = df[df["rotulo"].isin(ROTULOS)].assign(rotulo=lambda d: d["rotulo"].map(ROTULOS).astype(int))
    incertos = df[df["rotulo"] == "incerto"].assign(rotulo=1)
    cols = ["id", "paciente", "rotulo", "caminho", "fonte"]
    return principal[cols].reset_index(drop=True), incertos[cols].reset_index(drop=True)


# ------------------------------------------------------------------ covariáveis
def covariaveis_padchest(recorte: pd.DataFrame, medidas: pd.DataFrame, ids) -> pd.DataFrame:
    from src.cache.covariaveis import MEDIDAS
    from src.qualidade.fabricante import normalizar_fabricante
    r = recorte.drop_duplicates("ImageID").set_index("ImageID")
    m = medidas.drop_duplicates("ImageID", keep="last").set_index("ImageID")
    ids = pd.Index(sorted(set(map(str, ids))))
    sem_r, sem_m = ids.difference(r.index), ids.difference(m.index)
    if len(sem_r) or len(sem_m):
        raise RuntimeError(f"padchest: {len(sem_r)} imagens sem linha no recorte e {len(sem_m)} sem medidas")
    r, m = r.loc[ids], m.loc[ids]
    fab = [None if pd.isna(f) or str(f).strip() in ("", "nan") else normalizar_fabricante(f)
           for f in r["Manufacturer_DICOM"]]
    df = pd.DataFrame({"id": ids, "id_estudo": r["StudyID"].astype(str).to_numpy(),
                       "posicao": r["Projection"].map(POSICAO_PADCHEST).to_numpy(),
                       "idade": pd.to_numeric(r["idade"]).round().astype("Int64").to_numpy(),
                       "sexo": r["PatientSex_DICOM"].to_numpy(), "fabricante": fab})
    for c in MEDIDAS:
        df[c] = pd.to_numeric(m[c], errors="coerce").to_numpy()
    return df


def covariaveis(base: str, ids) -> pd.DataFrame:
    """Covariáveis das imagens ``ids``, a partir dos arquivos do notebook da base."""
    if base == "padchest":
        med = pd.concat([pd.read_csv(a) for a in sorted(glob.glob("/out/padchest/medidas/medidas_pacote_*.csv"))],
                        ignore_index=True)
        rec = pd.read_csv("/out/padchest_recorte.csv", dtype={"ImageID": str, "StudyID": str})
        med["ImageID"] = med["ImageID"].astype(str)
        return covariaveis_padchest(rec, med, ids)
    from src.cache.covariaveis import BASES, montar
    from src.passada.passada_completa import juntar
    return montar(base, pd.read_csv(BASES[base]["elegiveis"], low_memory=False), juntar(base), pd.Series(list(ids)))


# ------------------------------------------------------------------ execução
def rodar(base: str, simulado: bool, modelos: str = "/out/modelos", saida_raiz: str | None = None,
          predizer_fn=None, cov_fn=None, workers: int = 4, conversao: str = "percentis") -> dict:
    from src.predicoes.pos_treino import SEMENTES, TAXA
    from src.predicoes.schema import validar
    from src.predicoes.tabela import para_schema
    from src.treino.carregador import conferir_hashes

    chave = "padchest_janela" if conversao == "janela" else base
    if conversao == "janela" and base != "padchest":
        raise ValueError("a conversão alternativa (janela) só existe no PadChest")
    arq, particao = lista(chave, simulado)
    principal, incertos = imagens(base, arq, CACHE[chave])
    todas = pd.concat([principal, incertos], ignore_index=True)
    if simulado:
        trava = None
    else:                                    # a trava confere a lista inteira da base, não o subconjunto
        arq_base, _ = lista(base, False)
        ids_base = pd.read_csv(arq_base, usecols=["id"], dtype={"id": str})["id"]
        trava = conferir_trava(base, ids_base)
    conferir_hashes(todas, HASHES_CACHE.format(base=chave), marcas="")     # sempre a conferência completa
    sem_modelo = simulado and base == "padchest"
    saida = os.path.join(saida_raiz or ("/out/teste_simulado" if simulado else "/out/teste"), chave)
    pasta_cov = os.path.join(saida, "covariaveis")
    os.makedirs(os.path.join(pasta_cov, base), exist_ok=True)
    cov = (cov_fn or covariaveis)(base, todas["id"])
    cov.to_csv(os.path.join(pasta_cov, base, "covariaveis.csv"), index=False)   # para_schema lê daqui
    if predizer_fn is None:
        from src.predicoes.predizer import predizer
        predizer_fn = lambda pasta, img: predizer(pasta, img, workers=workers)  # noqa: E731
    tabs = {"principal": [], "incertos": []}
    problemas = []
    for m in TAXA:
        for s in SEMENTES:
            ex = f"{m}_s{s}_lr{TAXA[m]:g}"
            pasta = os.path.join(modelos, ex)
            if sem_modelo:
                res = {"modelo": m, "semente": s, "base": base}
                pred = pd.DataFrame({"id": todas["id"], "paciente": todas["paciente"], "fonte": base,
                                     "rotulo": todas["rotulo"], "logit_corrigido": 0.0, "prob": 0.5})
            else:
                res = json.load(open(os.path.join(pasta, "resumo.json"), encoding="utf-8"))
                pred = predizer_fn(pasta, todas)
            t = para_schema(pred, res, pasta, [base], dados=pasta_cov, particao=particao)
            erros = validar(t)
            if erros:
                problemas.append((ex, erros[:3]))
            inc = t["id_imagem"].isin(set(incertos["id"]))
            tabs["principal"].append(t[~inc])
            tabs["incertos"].append(t[inc])
            print(f"{ex}: {int((~inc).sum())} imagens (+{int(inc.sum())} incertas) | "
                  f"prob. média {t['probabilidade'].mean():.4f}" + (f" | ERROS: {erros[:2]}" if erros else ""), flush=True)
            if sem_modelo:
                break                                    # sem modelo, uma execução basta para conferir
        if sem_modelo:
            break
    arquivos = []
    for nome, lst in tabs.items():
        tab = pd.concat(lst, ignore_index=True)
        if nome == "incertos" and tab.empty:
            continue
        arq_saida = os.path.join(saida, "tabela_teste.csv.gz" if nome == "principal" else "tabela_teste_incertos.csv.gz")
        tab.to_csv(arq_saida, index=False, compression="gzip")
        arquivos.append(arq_saida)
    commit = _git("rev-parse", "--short", "HEAD").stdout.strip()
    resumo = {"base": base, "conversao": conversao, "particao": particao, "simulado": simulado, "sem_modelo": sem_modelo,
              "imagens": int(len(principal)), "positivos": int(principal["rotulo"].sum()),
              "incertos": int(len(incertos)), "pacientes": int(todas["paciente"].nunique()),
              "execucoes": int(len(tabs["principal"])), "commit": commit,
              "tag_codigo": trava["tag_codigo"] if trava else None,
              "gerado_em": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "problemas_schema": problemas, "arquivos": [os.path.basename(a) for a in arquivos]}
    with open(os.path.join(saida, "resumo_teste.json"), "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)
    resumo["_arquivos"] = arquivos + [os.path.join(saida, "resumo_teste.json")]
    return resumo


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", choices=sorted(FONTE), required=True)
    ap.add_argument("--simulado", action="store_true", help="antes da tag: validação (Plus, RSNA) ou sem modelo (PadChest)")
    ap.add_argument("--enviar", action="store_true", help="envia as tabelas ao dataset resultados (só fora do simulado)")
    ap.add_argument("--modelos", default="/out/modelos")
    ap.add_argument("--conversao", choices=["percentis", "janela"], default="percentis",
                    help="janela: sensibilidade da conversão no PadChest (Philips; src.padchest.janela)")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    a = ap.parse_args(argv)
    if a.simulado and a.enviar:
        raise SystemExit("o modo simulado não envia nada")
    r = rodar(a.base, a.simulado, a.modelos, workers=a.workers, conversao=a.conversao)
    print(f"\n{a.base} | partição {r['particao']}{' (SIMULADO' + (', sem modelo' if r['sem_modelo'] else '') + ')' if a.simulado else ''} | "
          f"{r['imagens']} imagens, {r['positivos']} positivos, {r['incertos']} incertas, {r['pacientes']} pacientes | "
          f"{r['execucoes']} execuções | commit {r['commit']}")
    print(f"Arquivos: {r['arquivos']}")
    if a.enviar and not r["problemas_schema"]:
        from src.cache.resultados import enviar
        enviar(r["_arquivos"], origem=f"{a.base}__teste" + ("_janela" if a.conversao == "janela" else ""))
    print("RESULTADO:", "OK" if not r["problemas_schema"] else f"PROBLEMAS NO SCHEMA: {r['problemas_schema'][:2]}")
    return 0 if not r["problemas_schema"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
