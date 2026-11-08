"""Dataset privado ``modelos``: os 15 modelos finais vão do notebook de treino aos notebooks das bases.

Por quê (F6, passos 6.5 e 6.8): as degradações espaciais do eixo B precisam da imagem na resolução
original, que só existe no notebook de cada base, e o teste nunca sai de lá. Então os modelos vão até
as imagens, e não o contrário.

O que vai (por execução final: M1 1e-4, M2 3e-4, M3 1e-4; sementes 0 a 4):
``<execução>__melhor.pt``, ``<execução>__resumo.json`` e ``<execução>__pos_treino.json`` (limiar e
temperatura), numa tabela de arquivos ``modelos``. Cerca de 0,4 GB.

Uso:
    python -m src.cache.modelos --enviar      # notebook "treino": envia (retoma se cair)
    python -m src.cache.modelos --baixar      # notebook da base: baixa para /out/modelos/<execução>/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil

from src.predicoes.pos_treino import SEMENTES, TAXA, pasta_final

USUARIO = "ads_alinelima"
DATASET = "modelos"
ARQUIVOS = ["melhor.pt", "resumo.json", "pos_treino.json"]
DESTINO = "/out/modelos"


def execucoes() -> list[str]:
    return [os.path.basename(pasta_final(m, s)) for m in TAXA for s in SEMENTES]


def _sha(caminho: str) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def _tabela(criar: bool):
    """Para enviar: a versão em edição do dataset. Para baixar: a fonte do workflow (o dataset precisa
    ter sido adicionado ao workflow da base) e, se falhar, a referência do dataset."""
    import redivis
    if not criar:
        try:
            t = redivis.table("modelos")
            t.get()
            return t
        except Exception as e:
            print(f"  (tabela modelos pela fonte do workflow falhou: {type(e).__name__}; usando a referência do dataset)")
        return redivis.user(USUARIO).dataset(DATASET, version="next").table("modelos")
    ds = redivis.user(USUARIO).dataset(DATASET)
    if not ds.exists():
        ds.create(public_access_level="none",
                  description="Os 15 modelos finais (pesos, limiar e temperatura) para os notebooks das bases. Privado.")
        print(f"Dataset {DATASET} criado (privado).")
    tab = redivis.user(USUARIO).dataset(DATASET, version="next").table("modelos")
    if not tab.exists():
        tab.create(description="Pesos (melhor.pt), resumo.json e pos_treino.json das 15 execuções finais",
                   is_file_index=True)
    return tab


def enviar(treinos: str = "/out/treinos") -> int:
    from src.cache.enviar_treino import enviar as enviar_lotes, linhas
    pasta = "/scratch/envio_modelos"
    os.makedirs(pasta, exist_ok=True)
    arquivos, manifesto = [], {}
    for ex in execucoes():
        for a in ARQUIVOS:
            origem = os.path.join(treinos, ex, a)
            if not os.path.exists(origem):
                raise FileNotFoundError(origem)
            destino = os.path.join(pasta, f"{ex}__{a}")
            shutil.copyfile(origem, destino)
            arquivos.append(destino)
            manifesto[f"{ex}__{a}"] = _sha(destino)
    man = os.path.join(pasta, "manifesto_sha256.json")
    json.dump(manifesto, open(man, "w"), indent=1)
    arquivos.append(man)
    gb = sum(os.path.getsize(p) for p in arquivos) / 1e9
    print(f"{len(execucoes())} execuções, {len(arquivos)} arquivos ({gb:.2f} GB)")
    tab = _tabela(criar=True)
    enviar_lotes(tab, arquivos, "/out/treinos/envio_modelos.txt")
    n = linhas(tab)
    print(f"Na tabela: {n} de {len(arquivos)} arquivos")
    print("RESULTADO:", "envio completo" if n == len(arquivos) else "INCOMPLETO — rode a mesma célula de novo")
    return 0 if n == len(arquivos) else 1


def baixar(destino: str = DESTINO) -> int:
    tmp = "/scratch/modelos_baixados"
    shutil.rmtree(tmp, ignore_errors=True)
    tab = _tabela(criar=False)
    os.makedirs(tmp, exist_ok=True)
    try:
        tab.to_directory().download(tmp)
    except Exception:                                   # versões antigas do cliente
        tab.download_files(tmp)
    arqs = {os.path.basename(p): os.path.join(r, p) for r, _, fs in os.walk(tmp) for p in fs}
    manifesto = json.load(open(arqs["manifesto_sha256.json"]))
    ruins = [n for n, h in manifesto.items() if n not in arqs or _sha(arqs[n]) != h]
    if ruins:
        print(f"RESULTADO: {len(ruins)} arquivos faltando ou diferentes (ex.: {ruins[0]})")
        return 1
    for nome in manifesto:
        ex, a = nome.split("__", 1)
        os.makedirs(os.path.join(destino, ex), exist_ok=True)
        shutil.copyfile(arqs[nome], os.path.join(destino, ex, a))
    print(f"{len(execucoes())} execuções em {destino}/ (SHA-256 conferido em {len(manifesto)} arquivos)")
    print("RESULTADO: modelos prontos")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--enviar", action="store_true")
    g.add_argument("--baixar", action="store_true")
    a = ap.parse_args(argv)
    return enviar() if a.enviar else baixar()


if __name__ == "__main__":
    raise SystemExit(main())
