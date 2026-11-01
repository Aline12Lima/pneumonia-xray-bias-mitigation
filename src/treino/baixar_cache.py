"""Baixa o cache de treino e validação do dataset ``cache_treino`` para o notebook de treino (F5).

Roda no notebook do workflow ``treino_modelos`` (que só tem o ``cache_treino`` como fonte, sem
nenhuma imagem de teste). Os arquivos vão para ``/out/dados/<base>/``, na mesma estrutura do
``/out`` dos notebooks de auditoria, para o carregador funcionar igual (``raiz="/out/dados"``).
O ``/out`` persiste entre sessões e não conta na cota de armazenamento: o download é feito uma vez.

No fim, confere o número de imagens e o SHA-256 de cada uma contra o passo 5.1.

Uso (CPU gratuita basta):
    python -m src.treino.baixar_cache --base plus
    python -m src.treino.baixar_cache --base rsna
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil

import pandas as pd

DESTINO = "/out/dados"
REF_DATASET = "cache_treino:976k"
USUARIO = "ads_alinelima"
LISTAS = {"plus": ["treino_subamostra.csv", "validacao.csv"], "rsna": ["treino.csv", "validacao.csv"]}


def _tabela(nome: str):
    """A tabela do dataset, pela fonte do workflow; se falhar, pela referência do dataset."""
    import redivis
    try:
        t = redivis.table(nome)
        t.get()
        return t
    except Exception as e:
        print(f"  (tabela {nome} pela fonte do workflow falhou: {type(e).__name__}; usando a referência do dataset)")
        return redivis.user(USUARIO).dataset(REF_DATASET, version="next").table(nome)


def _baixar(tab, pasta: str, esperados: int) -> int:
    os.makedirs(pasta, exist_ok=True)
    tem = len(os.listdir(pasta))
    if tem >= esperados:
        print(f"  já baixado: {tem} arquivos em {pasta}")
        return tem
    try:
        tab.download_files(path=pasta, overwrite=False)
    except TypeError:                                   # versões antigas sem o argumento overwrite
        tab.download_files(pasta)
    for dirpath, _, arquivos in os.walk(pasta):         # se vierem em subpastas, traz para a raiz
        if dirpath != pasta:
            for a in arquivos:
                shutil.move(os.path.join(dirpath, a), os.path.join(pasta, a))
    for d in [d for d in os.listdir(pasta) if os.path.isdir(os.path.join(pasta, d))]:
        shutil.rmtree(os.path.join(pasta, d), ignore_errors=True)
    return len(os.listdir(pasta))


def baixar(base: str) -> None:
    raiz = f"{DESTINO}/{base}"
    tmp = f"/scratch/listas_{base}"
    print(f"=== {base}: listas")
    shutil.rmtree(tmp, ignore_errors=True)
    _baixar(_tabela(f"{base}_listas"), tmp, 3)
    os.makedirs(f"{raiz}/listas", exist_ok=True)
    for nome in LISTAS[base]:
        shutil.copyfile(f"{tmp}/{nome}", f"{raiz}/listas/{nome}")
    shutil.copyfile(f"{tmp}/cache_hashes.csv", f"{raiz}/cache_hashes.csv")
    hashes = pd.read_csv(f"{raiz}/cache_hashes.csv")
    print(f"  listas: {', '.join(LISTAS[base])} | hashes: {len(hashes)} imagens")

    print(f"=== {base}: imagens ({len(hashes)})")
    n = _baixar(_tabela(f"{base}_c320"), f"{raiz}/passada/c320_8bits", len(hashes))
    print(f"  {n} arquivos em {raiz}/passada/c320_8bits")

    from src.treino.carregador import ler_particao, resumo
    print("=== conferência pelo carregador (contagens e SHA-256 de cada imagem)")
    for p in ("treino", "validacao"):
        print(f"  {p}: {resumo(ler_particao(base, p, raiz=DESTINO))}")
    tam = sum(os.path.getsize(f) for f in glob.glob(f"{raiz}/passada/c320_8bits/*.png")) / 1e9
    print(f"RESULTADO: {base} pronto para o treino ({tam:.2f} GB em {raiz})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", choices=sorted(LISTAS), required=True)
    baixar(ap.parse_args(argv).base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
