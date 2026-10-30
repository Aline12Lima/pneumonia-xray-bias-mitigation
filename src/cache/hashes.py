"""Hash de cada arquivo do cache de 320 px (F5, passo 5.1).

Para cada imagem das listas de uma base, confere que o arquivo do cache existe e calcula o SHA-256
dos bytes do PNG. Isso prova, mais adiante, que as imagens usadas no treino e na avaliação são
exatamente as mesmas (o cache não muda depois deste passo).

Saídas:

- ``/out/<base>/cache_hashes.csv``: uma linha por imagem (id, partição, arquivo, bytes, sha256).
  Fica só no Redivis, porque tem os IDs (termos de uso).
- ``/out/<base>/cache_hashes_resumo.json``: por partição, o número de arquivos, o tamanho e um
  **SHA-256 do conjunto** (calculado sobre as linhas "id sha256" ordenadas). Sem IDs: é o que vai
  para o repositório (``data/splits/cache_sha256.json``).

As listas de teste são lidas só para conferir os arquivos; nada aqui olha rótulos ou predições.

Uso (no notebook da base):
    python -m src.cache.hashes --base plus
    python -m src.cache.hashes --base rsna
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import pandas as pd

BASES = {
    "plus": {"cache": "/out/plus/passada/c320_8bits",
             "listas": {"treino_subamostra": "/out/plus/listas/treino_subamostra.csv",
                        "validacao": "/out/plus/listas/validacao.csv",
                        "teste": "/out/plus/listas_teste_trancado/teste.csv"}},
    "rsna": {"cache": "/out/rsna/passada/c320_8bits",
             "listas": {"treino": "/out/rsna/listas/treino.csv",
                        "validacao": "/out/rsna/listas/validacao.csv",
                        "teste": "/out/rsna/listas_teste_trancado/teste.csv"}},
}


def nome_cache(ident: str) -> str:
    """Mesmo nome usado pela passada completa (a chave do Plus tem barras)."""
    return str(ident).replace("/", "__").removesuffix(".dcm") + ".png"


def sha256_arquivo(caminho: str) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def sha256_conjunto(df: pd.DataFrame) -> str:
    """Um hash para a partição inteira: muda se qualquer arquivo ou ID mudar."""
    linhas = sorted(f"{i} {s}" for i, s in zip(df["id"].astype(str), df["sha256"]))
    return hashlib.sha256("\n".join(linhas).encode()).hexdigest()


def calcular(cache: str, listas: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    regs = []
    for particao, caminho in listas.items():
        ids = pd.read_csv(caminho, usecols=["id"])["id"].astype(str)
        for ident in ids:
            arq = os.path.join(cache, nome_cache(ident))
            if os.path.exists(arq):
                regs.append({"id": ident, "particao": particao, "arquivo": os.path.basename(arq),
                             "bytes": os.path.getsize(arq), "sha256": sha256_arquivo(arq)})
            else:
                regs.append({"id": ident, "particao": particao, "arquivo": os.path.basename(arq),
                             "bytes": 0, "sha256": ""})
    df = pd.DataFrame(regs)
    resumo = {}
    for particao, g in df.groupby("particao", sort=False):
        ok = g[g["sha256"] != ""]
        resumo[particao] = {"imagens_na_lista": int(len(g)), "arquivos_no_cache": int(len(ok)),
                            "faltando": int(len(g) - len(ok)), "megabytes": round(ok["bytes"].sum() / 1e6, 1),
                            "sha256_conjunto": sha256_conjunto(ok)}
    return df, resumo


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", choices=sorted(BASES), required=True)
    a = ap.parse_args(argv)
    cfg = BASES[a.base]
    df, resumo = calcular(cfg["cache"], cfg["listas"])
    saida = f"/out/{a.base}"
    df.to_csv(f"{saida}/cache_hashes.csv", index=False)
    with open(f"{saida}/cache_hashes_resumo.json", "w", encoding="utf-8") as f:
        json.dump({"base": a.base, "particoes": resumo}, f, indent=2, ensure_ascii=False)
    print(f"{a.base}: cache de 320 px conferido contra as listas")
    print(pd.DataFrame(resumo).T.to_string())
    falta = df[df["sha256"] == ""]
    if len(falta):
        print(f"\nATENÇÃO: {len(falta)} imagens das listas sem arquivo no cache:")
        print(falta[["id", "particao"]].head(20).to_string(index=False))
    else:
        print("\nTodas as imagens das listas têm arquivo no cache.")
    print(f"\nTabelas: {saida}/cache_hashes.csv e {saida}/cache_hashes_resumo.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
