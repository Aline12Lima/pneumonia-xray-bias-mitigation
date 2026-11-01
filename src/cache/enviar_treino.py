"""Envia o cache de treino e validação de uma base para o dataset privado ``cache_treino`` (F5).

Por quê: no Redivis, os arquivos do ``/out`` ficam presos ao notebook. Para o notebook de treino
(outro workflow, com GPU) ler as imagens do Plus e do RSNA, elas passam por um dataset privado.

O que vai (só treino e validação; **o teste nunca sai do notebook da base**):

- tabela ``<base>_c320``: os PNG de 320 px em 8 bits das imagens de treino e validação;
- tabela ``<base>_listas``: ``treino*.csv``, ``validacao.csv`` e ``cache_hashes.csv`` (só as linhas
  de treino e validação). Os nomes são os mesmos do ``/out``, para o carregador funcionar igual.

O envio é retomável: rodar de novo pula o que já está no dataset. A versão do dataset fica em
edição ("next"), privada; não publicar.

Uso (no notebook da base, CPU gratuita):
    python -m src.cache.enviar_treino --base plus
    python -m src.cache.enviar_treino --base plus --conferir     # só conta o que já está lá
"""
from __future__ import annotations

import argparse
import os
import shutil
import time

import pandas as pd

USUARIO = "ads_alinelima"
DATASET = "cache_treino"
LOTE = 500

BASES = {
    "plus": {"listas": {"treino_subamostra.csv": "/out/plus/listas/treino_subamostra.csv",
                        "validacao.csv": "/out/plus/listas/validacao.csv"},
             "hashes": "/out/plus/cache_hashes.csv", "cache": "/out/plus/passada/c320_8bits",
             "particoes": ["treino_subamostra", "validacao"]},
    "rsna": {"listas": {"treino.csv": "/out/rsna/listas/treino.csv",
                        "validacao.csv": "/out/rsna/listas/validacao.csv"},
             "hashes": "/out/rsna/cache_hashes.csv", "cache": "/out/rsna/passada/c320_8bits",
             "particoes": ["treino", "validacao"]},
}


def selecionar(base: str, raiz: str = "/out", pasta_listas: str | None = None) -> tuple[list[str], list[str]]:
    """Devolve (PNGs a enviar, arquivos de lista a enviar). Nunca inclui nada do teste."""
    cfg = BASES[base]
    troca = (lambda p: p.replace("/out", raiz, 1)) if raiz != "/out" else (lambda p: p)
    hashes = pd.read_csv(troca(cfg["hashes"]))
    hashes = hashes[hashes["particao"].isin(cfg["particoes"])]
    if (hashes["sha256"].isna() | (hashes["sha256"].astype(str) == "")).any():
        raise RuntimeError("há imagens de treino ou validação sem arquivo no cache; rode o passo 5.1 de novo")
    pngs = [os.path.join(troca(cfg["cache"]), a) for a in hashes["arquivo"]]

    pasta_listas = pasta_listas or f"/scratch/envio_{base}"
    os.makedirs(pasta_listas, exist_ok=True)
    listas = []
    for nome, origem in cfg["listas"].items():
        shutil.copyfile(troca(origem), os.path.join(pasta_listas, nome))
        listas.append(os.path.join(pasta_listas, nome))
    hashes.to_csv(os.path.join(pasta_listas, "cache_hashes.csv"), index=False)
    listas.append(os.path.join(pasta_listas, "cache_hashes.csv"))
    return pngs, listas


def _dataset():
    import redivis
    ds = redivis.user(USUARIO).dataset(DATASET)
    if not ds.exists():
        ds.create(public_access_level="none",
                  description="Cache de 320 px (8 bits) de treino e validação do Plus e do RSNA, para o notebook "
                              "de treino. Privado. Sem nenhuma imagem de teste.")
        print(f"Dataset {DATASET} criado (privado).")
    return redivis.user(USUARIO).dataset(DATASET, version="next")


def _tabela(ds, nome: str, descricao: str):
    tab = ds.table(nome)
    if not tab.exists():
        tab.create(description=descricao, is_file_index=True)
    return tab


def _nomes(tab) -> set[str]:
    try:
        return {getattr(f, "name", None) or f.properties.get("name") for f in tab.list_files()}
    except Exception as e:                        # tabela recém-criada e vazia devolve erro 500
        print(f"  (consulta dos já enviados falhou: {type(e).__name__}; considerando vazia)")
        return set()


def enviar(tab, arquivos: list[str]) -> int:
    for rodada in range(1, 30):
        ja = _nomes(tab)
        faltam = [p for p in arquivos if os.path.basename(p) not in ja]
        print(f"  rodada {rodada}: já no dataset {len(ja)} | faltam {len(faltam)}", flush=True)
        if not faltam:
            return len(ja)
        for i in range(0, len(faltam), LOTE):
            lote = faltam[i:i + LOTE]
            try:
                tab.add_files(files=lote, max_parallelization=8)
                print(f"    lote {i // LOTE + 1}: ok ({i + len(lote)}/{len(faltam)})", flush=True)
            except Exception as e:
                print(f"    lote {i // LOTE + 1}: falhou ({type(e).__name__}: {str(e)[:120]}); recomeça na próxima rodada")
                time.sleep(30)
                break
    return len(_nomes(tab))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", choices=sorted(BASES), required=True)
    ap.add_argument("--conferir", action="store_true", help="só conta o que já está no dataset")
    a = ap.parse_args(argv)

    pngs, listas = selecionar(a.base)
    gb = sum(os.path.getsize(p) for p in pngs) / 1e9
    print(f"{a.base}: {len(pngs)} imagens de treino e validação ({gb:.2f} GB) e {len(listas)} arquivos de lista. "
          "Nenhum arquivo de teste.")
    ds = _dataset()
    t_img = _tabela(ds, f"{a.base}_c320", f"{a.base}: cache de 320 px em 8 bits, só treino e validação")
    t_lst = _tabela(ds, f"{a.base}_listas", f"{a.base}: listas de treino e validação e hashes do cache")
    if a.conferir:
        print(f"No dataset: {len(_nomes(t_img))} de {len(pngs)} imagens | {len(_nomes(t_lst))} de {len(listas)} listas")
        return 0
    print("\nListas:")
    n_lst = enviar(t_lst, listas)
    print("\nImagens:")
    n_img = enviar(t_img, pngs)
    ok = n_img == len(pngs) and n_lst == len(listas)
    print(f"\nTotal no dataset: {n_img} de {len(pngs)} imagens | {n_lst} de {len(listas)} listas")
    print("RESULTADO:", "envio completo" if ok else "INCOMPLETO — rode a mesma célula de novo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
