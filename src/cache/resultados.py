"""Dataset privado ``resultados``: junta tabelas pequenas de notebooks de workflows diferentes.

Por quê: no Redivis, um notebook não lê o ``/out`` de outro workflow. O eixo B (e, no teste, as
predições) fica no notebook de cada base; o eixo A e a consolidação das famílias (Holm), no notebook de
treino. Cada notebook manda o que produziu; a consolidação baixa tudo.

Os arquivos ganham um prefixo com a origem (ex.: ``rsna__validacao__eixo_b_resultados.csv``).
Reenviar com o mesmo nome substitui.

Uso:
    python -m src.cache.resultados --enviar /out/rsna/eixo_b/validacao/eixo_b_resultados.csv --origem rsna__validacao
    python -m src.cache.resultados --baixar            # para /out/resultados/
"""
from __future__ import annotations

import argparse
import os
import shutil

USUARIO = "ads_alinelima"
DATASET = "resultados"
DESTINO = "/out/resultados"


def _tabela(criar: bool):
    import redivis
    if not criar:
        try:
            t = redivis.table("resultados")
            t.get()
            return t
        except Exception as e:
            print(f"  (tabela resultados pela fonte do workflow falhou: {type(e).__name__}; usando a referência do dataset)")
        return redivis.user(USUARIO).dataset(DATASET, version="next").table("resultados")
    ds = redivis.user(USUARIO).dataset(DATASET)
    if not ds.exists():
        ds.create(public_access_level="none", description="Tabelas de resultados (pequenas) para juntar notebooks. Privado.")
        print(f"Dataset {DATASET} criado (privado).")
    tab = redivis.user(USUARIO).dataset(DATASET, version="next").table("resultados")
    if not tab.exists():
        tab.create(description="Resultados das análises, com a origem no nome do arquivo", is_file_index=True)
    return tab


def nome_destino(arquivo: str, origem: str) -> str:
    return f"{origem}__{os.path.basename(arquivo)}"


def enviar(arquivos: list[str], origem: str) -> int:
    pasta = "/scratch/envio_resultados"
    os.makedirs(pasta, exist_ok=True)
    copias = []
    for a in arquivos:
        d = os.path.join(pasta, nome_destino(a, origem))
        shutil.copyfile(a, d)
        copias.append(d)
    _tabela(criar=True).add_files(files=copias)
    print(f"{len(copias)} arquivos enviados: {[os.path.basename(c) for c in copias]}")
    print("RESULTADO: enviado")
    return 0


def baixar(destino: str = DESTINO) -> int:
    tmp = "/scratch/resultados_baixados"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    tab = _tabela(criar=False)
    try:
        tab.to_directory().download(tmp)
    except Exception:
        tab.download_files(tmp)
    os.makedirs(destino, exist_ok=True)
    n = 0
    for r, _, fs in os.walk(tmp):
        for f in fs:
            shutil.copyfile(os.path.join(r, f), os.path.join(destino, f))
            n += 1
    print(f"{n} arquivos em {destino}: {sorted(os.listdir(destino))}")
    print("RESULTADO: baixado")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--enviar", nargs="+")
    ap.add_argument("--origem")
    ap.add_argument("--baixar", action="store_true")
    a = ap.parse_args(argv)
    if a.enviar:
        if not a.origem:
            ap.error("--origem é obrigatório com --enviar")
        return enviar(a.enviar, a.origem)
    if a.baixar:
        return baixar()
    ap.error("use --enviar ou --baixar")


if __name__ == "__main__":
    raise SystemExit(main())
