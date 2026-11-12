"""Cópia de 320 px do PadChest com a conversão alternativa (janela do CSV), só no Philips (protocolo §6;
F3 › PadChest › Conversão; F6 › PadChest › Sensibilidade da conversão).

O protocolo pede uma análise de sensibilidade com uma conversão alternativa no PadChest, porque a
conversão por percentis não é a apresentação nativa. Pela auditoria, a janela do CSV reescalada só serve
como verificação no Philips (no Imaging Dynamics ela corta as pontas). Esta cópia usa
``janela_8bits`` (a mesma das medidas ``q_*_janela`` da segunda passada) e ``redimensionar(·, 320)``.

Não precisa de rótulo nem de modelo. Para não baixar os 13 pacotes mais uma vez, a cópia é gravada na
mesma leitura dos zips da geração do eixo B do PadChest (F8, ``src.perturbations.eixo_b``): para cada
pacote aberto, grava as imagens Philips da lista que ainda não têm cópia. Também pode rodar sozinha.

Saídas:
- ``/out/padchest/passada/c320_janela/<nome>.png``;
- ``/out/padchest/lista_janela.csv``: as linhas da lista trancada do PadChest que entram (Philips, com
  janela no CSV), para ``src.cache.hashes --base padchest_janela`` e ``src.predicoes.teste --conversao janela``.

Uso (notebook do ``padchest_teste``, CPU; precisa de ``PADCHEST_LINK``):
    python -m src.padchest.janela              # sozinha (uma passada pelos 13 pacotes)
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import pandas as pd

CACHE_JANELA = "/out/padchest/passada/c320_janela"
LISTA_JANELA = "/out/padchest/lista_janela.csv"
LISTA_TESTE = "/out/padchest/listas_teste_trancado/teste.csv"
FABRICANTE = "Philips"


def selecionar(lista: pd.DataFrame, recorte: pd.DataFrame) -> pd.DataFrame:
    """Linhas da lista com fabricante Philips e janela completa no CSV (centro, largura e bits)."""
    from src.padchest.segunda_passada import numero
    from src.qualidade.fabricante import normalizar_fabricante
    r = recorte.drop_duplicates("ImageID").set_index("ImageID")
    r = r.loc[r.index.intersection(lista["id"].astype(str))]
    ok = (r["Manufacturer_DICOM"].map(lambda f: normalizar_fabricante(f) if isinstance(f, str) else None) == FABRICANTE)
    for c in ("WindowCenter_DICOM", "WindowWidth_DICOM", "BitsStored_DICOM"):
        ok &= r[c].map(numero).pipe(np.isfinite)
    ok &= r["WindowWidth_DICOM"].map(numero) > 0
    return lista[lista["id"].astype(str).isin(r.index[ok])].reset_index(drop=True)


def converter_janela(g: np.ndarray, linha) -> np.ndarray | None:
    from src.imagem.conversao import redimensionar
    from src.padchest.segunda_passada import janela_8bits, numero
    g = g if g.ndim == 2 else g[..., 0]
    j8 = janela_8bits(g, numero(linha["WindowCenter_DICOM"]), numero(linha["WindowWidth_DICOM"]),
                      numero(linha["BitsStored_DICOM"]))
    return None if j8 is None else redimensionar(j8, 320)


def preparar(lista_teste: str = LISTA_TESTE, recorte: str | None = None) -> pd.DataFrame:
    """Grava ``lista_janela.csv`` e devolve as linhas do recorte das imagens que entram."""
    from src.padchest.segunda_passada import RECORTE
    lista = pd.read_csv(lista_teste, dtype={"id": str})
    rec = pd.read_csv(recorte or RECORTE, dtype={"ImageID": str})
    sel = selecionar(lista, rec)
    sel.to_csv(LISTA_JANELA, index=False)
    return rec[rec["ImageID"].isin(sel["id"])].set_index("ImageID")


def gravar_do_zip(z, nomes: dict, rec: pd.DataFrame, pacote: int, cache: str = CACHE_JANELA) -> int:
    """Grava as cópias que faltam das imagens do ``pacote`` presentes em ``rec``. Devolve quantas gravou."""
    from src.cache.hashes import nome_cache
    os.makedirs(cache, exist_ok=True)
    feitas = 0
    for ident, linha in rec[rec["ImageDir"] == pacote].iterrows():
        destino = os.path.join(cache, nome_cache(ident))
        if os.path.exists(destino) or ident not in nomes:
            continue
        g = cv2.imdecode(np.frombuffer(z.read(nomes[ident]), np.uint8), cv2.IMREAD_UNCHANGED)
        img = None if g is None else converter_janela(g, linha)
        if img is None:
            continue
        tmp = destino + ".tmp.png"
        if cv2.imwrite(tmp, img):
            os.replace(tmp, destino)
            feitas += 1
    return feitas


def main(argv=None) -> int:
    import zipfile
    from src.cache.hashes import nome_cache
    from src.padchest.segunda_passada import PACOTES, baixar_pacote
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args(argv)
    link = os.environ.get("PADCHEST_LINK")
    if not link:
        raise SystemExit("Defina PADCHEST_LINK no notebook (o link não fica no repositório).")
    rec = preparar()
    falta = rec[[not os.path.exists(os.path.join(CACHE_JANELA, nome_cache(i))) for i in rec.index]]
    pacotes = [n for n in PACOTES if (falta["ImageDir"] == n).any()]
    print(f"Philips com janela na lista: {len(rec)} | sem cópia: {len(falta)} | pacotes: {pacotes}", flush=True)
    for n in pacotes:
        zipf = baixar_pacote(link, n)
        with zipfile.ZipFile(zipf) as z:
            nomes = {os.path.basename(m): m for m in z.namelist() if m.lower().endswith(".png")}
            k = gravar_do_zip(z, nomes, falta, n)
        os.remove(zipf)
        print(f"Pacote {n}: {k} cópias | zip apagado", flush=True)
    resta = sum(not os.path.exists(os.path.join(CACHE_JANELA, nome_cache(i))) for i in rec.index)
    print("RESULTADO:", "cópia completa" if resta == 0 else f"INCOMPLETA ({resta} faltando)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
