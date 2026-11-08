"""Cache de 320 px do PadChest para o teste externo (F6, preparação do passo 6.8; protocolo §6, §7.4).

O PadChest só entra como teste externo (D-006) e, até aqui, só tinha as medidas de qualidade (segunda
passada): nenhuma cópia em 320 px. Este script baixa de novo os 13 pacotes do recorte, um por vez, e
grava, para cada imagem da lista do PadChest, a mesma imagem que os modelos viram nas outras bases:

- **conversão oficial** ``para_8bits`` (percentis 0,5–99,5, sem inversão: tudo MONOCHROME2), a mesma
  usada nas medidas da segunda passada (F3 › PadChest › Conversão);
- ``redimensionar(img8, 320)`` (INTER_AREA, sem manter proporção), igual ao Plus e ao RSNA.

Saída: ``/out/padchest/passada/c320_8bits/<nome>.png`` (mesmo nome de ``src.cache.hashes.nome_cache``).
Depois, ``python -m src.cache.hashes --base padchest`` calcula o SHA-256 de cada arquivo e o do conjunto.

O que este script **não** faz: não lê rótulos nem predições. Da lista trancada lê só a coluna ``id``
(como ``src.cache.hashes``), para saber quais arquivos gravar. Nenhum modelo é aplicado aqui.

Termos de uso do PadChest (cláusulas 5 a 7): a cópia de 320 px fica só no Redivis, privada; o link de
download vem da variável de ambiente ``PADCHEST_LINK``, definida só no notebook privado.

Retomada: arquivos já gravados são pulados; um pacote interrompido recomeça das imagens que faltam.
Cada PNG é gravado num arquivo temporário e renomeado, para não deixar arquivo pela metade.
Enquanto um pacote é convertido, o próximo já está sendo baixado. Cada zip é apagado depois.

Caminho principal (desde 8/11): ``python -m src.padchest.segunda_passada --com-cache``, que grava as
medidas e esta cópia na mesma leitura de cada zip. Este script fica como complemento, para gravar só o
cache de imagens que faltarem.

Uso (no notebook do workflow ``padchest_teste``, CPU):
    python -m src.padchest.cache320
"""
from __future__ import annotations

import argparse
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd

from src.padchest.segunda_passada import PACOTES, RECORTE, baixar_pacote

LISTA = "/out/padchest/listas_teste_trancado/teste.csv"
CACHE = "/out/padchest/passada/c320_8bits"
FALHAS = "/out/padchest/passada/c320_falhas.csv"


def converter(g: np.ndarray) -> np.ndarray:
    """PNG de 16 bits do PadChest → 8 bits (conversão oficial) → 320 × 320."""
    from src.imagem.conversao import para_8bits, redimensionar
    g = g if g.ndim == 2 else g[..., 0]
    return redimensionar(para_8bits(g), 320)


def alvo(lista: str = LISTA, recorte: str = RECORTE) -> pd.DataFrame:
    """IDs da lista (só a coluna ``id``) com o pacote de cada um, vindo do recorte."""
    ids = pd.read_csv(lista, usecols=["id"])["id"].astype(str)
    rec = pd.read_csv(recorte, usecols=["ImageID", "ImageDir"]).drop_duplicates("ImageID")
    df = pd.DataFrame({"id": ids}).merge(rec.rename(columns={"ImageID": "id"}), on="id", how="left")
    if df["ImageDir"].isna().any():
        raise RuntimeError(f"{int(df['ImageDir'].isna().sum())} IDs da lista sem pacote no recorte")
    return df.astype({"ImageDir": int})


def faltando(df: pd.DataFrame, cache: str = CACHE) -> pd.DataFrame:
    from src.cache.hashes import nome_cache
    return df[[not os.path.exists(os.path.join(cache, nome_cache(i))) for i in df["id"]]]


def processar_pacote(n: int, zipf: str, ids: list[str], cache: str = CACHE) -> list[dict]:
    """Converte as imagens ``ids`` do zip do pacote n. Devolve as falhas (id, pacote, erro)."""
    from src.cache.hashes import nome_cache
    os.makedirs(cache, exist_ok=True)
    falhas, t0 = [], time.time()
    with zipfile.ZipFile(zipf) as z:
        nomes = {os.path.basename(m): m for m in z.namelist() if m.lower().endswith(".png")}
        for i, ident in enumerate(ids, 1):
            try:
                if ident not in nomes:
                    raise KeyError("não está no zip")
                g = cv2.imdecode(np.frombuffer(z.read(nomes[ident]), np.uint8), cv2.IMREAD_UNCHANGED)
                if g is None:
                    raise ValueError("não decodificou")
                destino = os.path.join(cache, nome_cache(ident))
                tmp = destino + ".tmp.png"
                if not cv2.imwrite(tmp, converter(g)):
                    raise IOError("não gravou")
                os.replace(tmp, destino)
            except Exception as e:
                falhas.append({"id": ident, "pacote": n, "erro": f"{type(e).__name__}: {str(e)[:80]}"})
            if i % 1000 == 0:
                ritmo = (time.time() - t0) / i
                print(f"  pacote {n}: {i}/{len(ids)} | {ritmo:.2f} s por imagem", flush=True)
    return falhas


def rodar(link: str, pacotes=PACOTES) -> tuple[list[int], list[dict]]:
    df = faltando(alvo())
    por_pacote = {n: g["id"].tolist() for n, g in df.groupby("ImageDir") if n in pacotes}
    faltam = sorted(por_pacote)
    print(f"Imagens da lista sem cache: {len(df)} | pacotes a baixar: {faltam}", flush=True)
    erros_download, falhas = [], []
    with ThreadPoolExecutor(1) as ex:
        futuro = ex.submit(baixar_pacote, link, faltam[0]) if faltam else None
        for k, n in enumerate(faltam):
            try:
                zipf = futuro.result()
            except Exception as e:
                print(f"Pacote {n}: download FALHOU ({type(e).__name__}: {str(e)[:150]})", flush=True)
                erros_download.append(n)
                zipf = None
            futuro = ex.submit(baixar_pacote, link, faltam[k + 1]) if k + 1 < len(faltam) else None
            if zipf:
                t0 = time.time()
                f = processar_pacote(n, zipf, por_pacote[n])
                falhas += f
                os.remove(zipf)
                print(f"Pacote {n}: {len(por_pacote[n]) - len(f)} gravadas, {len(f)} falhas, "
                      f"{(time.time() - t0) / 60:.1f} min | zip apagado", flush=True)
    return erros_download, falhas


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pacotes", type=int, nargs="*", default=PACOTES)
    a = ap.parse_args(argv)
    link = os.environ.get("PADCHEST_LINK")
    if not link:
        raise SystemExit("Defina PADCHEST_LINK no notebook (o link não fica no repositório).")
    inicio = time.time()
    erros, falhas = rodar(link, a.pacotes)
    if erros:
        print(f"\nDownload falhou nos pacotes {erros}; pausa de 2 min e nova tentativa", flush=True)
        time.sleep(120)
        erros, mais = rodar(link, erros)
        falhas += mais
    if falhas:
        pd.DataFrame(falhas).to_csv(FALHAS, index=False)
    resta = len(faltando(alvo()))
    print(f"\nFim da sessão: {(time.time() - inicio) / 3600:.1f} h | imagens sem cache: {resta}"
          + (f" | falhas de conversão: {len(falhas)} (em {FALHAS})" if falhas else ""))
    print("RESULTADO:", "cache completo" if resta == 0 else "INCOMPLETO — rode a mesma célula de novo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
