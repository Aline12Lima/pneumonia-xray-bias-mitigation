"""Passada completa do RSNA e do CheXpert Plus (F3 › Todas › Passada completa).

Uma leitura de cada DICOM elegível faz tudo de uma vez, na resolução original:

- conversão oficial para 8 bits (percentis 0,5–99,5, MONOCHROME1 invertido);
- máscara do pulmão (mesmo procedimento das três bases) e razão entre os pulmões;
- as oito medidas de qualidade (colunas ``q_*``);
- se o DICOM tiver janela (GE no Plus): as mesmas medidas com a janela, colunas ``q_*_janela``
  (sensibilidade da conversão, F3 › Todas › Conversão);
- resumo de resolução (linhas, colunas, espaçamento do pixel, fabricante, bits);
- duplicatas: MD5 dos pixels (exata) e hash de diferença de 64 bits (quase idêntica);
- o cache de treino: a imagem em 8 bits redimensionada para 320 × 320 (PNG), passo 5.1 do plano.

Cada DICOM é apagado assim que é processado. Guarda só a tabela de medidas e o cache de 320 px.

Retomada: os resultados são gravados em blocos de 200 imagens em ``/out``. Se o notebook parar
(limite de 6 h na configuração gratuita), rodar de novo continua das imagens que faltam.

Uso (no notebook do workflow da base, depois do ``git clone``):
    python -m src.passada.passada_completa --base rsna
    python -m src.passada.passada_completa --base plus
    python -m src.passada.passada_completa --base rsna --resumo     # lê só o /out
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SALVAR_A_CADA = 200
ADIANTADOS = 32          # downloads em andamento enquanto as imagens anteriores são processadas

BASES = {
    "rsna": {"tabela": "/out/rsna/rsna_elegiveis.csv", "id": "SOPInstanceUID",
             "saida": "/out/rsna/passada", "tmp": "/scratch/passada_rsna"},
    "plus": {"tabela": "/out/plus_elegiveis_v2.csv", "id": "chave_dcm",
             "saida": "/out/plus/passada", "tmp": "/scratch/passada_plus"},
}


# ------------------------------------------------------------------ uma imagem
def nome_cache(ident: str) -> str:
    """Nome de arquivo seguro para o cache (a chave do Plus tem barras)."""
    return str(ident).replace("/", "__").removesuffix(".dcm") + ".png"


def _num(v) -> float:
    try:
        return float(v[0] if isinstance(v, (list, tuple)) or hasattr(v, "__len__") and not isinstance(v, str) else v)
    except Exception:
        return float("nan")


def janela_dicom(ds, px: np.ndarray) -> np.ndarray | None:
    """Conversão para 8 bits pela janela gravada no DICOM (depois da LUT de modalidade). None se não houver."""
    centro, largura = _num(getattr(ds, "WindowCenter", None)), _num(getattr(ds, "WindowWidth", None))
    if not (np.isfinite(centro) and np.isfinite(largura)) or largura <= 0:
        return None
    inc = float(getattr(ds, "RescaleSlope", 1) or 1)
    ofs = float(getattr(ds, "RescaleIntercept", 0) or 0)
    v = px.astype(np.float32) * inc + ofs
    x = np.clip((v - (centro - largura / 2)) / largura, 0.0, 1.0)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        x = 1.0 - x
    return (x * 255.0).round().astype(np.uint8)


def processar_dicom(caminho: str, segmentar) -> tuple[dict, np.ndarray]:
    """Lê um DICOM e devolve (colunas da tabela, imagem de 320 px em 8 bits)."""
    import pydicom
    from src.imagem.conversao import para_8bits, redimensionar
    from src.imagem.mascara import razao_pulmoes
    from src.padchest.segunda_passada import dhash
    from src.qualidade.medidas import medidas_qualidade

    ds = pydicom.dcmread(caminho)
    px = ds.pixel_array
    px = px if px.ndim == 2 else px[..., 0]
    mono1 = getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1"
    img8 = para_8bits(px, monochrome1=mono1)
    pulmoes = segmentar(img8)
    roi = pulmoes.max(axis=0)
    esp = getattr(ds, "PixelSpacing", None) or getattr(ds, "ImagerPixelSpacing", None)
    reg = {"linhas": px.shape[0], "colunas": px.shape[1],
           "espacamento_mm": float(esp[0]) if esp else np.nan,
           "fabricante": str(getattr(ds, "Manufacturer", "")), "fotometria": str(getattr(ds, "PhotometricInterpretation", "")),
           "bits": int(getattr(ds, "BitsStored", 0) or 0),
           "md5_pixels": hashlib.md5(np.ascontiguousarray(px).tobytes()).hexdigest(),
           "dhash": dhash(img8), "razao_pulmoes": razao_pulmoes(pulmoes)}
    reg.update(medidas_qualidade(img8, roi, bruta=px))
    j8 = janela_dicom(ds, px)
    reg["janela_disponivel"] = j8 is not None
    if j8 is not None:
        reg.update({f"{k}_janela": v for k, v in medidas_qualidade(j8, roi, bruta=px).items()})
    return reg, redimensionar(img8, 320)


# ------------------------------------------------------------------ download
def baixador(base: str, alvo: pd.DataFrame):
    """Devolve uma função ident -> caminho local do DICOM baixado."""
    import redivis

    def _baixar(arquivo, destino):
        try:
            arquivo.download(destino, progress=False)
        except TypeError:
            arquivo.download(destino)

    tmp = BASES[base]["tmp"]
    if base == "rsna":
        # no RSNA, tabela.file(nome) não acha o arquivo: usar os objetos da listagem
        arquivos = {f.name: f for f in redivis.table("dicom:sjtd").list_files()}
        obter = lambda ident: arquivos[f"{ident}.dcm"]                                   # noqa: E731
    else:
        origem = dict(zip(alvo["chave_dcm"], alvo["path_to_image"].str.split("/").str[0]))
        tabelas = {"train": redivis.table("dicom_train:1934"), "valid": redivis.table("dicom_valid:mg73")}
        obter = lambda ident: tabelas[origem[ident]].file(ident)                         # noqa: E731

    def baixar(ident: str) -> str:
        destino = os.path.join(tmp, hashlib.md5(str(ident).encode()).hexdigest())
        for tentativa in range(4):
            achados = glob.glob(f"{destino}/**/*.dcm", recursive=True)
            if achados:
                return achados[0]
            os.makedirs(destino, exist_ok=True)
            try:
                _baixar(obter(ident), destino)
            except Exception as erro:
                if tentativa == 3:
                    raise RuntimeError(f"falhou ao baixar {ident}") from erro
                time.sleep(2 ** tentativa)
        return glob.glob(f"{destino}/**/*.dcm", recursive=True)[0]

    return baixar


# ------------------------------------------------------------------ a passada
def rodar(base: str, limite: int | None = None) -> None:
    from src.imagem.mascara import segmentar_pulmoes

    cfg = BASES[base]
    saida, tmp = cfg["saida"], cfg["tmp"]
    blocos, cache = f"{saida}/blocos", f"{saida}/c320_8bits"
    for p in (blocos, cache, tmp):
        os.makedirs(p, exist_ok=True)

    alvo = pd.read_csv(cfg["tabela"])
    feitos = set()
    for b in glob.glob(f"{blocos}/*.csv"):
        feitos |= set(pd.read_csv(b, usecols=["id"])["id"].astype(str))
    falta = alvo[~alvo[cfg["id"]].astype(str).isin(feitos)]
    if limite:
        falta = falta.head(limite)
    print(f"{base}: {len(alvo)} elegíveis | {len(feitos)} já feitas | processando {len(falta)}", flush=True)
    if falta.empty:
        return

    baixar = baixador(base, alvo)
    ids = falta[cfg["id"]].astype(str).tolist()
    regs, t0 = [], time.time()

    def salvar():
        if regs:
            pd.DataFrame(regs).to_csv(f"{blocos}/bloco_{time.time_ns()}.csv", index=False)
            regs.clear()

    with ThreadPoolExecutor(8) as ex:
        futuros = {i: ex.submit(baixar, ids[i]) for i in range(min(ADIANTADOS, len(ids)))}
        for i, ident in enumerate(ids):
            if i + ADIANTADOS < len(ids):
                futuros[i + ADIANTADOS] = ex.submit(baixar, ids[i + ADIANTADOS])
            reg = {"id": ident, "base": base}
            caminho = None
            try:
                caminho = futuros.pop(i).result()
                colunas, c320 = processar_dicom(caminho, segmentar_pulmoes)
                cv2.imwrite(f"{cache}/{nome_cache(ident)}", c320)
                reg.update(colunas, abriu=True)
            except Exception as e:
                reg.update(abriu=False, erro=f"{type(e).__name__}: {str(e)[:80]}")
            finally:
                if caminho:
                    shutil.rmtree(os.path.dirname(caminho), ignore_errors=True)
            regs.append(reg)
            if (i + 1) % SALVAR_A_CADA == 0:
                salvar()
                ritmo = (time.time() - t0) / (i + 1)
                print(f"  {i + 1}/{len(ids)} | {ritmo:.2f} s por imagem | faltam {ritmo * (len(ids) - i - 1) / 60:.0f} min",
                      flush=True)
    salvar()
    print(f"{base}: sessão terminou em {(time.time() - t0) / 3600:.1f} h", flush=True)


# ------------------------------------------------------------------ resumo
def juntar(base: str) -> pd.DataFrame:
    blocos = glob.glob(f"{BASES[base]['saida']}/blocos/*.csv")
    if not blocos:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(b) for b in blocos], ignore_index=True).drop_duplicates("id", keep="last")


def resumo(base: str) -> None:
    from src.padchest.segunda_passada import comparar

    cfg = BASES[base]
    df = juntar(base)
    alvo = pd.read_csv(cfg["tabela"])
    ok = df[df["abriu"].astype(bool)] if len(df) else df
    print(f"{base}: {len(df)} de {len(alvo)} processadas | abriram: {len(ok)} | "
          f"máscara vazia: {int((ok['q_aux_roi_vazia'] == 1).sum())} | "
          f"razão entre pulmões < 0,35: {int((ok['razao_pulmoes'] < 0.35).sum())}")
    if len(df) < len(alvo):
        print("Ainda faltam imagens: rode a passada de novo.")
    print("\n=== Resolução por fabricante (mediana)")
    print(ok.groupby("fabricante").agg(imagens=("id", "size"), linhas=("linhas", "median"), colunas=("colunas", "median"),
                                      espacamento_mm=("espacamento_mm", "median")).to_string())
    md5 = ok[ok.duplicated("md5_pixels", keep=False)]
    dh = ok[ok.duplicated("dhash", keep=False)]
    print(f"\nPixels idênticos (MD5): {len(md5)} imagens em {md5['md5_pixels'].nunique()} grupos")
    print(f"Hash de diferença igual: {len(dh)} imagens em {dh['dhash'].nunique()} grupos (conferir antes de excluir)")
    if "q_brilho_janela" in ok.columns and ok["janela_disponivel"].astype(bool).any():
        print("\n=== Percentis × janela do DICOM (imagens com janela)")
        print(comparar(ok).to_string(index=False))
    cache = glob.glob(f"{cfg['saida']}/c320_8bits/*.png")
    tam = sum(os.path.getsize(f) for f in cache) / 1e9
    print(f"\nCache de 320 px: {len(cache)} imagens, {tam:.2f} GB | tabelas: "
          f"{sum(os.path.getsize(b) for b in glob.glob(cfg['saida'] + '/blocos/*.csv')) / 1e6:.1f} MB")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", choices=sorted(BASES), required=True)
    ap.add_argument("--resumo", action="store_true", help="só resume o que já está no /out")
    ap.add_argument("--limite", type=int, default=None, help="processa só as primeiras N imagens que faltam (teste)")
    a = ap.parse_args(argv)
    if a.resumo:
        resumo(a.base)
    else:
        rodar(a.base, a.limite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
