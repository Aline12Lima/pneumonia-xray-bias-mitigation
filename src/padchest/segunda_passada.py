"""Segunda passada do PadChest (F3 › PadChest › Segunda passada).

Baixa de novo os 13 pacotes do recorte (0–11 e 54), um por vez, e calcula em cada imagem, na
resolução original:

- a máscara do pulmão (mesmo procedimento das três bases), feita uma vez, sobre a conversão oficial;
- as oito medidas de qualidade com a **conversão oficial** (percentis 0,5–99,5), colunas ``q_*``;
- as mesmas medidas com a **janela do CSV reescalada** para a escala dos PNG, colunas ``q_*_janela``
  (análise de sensibilidade da conversão; a máscara é a mesma, só a conversão muda);
- um hash de diferença (dHash, 64 bits) para procurar duplicatas quase idênticas (passo 3.10).

Guarda só tabelas: ``/out/padchest/medidas/medidas_pacote_<n>.csv``. Cada zip é apagado depois de
processado. Nenhuma imagem é guardada.

Termos de uso (cláusula 6): o link de download **não** está neste arquivo. Ele vem da variável de
ambiente ``PADCHEST_LINK``, definida só no notebook privado.

Retomada: o notebook gratuito para em 6 h. Ao rodar de novo, pacotes prontos são pulados, e um pacote
interrompido recomeça das imagens que faltam (as parciais ficam em ``/out``). Enquanto um pacote é
processado, o próximo já está sendo baixado.

Uso (no notebook do workflow ``auditoria_padchest``):
    python -m src.padchest.segunda_passada            # processa o que falta
    python -m src.padchest.segunda_passada --resumo   # compara as duas conversões (lê só o /out)
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd

PACOTES = list(range(12)) + [54]
SAIDA = "/out/padchest/medidas"
TMP = "/scratch/padchest2"
RECORTE = "/out/padchest_recorte.csv"
SALVAR_A_CADA = 200
TETO_PNG = 65535


# ------------------------------------------------------------------ conversões
def numero(v) -> float:
    """Primeiro número de um campo do CSV (a janela às vezes vem como lista, ex.: "[2047, 2047]")."""
    m = re.search(r"-?\d+(\.\d+)?", str(v))
    return float(m.group()) if m else float("nan")


def escala_png(bits: float) -> float:
    """Fator da reescala linear dos PNG: 12 bits × 16,004; 10 bits × 64,06 (padchest_final.md §4)."""
    return TETO_PNG / (2 ** int(bits) - 1)


def janela_8bits(g: np.ndarray, centro: float, largura: float, bits: float) -> np.ndarray | None:
    """Converte para 8 bits com a janela do CSV levada para a escala do PNG. None se faltar dado."""
    if not all(np.isfinite([centro, largura, bits])) or largura <= 0:
        return None
    s = escala_png(bits)
    lo, hi = (centro - largura / 2) * s, (centro + largura / 2) * s
    x = np.clip((g.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    return (x * 255.0).round().astype(np.uint8)


def dhash(img8: np.ndarray) -> str:
    """Hash de diferença de 64 bits (texto hexadecimal). Imagens quase iguais dão hashes próximos."""
    p = cv2.resize(img8, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (p[:, 1:] > p[:, :-1]).ravel()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


# ------------------------------------------------------------------ uma imagem
def medir_imagem(g: np.ndarray, linha, segmentar) -> dict:
    """Todas as colunas de uma imagem. ``segmentar(img8) -> (2, H, W)`` com os dois pulmões."""
    from src.imagem.conversao import para_8bits
    from src.imagem.mascara import razao_pulmoes
    from src.qualidade.medidas import medidas_qualidade

    img8 = para_8bits(g)                                   # PadChest: tudo MONOCHROME2, sem inversão
    pulmoes = segmentar(img8)
    roi = pulmoes.max(axis=0)
    reg = {"alt": g.shape[0], "larg": g.shape[1], "dhash": dhash(img8),
           "razao_pulmoes": razao_pulmoes(pulmoes)}
    reg.update(medidas_qualidade(img8, roi, bruta=g))
    j8 = janela_8bits(g, numero(linha.WindowCenter_DICOM), numero(linha.WindowWidth_DICOM),
                      numero(linha.BitsStored_DICOM))
    reg["janela_disponivel"] = j8 is not None
    if j8 is not None:
        reg.update({f"{k}_janela": v for k, v in medidas_qualidade(j8, roi, bruta=g).items()})
    return reg


# ------------------------------------------------------------------ download (mesma lógica do bloco P3b)
def _origem(link: str) -> tuple[str, str]:
    host, token = link.rstrip("/").split("/s/")
    return host, token.split("/")[0]


def url_e_tamanho(link: str, nome: str) -> tuple[str, int]:
    import requests
    host, token = _origem(link)
    u = f"{host}/public.php/dav/files/{token}/{nome}"
    r = requests.get(u, headers={"Range": "bytes=0-0"}, stream=True, timeout=60)
    st, cr = r.status_code, r.headers.get("Content-Range")
    r.close()
    if st != 206 or not cr:
        raise IOError(f"{nome}: status {st}")
    return u, int(cr.split("/")[-1])


def baixar(url: str, total: int, destino: str, partes: int = 4) -> None:
    """Download em 4 partes, confere o status 206 e espera mais a cada falha (versão robusta P3b)."""
    import requests
    with open(destino, "wb") as f:
        f.truncate(total)
    limites = [(i * total // partes, (i + 1) * total // partes - 1) for i in range(partes)]

    def parte(i):
        a, b = limites[i]
        pos, ultimo = a, ""
        for k in range(60):
            if pos > b:
                return
            try:
                with requests.get(url, headers={"Range": f"bytes={pos}-{b}"}, stream=True, timeout=(30, 120)) as r:
                    if r.status_code != 206:
                        raise IOError(f"status {r.status_code}")
                    with open(destino, "r+b") as f:
                        f.seek(pos)
                        for blk in r.iter_content(8 * 1024 * 1024):
                            f.write(blk)
                            pos += len(blk)
            except Exception as e:
                ultimo = f"{type(e).__name__}: {str(e)[:80]}"
                time.sleep(min(15 * (k + 1), 120))
        if pos <= b:
            raise IOError(f"parte {i} incompleta após 60 tentativas; último erro: {ultimo}")

    with ThreadPoolExecutor(partes) as ex:
        list(ex.map(parte, range(partes)))


def baixar_pacote(link: str, n: int) -> str:
    """Baixa o zip do pacote n para o /scratch e devolve o caminho."""
    os.makedirs(TMP, exist_ok=True)
    zipf = f"{TMP}/{n}.zip"
    url, total = url_e_tamanho(link, f"{n}.zip")
    if os.path.exists(zipf) and os.path.getsize(zipf) == total and zipfile.is_zipfile(zipf):
        return zipf
    t0 = time.time()
    try:
        baixar(url, total, zipf)
    except Exception:
        if os.path.exists(zipf):
            os.remove(zipf)
        raise
    print(f"Pacote {n}: baixado ({total / 1e9:.1f} GB em {(time.time() - t0) / 60:.1f} min)", flush=True)
    return zipf


# ------------------------------------------------------------------ um pacote
def processar_pacote(n: int, zipf: str, rec: pd.DataFrame, segmentar) -> None:
    # parciais em blocos separados: as colunas podem variar (imagem sem janela, imagem que não abriu)
    final, pasta = f"{SAIDA}/medidas_pacote_{n}.csv", f"{SAIDA}/parcial_pacote_{n}"
    os.makedirs(pasta, exist_ok=True)
    blocos = sorted(glob.glob(f"{pasta}/*.csv"))
    alvo = rec[rec["ImageDir"] == n]
    feitos = set(pd.concat([pd.read_csv(b) for b in blocos])["ImageID"]) if blocos else set()
    falta = alvo[~alvo["ImageID"].isin(feitos)]
    print(f"Pacote {n}: {len(alvo)} imagens do recorte | {len(feitos)} já feitas | medindo {len(falta)}", flush=True)
    t0, regs = time.time(), []

    def salvar():
        if regs:
            pd.DataFrame(regs).to_csv(f"{pasta}/bloco_{time.time_ns()}.csv", index=False)
            regs.clear()

    with zipfile.ZipFile(zipf) as z:
        nomes = {os.path.basename(m): m for m in z.namelist() if m.lower().endswith(".png")}
        for i, linha in enumerate(falta.itertuples(), 1):
            reg = {"ImageID": linha.ImageID, "ImageDir": n}
            try:
                g = cv2.imdecode(np.frombuffer(z.read(nomes[linha.ImageID]), np.uint8), cv2.IMREAD_UNCHANGED)
                if g is None:
                    raise ValueError("não decodificou")
                g = g if g.ndim == 2 else g[..., 0]
                reg.update(medir_imagem(g, linha, segmentar), abriu=True)
            except Exception as e:
                reg.update(abriu=False, erro=f"{type(e).__name__}: {str(e)[:80]}")
            regs.append(reg)
            if i % SALVAR_A_CADA == 0:
                salvar()
                ritmo = (time.time() - t0) / i
                print(f"  pacote {n}: {i}/{len(falta)} | {ritmo:.2f} s por imagem | "
                      f"faltam {ritmo * (len(falta) - i) / 60:.0f} min", flush=True)
    salvar()
    juntos = pd.concat([pd.read_csv(b) for b in sorted(glob.glob(f"{pasta}/*.csv"))], ignore_index=True)
    juntos.drop_duplicates("ImageID", keep="last").to_csv(final, index=False)
    shutil.rmtree(pasta)
    os.remove(zipf)
    print(f"Pacote {n}: pronto em {(time.time() - t0) / 60:.1f} min | zip apagado", flush=True)


def rodar(link: str, pacotes=PACOTES) -> list[int]:
    """Processa os pacotes que faltam, baixando o próximo enquanto mede o atual. Devolve os que falharam."""
    from src.imagem.mascara import segmentar_pulmoes

    os.makedirs(SAIDA, exist_ok=True)
    rec = pd.read_csv(RECORTE)
    faltam = [n for n in pacotes if not os.path.exists(f"{SAIDA}/medidas_pacote_{n}.csv")]
    print(f"Pacotes prontos: {len(pacotes) - len(faltam)} de {len(pacotes)} | faltam {faltam}", flush=True)
    falhas = []
    with ThreadPoolExecutor(1) as ex:
        futuro = ex.submit(baixar_pacote, link, faltam[0]) if faltam else None
        for k, n in enumerate(faltam):
            try:
                zipf = futuro.result()
            except Exception as e:
                print(f"Pacote {n}: download FALHOU ({type(e).__name__}: {str(e)[:150]})", flush=True)
                falhas.append(n)
                zipf = None
            futuro = ex.submit(baixar_pacote, link, faltam[k + 1]) if k + 1 < len(faltam) else None
            if zipf:
                try:
                    processar_pacote(n, zipf, rec, segmentar_pulmoes)
                except Exception as e:
                    print(f"Pacote {n}: FALHOU ({type(e).__name__}: {str(e)[:150]})", flush=True)
                    falhas.append(n)
    return falhas


# ------------------------------------------------------------------ comparação das duas conversões
def comparar(df: pd.DataFrame) -> pd.DataFrame:
    """Por medida: concordância de postos (Spearman) e diferença mediana entre as duas conversões."""
    from src.qualidade.medidas import PRIMARIAS, SECUNDARIAS

    linhas = []
    for col in PRIMARIAS + SECUNDARIAS:
        par = df[[col, f"{col}_janela"]].dropna() if f"{col}_janela" in df.columns else pd.DataFrame()
        if len(par) < 3:
            continue
        a, b = par[col], par[f"{col}_janela"]
        linhas.append({"medida": col, "n": len(par), "spearman": round(a.corr(b, method="spearman"), 3),
                       "mediana_percentis": round(a.median(), 4), "mediana_janela": round(b.median(), 4),
                       "diferenca_mediana": round((b - a).median(), 4)})
    return pd.DataFrame(linhas)


def resumo() -> None:
    arqs = sorted(glob.glob(f"{SAIDA}/medidas_pacote_*.csv"))
    df = pd.concat([pd.read_csv(a) for a in arqs], ignore_index=True)
    rec = pd.read_csv(RECORTE)[["ImageID", "Manufacturer_DICOM", "Projection", "pos"]]
    df = df.merge(rec, on="ImageID", how="left")
    ok = df[df["abriu"].astype(bool)]
    print(f"Pacotes: {len(arqs)} de 13 | imagens: {len(df)} | abriram: {len(ok)} | "
          f"com janela no CSV: {int(ok['janela_disponivel'].astype(bool).sum())} | "
          f"máscara vazia: {int((ok['q_aux_roi_vazia'] == 1).sum())}")
    print("\n=== Conversão oficial (percentis) × janela do CSV reescalada — todas as imagens")
    print(comparar(ok).to_string(index=False))
    for fab, g in ok.groupby("Manufacturer_DICOM"):
        if len(g) >= 100:
            print(f"\n=== {fab} ({len(g)} imagens)")
            print(comparar(g).to_string(index=False))
    dup = ok[ok.duplicated("dhash", keep=False)]
    print(f"\nHash de diferença igual: {len(dup)} imagens em {dup['dhash'].nunique()} grupos "
          f"(candidatas a duplicata quase idêntica; conferir antes de excluir)")
    tam = sum(os.path.getsize(a) for a in arqs) / 1e6
    print(f"Espaço das tabelas em /out/padchest/medidas: {tam:.1f} MB")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resumo", action="store_true", help="só compara as duas conversões, lendo o /out")
    ap.add_argument("--pacotes", type=int, nargs="*", default=PACOTES)
    a = ap.parse_args(argv)
    if a.resumo:
        resumo()
        return 0
    link = os.environ.get("PADCHEST_LINK")
    if not link:
        raise SystemExit("Defina PADCHEST_LINK no notebook (o link não fica no repositório).")
    inicio = time.time()
    falhas = rodar(link, a.pacotes)
    if falhas:
        print(f"\nFalharam {falhas}; pausa de 2 min e nova tentativa", flush=True)
        time.sleep(120)
        falhas = rodar(link, falhas)
    prontos = sorted(int(re.search(r"_(\d+)\.csv", f).group(1)) for f in glob.glob(f"{SAIDA}/medidas_pacote_*.csv"))
    print(f"\nFim da sessão: {(time.time() - inicio) / 3600:.1f} h | prontos ({len(prontos)} de 13): {prontos}"
          + (f" | ainda falharam: {falhas}" if falhas else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
