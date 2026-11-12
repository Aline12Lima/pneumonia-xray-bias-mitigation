"""Piloto das degradações do eixo B (F4 › Todas › Degradações do eixo B, opção C).

Roda em **50 imagens da validação** (nunca do teste) e responde se a opção C é viável:

1. **tempo** por imagem e **espaço** das versões degradadas em 320 px;
2. **sobrevivência**: quanto de cada degradação espacial aplicada na resolução original ainda
   aparece depois da redução para 320 px (o que o modelo vê);
3. **ancoragem**: que parâmetro leva a mediana da medida aos percentis p25, p10, p5 e p1
   (ou p75, p90, p95 e p99) da validação da própria base, e se a grade alcança esses alvos;
4. **fotométricas**: se brilho e contraste medidos em 320 px acompanham os medidos no original
   (justifica aplicar gama, contraste e brilho em 320 px).

Nada depende de modelo nem de rótulo. Grava só em ``/out/<base>/eixo_b_piloto/``
(uma tabela e uma figura; as imagens degradadas não são guardadas, só o tamanho delas).

**Ancoragem da F8** (``--f8``; F6 › Eixo B › Ancoragem da F8): a mesma rotina com a grade fina perto
do neutro para desfoque, resolução e ruído (lição do ensaio: a resolução 1,5 já derrubava a nitidez
para além do p1), gravada à parte em ``/out/<base>/eixo_b_ancoragem_f8/``, sem tocar na do ensaio.
No **PadChest**, que não tem validação, os percentis são os da própria base (as medidas das 24.773
imagens do recorte) e as 50 imagens vêm do pacote 0 (metade de cada um dos dois fabricantes), lidas
do zip (precisa de ``PADCHEST_LINK``). Continua sem rótulo e sem modelo.

Uso (no notebook da base, depois do ``git pull``):
    python -m src.perturbations.piloto --base plus
    python -m src.perturbations.piloto --base plus --resumo      # só relê a tabela
    python -m src.perturbations.piloto --base plus --f8          # ancoragem da F8 (grade fina)
    python -m src.perturbations.piloto --base padchest --f8      # idem, no PadChest (pacote 0)
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import time
import warnings

import cv2
import numpy as np
import pandas as pd

from src.perturbations.degradacoes import ESPACIAIS, FOTOMETRICAS, MEDIDA_ALVO, aplicar, interpolar_parametro

N_IMAGENS = 50
SEMENTE = 2026

GRADE = {
    "desfoque": [0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24],       # sigma em pixels da imagem original
    "ruido": [1, 2, 4, 6, 8, 12, 16, 24],                  # desvio em tons de 8 bits
    "resolucao": [1.5, 2, 3, 4, 6, 8, 12, 16],             # fator de redução e reampliação
    "gama": [0.5, 0.67, 0.8, 0.9, 1.1, 1.25, 1.5, 2.0],
    "contraste": [0.4, 0.55, 0.7, 0.85, 0.95],
    "brilho": [-60, -40, -25, -10, 10, 25, 40, 60],
}
# grade da F8: a mesma, com passos perto do neutro onde a medida cai rápido (lição do ensaio)
GRADE_F8 = {
    "desfoque": [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24],
    "ruido": [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24],
    "resolucao": [1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.75, 2, 2.5, 3, 4, 6, 8, 12, 16],
    "gama": [0.4, 0.45, 0.5, 0.6, 0.67, 0.75, 0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.25, 1.35, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0],
    "contraste": [0.4, 0.5, 0.55, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975],
    "brilho": [-80, -70, -60, -50, -40, -30, -25, -20, -15, -10, -5, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80],
}
PACOTE_PILOTO_PADCHEST = 0
PERCENTIS = {"baixo": [25, 10, 5, 1], "alto": [75, 90, 95, 99]}
NEUTRO = {"gama": 1.0, "contraste": 1.0, "brilho": 0.0, "desfoque": 0.0, "ruido": 0.0, "resolucao": 1.0}
MEDIDAS = ["q_brilho", "q_contraste", "q_nitidez", "q_ruido"]

BASES = {
    "plus": {"validacao": "/out/plus/listas/validacao.csv", "blocos": "/out/plus/passada/blocos"},
    "rsna": {"validacao": "/out/rsna/listas/validacao.csv", "blocos": "/out/rsna/passada/blocos"},
    "padchest": {},                                     # sem validação: percentis da base inteira (só medidas)
}


def _saida(base: str, f8: bool = False) -> str:
    return f"/out/{base}/eixo_b_ancoragem_f8" if f8 else f"/out/{base}/eixo_b_piloto"


def _semente_imagem(ident: str) -> int:
    return int(hashlib.md5(ident.encode()).hexdigest()[:8], 16)


def _medir(img8: np.ndarray, roi: np.ndarray) -> dict[str, float]:
    from src.qualidade.medidas import medidas_qualidade
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # em 320 px a função avisa que não é a resolução original
        q = medidas_qualidade(img8, roi)
    return {k: q[k] for k in MEDIDAS}


def naturais(base: str) -> pd.DataFrame:
    """Medidas da validação, na resolução original, tiradas da passada completa (âncora).
    PadChest: as medidas de todas as imagens do recorte (segunda passada), sem rótulo."""
    from src.qualidade.fabricante import normalizar_fabricante
    if base == "padchest":
        import glob
        med = pd.concat([pd.read_csv(a) for a in sorted(glob.glob("/out/padchest/medidas/medidas_pacote_*.csv"))],
                        ignore_index=True)
        rec = pd.read_csv("/out/padchest_recorte.csv", usecols=["ImageID", "Manufacturer_DICOM"])
        df = med.merge(rec, on="ImageID", how="left").rename(columns={"ImageID": "id"})   # ImageDir já vem das medidas
        df = df[df["abriu"].astype(bool)].copy()
        df["fabricante"] = df["Manufacturer_DICOM"].map(normalizar_fabricante)
        return df
    from src.passada.passada_completa import juntar
    val = pd.read_csv(BASES[base]["validacao"], usecols=["id"])["id"].astype(str)
    df = juntar(base)
    df = df[df["abriu"].astype(bool) & df["id"].astype(str).isin(val)].copy()
    df["fabricante"] = df["fabricante"].map(normalizar_fabricante)
    return df


def sortear(nat: pd.DataFrame) -> pd.DataFrame:
    """50 imagens; metade de cada um dos dois fabricantes mais frequentes (Plus: Agfa e GE)."""
    fab = nat["fabricante"].value_counts()
    grupos = [f for f in ("Agfa", "GE") if fab.get(f, 0) >= N_IMAGENS // 2]
    if len(grupos) < 2:
        grupos = [f for f in fab.index[:2] if fab[f] >= N_IMAGENS // 2]
    if len(grupos) == 2:
        partes = [nat[nat["fabricante"] == f].sample(N_IMAGENS // 2, random_state=SEMENTE) for f in grupos]
        return pd.concat(partes).reset_index(drop=True)
    return nat.sample(N_IMAGENS, random_state=SEMENTE).reset_index(drop=True)


def carregar(origem) -> np.ndarray:
    """Imagem em 8 bits pela conversão oficial: ``origem`` é o caminho de um DICOM (Plus, RSNA) ou a
    matriz de 16 bits de um PNG do PadChest (tudo MONOCHROME2)."""
    from src.imagem.conversao import para_8bits
    if isinstance(origem, np.ndarray):
        return para_8bits(origem if origem.ndim == 2 else origem[..., 0])
    import pydicom
    ds = pydicom.dcmread(origem)
    px = ds.pixel_array
    px = px if px.ndim == 2 else px[..., 0]
    return para_8bits(px, monochrome1=getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1")


def uma_imagem(origem, ident: str, segmentar, grade: dict | None = None) -> tuple[list[dict], dict]:
    from src.imagem.conversao import redimensionar

    grade = grade or GRADE
    t0 = time.time()
    img8 = carregar(origem)
    roi = segmentar(img8)
    roi320 = cv2.resize(roi.astype(np.uint8), (320, 320), interpolation=cv2.INTER_NEAREST)
    c320 = redimensionar(img8, 320)
    sem = _semente_imagem(ident)
    linhas = []

    def linha(tipo, par, orig, c):
        r = {"id": ident, "tipo": tipo, "parametro": par, "linhas": img8.shape[0], "colunas": img8.shape[1],
             "png_kb": len(cv2.imencode(".png", c)[1]) / 1024}
        r.update({f"{k}_320": v for k, v in _medir(c, roi320).items()})
        if orig is not None:
            r.update({f"{k}_orig": v for k, v in _medir(orig, roi).items()})
        linhas.append(r)

    linha("natural", 0.0, img8, c320)
    figura = {"natural": c320}
    for tipo in ESPACIAIS:                           # na resolução original, depois reduz para 320 px
        for p in grade[tipo]:
            d = aplicar(img8, tipo, p, sem)
            d320 = redimensionar(d, 320)
            linha(tipo, float(p), d, d320)
            figura[(tipo, p)] = d320
    for tipo in FOTOMETRICAS:                        # direto na imagem de 320 px
        for p in grade[tipo]:
            linha(tipo, float(p), None, aplicar(c320, tipo, p, sem))
    for r in linhas:
        r["segundos_imagem"] = time.time() - t0
    return linhas, figura


def montar_figura(figura: dict, destino: str) -> None:
    """Uma imagem: por linha, uma degradação espacial; recorte central de 160 px ampliado (para ver o detalhe)."""
    def recorte(c):
        x = c[80:240, 80:240]
        return cv2.resize(x, (240, 240), interpolation=cv2.INTER_NEAREST)
    linhas = []
    for tipo in ESPACIAIS:
        g = [p for p in figura if isinstance(p, tuple) and p[0] == tipo]
        todos = sorted(p for _, p in g)
        params = [todos[i] for i in np.linspace(0, len(todos) - 1, 5).round().astype(int)]
        painel = [recorte(figura["natural"])] + [recorte(figura[(tipo, p)]) for p in params]
        painel = [cv2.putText(cv2.cvtColor(im, cv2.COLOR_GRAY2BGR), txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                              0.5, (0, 0, 255), 1) for im, txt in
                  zip(painel, ["natural"] + [f"{tipo} {p}" for p in params])]
        linhas.append(np.hstack(painel))
    cv2.imwrite(destino, np.vstack(linhas))


def imagens_da_amostra(base: str, ids: list[str]):
    """Gera (ident, origem) para ``carregar``: DICOM baixado (Plus, RSNA) ou matriz do zip (PadChest)."""
    if base == "padchest":
        import zipfile
        from src.padchest.segunda_passada import baixar_pacote
        link = os.environ.get("PADCHEST_LINK")
        if not link:
            raise SystemExit("Defina PADCHEST_LINK no notebook (o link não fica no repositório).")
        zipf = baixar_pacote(link, PACOTE_PILOTO_PADCHEST)
        try:
            with zipfile.ZipFile(zipf) as z:
                nomes = {os.path.basename(m): m for m in z.namelist() if m.lower().endswith(".png")}
                for ident in ids:
                    g = cv2.imdecode(np.frombuffer(z.read(nomes[ident]), np.uint8), cv2.IMREAD_UNCHANGED)
                    yield ident, g
        finally:
            os.remove(zipf)
        return
    from src.passada.passada_completa import BASES as BP, baixador
    alvo = pd.read_csv(BP[base]["tabela"])
    alvo = alvo[alvo[BP[base]["id"]].astype(str).isin(ids)]
    baixar = baixador(base, alvo)
    for ident in ids:
        caminho = baixar(ident)
        try:
            yield ident, caminho
        finally:
            shutil.rmtree(os.path.dirname(caminho), ignore_errors=True)


def rodar(base: str, f8: bool = False) -> None:
    from src.imagem.mascara import segmentar

    saida = _saida(base, f8)
    os.makedirs(saida, exist_ok=True)
    nat = naturais(base)
    candidatas = nat[nat["ImageDir"] == PACOTE_PILOTO_PADCHEST] if base == "padchest" else nat
    amostra = sortear(candidatas)
    amostra[["id", "fabricante"]].to_csv(f"{saida}/amostra.csv", index=False)
    print(f"{base}: {len(nat)} imagens com medidas (âncora) | piloto em {len(amostra)} | "
          f"grade {'da F8 (fina)' if f8 else 'do ensaio'}", flush=True)
    todas, t0 = [], time.time()
    it = imagens_da_amostra(base, amostra["id"].astype(str).tolist())
    for i in range(len(amostra)):
        try:
            ident, origem = next(it)
        except StopIteration:
            break
        except Exception as e:
            print(f"  imagem {i}: {type(e).__name__}: {str(e)[:80]}", flush=True)
            continue
        try:
            linhas, fig = uma_imagem(origem, ident, segmentar, GRADE_F8 if f8 else GRADE)
            todas += linhas
            if not os.path.exists(f"{saida}/figura_espaciais.png"):
                montar_figura(fig, f"{saida}/figura_espaciais.png")
        except Exception as e:
            print(f"  {ident}: {type(e).__name__}: {str(e)[:80]}", flush=True)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(amostra)} | {(time.time() - t0) / (i + 1):.1f} s por imagem", flush=True)
    it.close()
    df = pd.DataFrame(todas).merge(amostra[["id", "fabricante"]], on="id", how="left")
    df.to_csv(f"{saida}/piloto.csv", index=False)
    print(f"Tabela: {saida}/piloto.csv | figura: {saida}/figura_espaciais.png\n", flush=True)
    resumo(base, f8)


def tabela_resumo(df: pd.DataFrame, nat: pd.DataFrame) -> pd.DataFrame:
    """Por degradação e parâmetro: mediana da medida-alvo no original e em 320 px, e o efeito em
    desvios-padrão das mesmas imagens naturais, medidas no mesmo espaço (original ou 320 px).
    "sobrevive" = efeito em 320 px ÷ efeito no original: perto de 1, a degradação chega inteira ao
    modelo; perto de 0, a redução para 320 px a apaga."""
    n = df[df["tipo"] == "natural"]
    linhas = []
    for (tipo, par), g in df[df["tipo"] != "natural"].groupby(["tipo", "parametro"]):
        medida = MEDIDA_ALVO[tipo][0]
        base320 = n.set_index("id")[f"{medida}_320"]
        d320 = (g.set_index("id")[f"{medida}_320"] - base320).median() / base320.std()
        reg = {"degradacao": tipo, "parametro": par, "medida": medida,
               "mediana_320": g[f"{medida}_320"].median(), "efeito_320_dp": d320,
               "png_kb": g["png_kb"].mean()}
        if f"{medida}_orig" in g and g[f"{medida}_orig"].notna().any():
            baseo = n.set_index("id")[f"{medida}_orig"]
            reg["mediana_orig"] = g[f"{medida}_orig"].median()
            reg["efeito_orig_dp"] = (g.set_index("id")[f"{medida}_orig"] - baseo).median() / baseo.std()
            reg["sobrevive"] = reg["efeito_320_dp"] / reg["efeito_orig_dp"] if reg["efeito_orig_dp"] else np.nan
        linhas.append(reg)
    return pd.DataFrame(linhas)


def ancoragem(df: pd.DataFrame, nat: pd.DataFrame) -> pd.DataFrame:
    """Parâmetro que leva a mediana da medida a cada percentil da validação (NaN = a grade não alcança)."""
    linhas = []
    for tipo, (medida, lado) in MEDIDA_ALVO.items():
        if not (df["tipo"] == tipo).any():
            continue
        espaco = "orig" if tipo in ESPACIAIS else "320"
        # a imagem natural é o ponto "sem degradação" da grade (gama 1, contraste 1, brilho 0, desfoque 0,
        # ruído 0, resolução 1): sem ele, alvos entre a natural e o primeiro passo da grade ficavam de fora
        nat_pts = df[df["tipo"] == "natural"].assign(tipo=tipo, parametro=NEUTRO[tipo])
        g = pd.concat([nat_pts, df[df["tipo"] == tipo]]).groupby("parametro")
        med = g[f"{medida}_{espaco}"].median()
        lados = ["baixo", "alto"] if lado == "dois lados" else [lado]
        for l in lados:
            for pc in PERCENTIS[l]:
                alvo = float(np.percentile(nat[medida].dropna(), pc))
                linhas.append({"degradacao": tipo, "medida": medida, "percentil": f"p{pc}", "alvo": alvo,
                               "parametro": interpolar_parametro(med.index, med.values, alvo)})
    return pd.DataFrame(linhas)


def resumo(base: str, f8: bool = False) -> None:
    saida = _saida(base, f8)
    df = pd.read_csv(f"{saida}/piloto.csv")
    nat = naturais(base)
    n = df[df["tipo"] == "natural"]
    t = n["segundos_imagem"]
    print(f"=== {base}: {n['id'].nunique()} imagens | {t.median():.1f} s por imagem (mediana; todas as versões)")
    print(f"Resolução original (mediana): {n['linhas'].median():.0f} × {n['colunas'].median():.0f}"
          f" | redução até 320 px: cerca de {n['linhas'].median() / 320:.1f} vezes")
    print("\n=== Fotométricas: medida em 320 px × no original (imagens naturais)")
    for m in ("q_brilho", "q_contraste"):
        r = n[[f"{m}_orig", f"{m}_320"]].corr(method="spearman").iloc[0, 1]
        dif = (n[f"{m}_320"] - n[f"{m}_orig"]).abs().median()
        print(f"  {m}: concordância de postos {r:.3f} | diferença mediana {dif:.2f} tons")
    tab = tabela_resumo(df, nat)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\n=== Espaciais: efeito no original e em 320 px (em desvios-padrão das imagens naturais)")
        print(tab[tab["degradacao"].isin(ESPACIAIS)][["degradacao", "parametro", "medida", "efeito_orig_dp",
                                                       "efeito_320_dp", "sobrevive", "png_kb"]]
              .round(3).to_string(index=False))
        print("\n=== Fotométricas (em 320 px)")
        print(tab[tab["degradacao"].isin(FOTOMETRICAS)][["degradacao", "parametro", "medida", "mediana_320",
                                                          "efeito_320_dp"]].round(3).to_string(index=False))
        anc = ancoragem(df, nat)
        anc.to_csv(f"{saida}/ancoragem.csv", index=False)
        print("\n=== Ancoragem nos percentis da validação (parâmetro vazio = a grade não alcança)")
        print(anc.round(3).to_string(index=False))
    kb = df[df["tipo"].isin(ESPACIAIS)]["png_kb"].mean()
    print(f"\n=== Projeção para a F8: {kb:.0f} KB por versão espacial em 320 px; com 12 versões espaciais "
          f"(3 degradações × 4 níveis), cerca de {kb * 12 / 1024:.2f} MB por imagem da amostra")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", choices=sorted(BASES), required=True)
    ap.add_argument("--resumo", action="store_true", help="só relê a tabela do piloto")
    ap.add_argument("--f8", action="store_true", help="ancoragem da F8 (grade fina), em eixo_b_ancoragem_f8/")
    a = ap.parse_args(argv)
    if a.base == "padchest" and not a.f8:
        raise SystemExit("no PadChest só existe a ancoragem da F8 (--f8)")
    if a.resumo:
        resumo(a.base, a.f8)
    else:
        rodar(a.base, a.f8)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
