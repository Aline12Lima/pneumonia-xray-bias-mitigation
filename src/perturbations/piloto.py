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

Uso (no notebook da base, depois do ``git pull``):
    python -m src.perturbations.piloto --base plus
    python -m src.perturbations.piloto --base plus --resumo      # só relê a tabela
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
PERCENTIS = {"baixo": [25, 10, 5, 1], "alto": [75, 90, 95, 99]}
MEDIDAS = ["q_brilho", "q_contraste", "q_nitidez", "q_ruido"]

BASES = {
    "plus": {"validacao": "/out/plus/listas/validacao.csv", "blocos": "/out/plus/passada/blocos"},
    "rsna": {"validacao": "/out/rsna/listas/validacao.csv", "blocos": "/out/rsna/passada/blocos"},
}


def _saida(base: str) -> str:
    return f"/out/{base}/eixo_b_piloto"


def _semente_imagem(ident: str) -> int:
    return int(hashlib.md5(ident.encode()).hexdigest()[:8], 16)


def _medir(img8: np.ndarray, roi: np.ndarray) -> dict[str, float]:
    from src.qualidade.medidas import medidas_qualidade
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # em 320 px a função avisa que não é a resolução original
        q = medidas_qualidade(img8, roi)
    return {k: q[k] for k in MEDIDAS}


def naturais(base: str) -> pd.DataFrame:
    """Medidas da validação, na resolução original, tiradas da passada completa (âncora)."""
    from src.qualidade.fabricante import normalizar_fabricante
    from src.passada.passada_completa import juntar
    val = pd.read_csv(BASES[base]["validacao"], usecols=["id"])["id"].astype(str)
    df = juntar(base)
    df = df[df["abriu"].astype(bool) & df["id"].astype(str).isin(val)].copy()
    df["fabricante"] = df["fabricante"].map(normalizar_fabricante)
    return df


def sortear(nat: pd.DataFrame) -> pd.DataFrame:
    """50 imagens; no Plus, metade Agfa e metade GE (resoluções diferentes)."""
    fab = nat["fabricante"].value_counts()
    grupos = [f for f in ("Agfa", "GE") if fab.get(f, 0) >= N_IMAGENS // 2]
    if len(grupos) == 2:
        partes = [nat[nat["fabricante"] == f].sample(N_IMAGENS // 2, random_state=SEMENTE) for f in grupos]
        return pd.concat(partes).reset_index(drop=True)
    return nat.sample(N_IMAGENS, random_state=SEMENTE).reset_index(drop=True)


def uma_imagem(caminho: str, ident: str, segmentar) -> tuple[list[dict], dict]:
    import pydicom
    from src.imagem.conversao import para_8bits, redimensionar

    t0 = time.time()
    ds = pydicom.dcmread(caminho)
    px = ds.pixel_array
    px = px if px.ndim == 2 else px[..., 0]
    img8 = para_8bits(px, monochrome1=getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1")
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
        for p in GRADE[tipo]:
            d = aplicar(img8, tipo, p, sem)
            d320 = redimensionar(d, 320)
            linha(tipo, float(p), d, d320)
            figura[(tipo, p)] = d320
    for tipo in FOTOMETRICAS:                        # direto na imagem de 320 px
        for p in GRADE[tipo]:
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
        params = [GRADE[tipo][i] for i in np.linspace(0, len(GRADE[tipo]) - 1, 5).round().astype(int)]
        painel = [recorte(figura["natural"])] + [recorte(figura[(tipo, p)]) for p in params]
        painel = [cv2.putText(cv2.cvtColor(im, cv2.COLOR_GRAY2BGR), txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                              0.5, (0, 0, 255), 1) for im, txt in
                  zip(painel, ["natural"] + [f"{tipo} {p}" for p in params])]
        linhas.append(np.hstack(painel))
    cv2.imwrite(destino, np.vstack(linhas))


def rodar(base: str) -> None:
    from src.imagem.mascara import segmentar
    from src.passada.passada_completa import BASES as BP, baixador

    saida = _saida(base)
    os.makedirs(saida, exist_ok=True)
    nat = naturais(base)
    amostra = sortear(nat)
    amostra[["id", "fabricante"]].to_csv(f"{saida}/amostra.csv", index=False)
    print(f"{base}: {len(nat)} imagens da validação com medidas | piloto em {len(amostra)}", flush=True)
    alvo = pd.read_csv(BP[base]["tabela"])
    alvo = alvo[alvo[BP[base]["id"]].astype(str).isin(amostra["id"])]
    baixar = baixador(base, alvo)
    todas, t0 = [], time.time()
    for i, ident in enumerate(amostra["id"]):
        caminho = None
        try:
            caminho = baixar(ident)
            linhas, fig = uma_imagem(caminho, ident, segmentar)
            todas += linhas
            if i == 0:
                montar_figura(fig, f"{saida}/figura_espaciais.png")
        except Exception as e:
            print(f"  {ident}: {type(e).__name__}: {str(e)[:80]}", flush=True)
        finally:
            if caminho:
                shutil.rmtree(os.path.dirname(caminho), ignore_errors=True)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(amostra)} | {(time.time() - t0) / (i + 1):.1f} s por imagem", flush=True)
    df = pd.DataFrame(todas).merge(amostra[["id", "fabricante"]], on="id", how="left")
    df.to_csv(f"{saida}/piloto.csv", index=False)
    print(f"Tabela: {saida}/piloto.csv | figura: {saida}/figura_espaciais.png\n", flush=True)
    resumo(base)


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
        g = df[df["tipo"] == tipo].groupby("parametro")
        espaco = "orig" if tipo in ESPACIAIS else "320"
        med = g[f"{medida}_{espaco}"].median()
        lados = ["baixo", "alto"] if lado == "dois lados" else [lado]
        for l in lados:
            for pc in PERCENTIS[l]:
                alvo = float(np.percentile(nat[medida].dropna(), pc))
                linhas.append({"degradacao": tipo, "medida": medida, "percentil": f"p{pc}", "alvo": alvo,
                               "parametro": interpolar_parametro(med.index, med.values, alvo)})
    return pd.DataFrame(linhas)


def resumo(base: str) -> None:
    saida = _saida(base)
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
    a = ap.parse_args(argv)
    if a.resumo:
        resumo(a.base)
    else:
        rodar(a.base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
