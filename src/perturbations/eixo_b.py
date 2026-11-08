"""Eixo B: perturbação controlada (protocolo §8.4; F4 › Todas › Degradações do eixo B; F6 › Eixo B).

Roda no notebook de cada base, em duas etapas:

``--gerar`` (CPU): sorteia a amostra, baixa o DICOM, converte para 8 bits, segmenta a máscara e grava
em 320 px a imagem natural e cada versão degradada:
- fotométricas (gama, contraste, brilho): aplicadas à imagem de 320 px;
- espaciais (desfoque, ruído, resolução): aplicadas na resolução original e depois reduzidas;
- níveis: os parâmetros da ancoragem (no ensaio, a do piloto de 30/10, ``eixo_b_piloto/ancoragem.csv``);
  níveis que a grade não alcança ficam de fora e são listados;
- erro de ancoragem: a medida-alvo é medida em cada versão (no original para as espaciais, em 320 px
  para as fotométricas) e comparada com o percentil-alvo.
Retomada: cada imagem pronta vai para o manifesto; rodar de novo pula as que já estão lá.

``--predizer`` (GPU): baixa nada; usa os 15 modelos de ``/out/modelos`` (``src.cache.modelos``) e
grava a tabela no formato do schema, com ``degradacao`` e ``nivel``.

Amostra: no ensaio (``--particao validacao``), 100 positivos e 400 negativos da validação, sorteados
por paciente (semente 2026; F6 › Eixo B › Amostra do ensaio). O teste só depois do destrancamento.

Uso (notebook da base):
    python -m src.perturbations.eixo_b --base plus --gerar                 # CPU
    python -m src.perturbations.eixo_b --base plus --predizer              # GPU
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import warnings

import cv2
import numpy as np
import pandas as pd

from src.perturbations.degradacoes import ESPACIAIS, FOTOMETRICAS, MEDIDA_ALVO, aplicar

SEMENTE = 2026
POSITIVOS_ENSAIO, NEG_POR_POS = 100, 4
FONTE = {"plus": "chexpert_plus", "rsna": "rsna"}


def _pasta(base: str, particao: str) -> str:
    return f"/out/{base}/eixo_b/{particao}"


def _semente_imagem(ident: str) -> int:
    return int(hashlib.md5(ident.encode()).hexdigest()[:8], 16)


def sortear(tab: pd.DataFrame, positivos: int | None, neg_por_pos: int = NEG_POR_POS) -> pd.DataFrame:
    """Uma imagem por paciente sorteado; ``positivos=None`` = todos os positivos (teste)."""
    rng = np.random.default_rng(SEMENTE)

    def por_paciente(d: pd.DataFrame, n: int | None) -> pd.DataFrame:
        pacs = d["paciente"].unique()
        pacs = pacs[rng.permutation(len(pacs))]
        if n is not None:
            pacs = pacs[:n]
        escolhidas = d[d["paciente"].isin(pacs)].groupby("paciente", sort=False).sample(1, random_state=SEMENTE)
        return escolhidas
    pos = tab[tab["rotulo"] == 1] if positivos is None else por_paciente(tab[tab["rotulo"] == 1], positivos)
    neg = por_paciente(tab[tab["rotulo"] == 0], len(pos) * neg_por_pos)
    return pd.concat([pos, neg]).reset_index(drop=True)


def niveis(base: str, ancoragem: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    anc = pd.read_csv(ancoragem or f"/out/{base}/eixo_b_piloto/ancoragem.csv")
    fora = [f"{r.degradacao} {r.percentil}" for r in anc.itertuples() if pd.isna(r.parametro)]
    return anc.dropna(subset=["parametro"]).reset_index(drop=True), fora


def _medir(img8, roi, medida):
    from src.qualidade.medidas import medidas_qualidade
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(medidas_qualidade(img8, roi)[medida])


def uma_imagem(caminho_dcm: str, ident: str, niv: pd.DataFrame, pasta_img: str, segmentar) -> list[dict]:
    import pydicom
    from src.imagem.conversao import para_8bits, redimensionar
    from src.passada.passada_completa import nome_cache

    ds = pydicom.dcmread(caminho_dcm)
    px = ds.pixel_array
    px = px if px.ndim == 2 else px[..., 0]
    img8 = para_8bits(px, monochrome1=getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1")
    roi = segmentar(img8)
    roi320 = cv2.resize(roi.astype(np.uint8), (320, 320), interpolation=cv2.INTER_NEAREST)
    c320 = redimensionar(img8, 320)
    raiz = nome_cache(ident).removesuffix(".png")
    sem = _semente_imagem(ident)
    linhas = []

    def gravar(img, tipo, nivel, parametro, medida, valor):
        arq = os.path.join(pasta_img, f"{raiz}__{tipo}__{nivel}.png")
        cv2.imwrite(arq, img)
        linhas.append({"id": ident, "degradacao": tipo, "nivel": nivel, "parametro": parametro,
                       "caminho": arq, "medida": medida, "valor_medido": valor})

    gravar(c320, "nenhuma", "0", 0.0, "", np.nan)
    for r in niv.itertuples():
        medida = MEDIDA_ALVO[r.degradacao][0]
        if r.degradacao in ESPACIAIS:
            d = aplicar(img8, r.degradacao, float(r.parametro), sem)
            valor = _medir(d, roi, medida) if roi.any() else np.nan
            d = redimensionar(d, 320)
        else:
            d = aplicar(c320, r.degradacao, float(r.parametro), sem)
            valor = _medir(d, roi320, medida) if roi320.any() else np.nan
        gravar(d, r.degradacao, r.percentil, float(r.parametro), medida, valor)
    return linhas


def gerar(base: str, particao: str, limite: int | None = None) -> int:
    from src.imagem.mascara import segmentar
    from src.passada.passada_completa import BASES as BP, baixador
    from src.treino.carregador import ler_particao

    if particao != "validacao":
        raise ValueError("só a validação no ensaio geral; o teste entra depois do destrancamento (passo 6.7)")
    pasta = _pasta(base, particao)
    pasta_img = os.path.join(pasta, "imagens")
    os.makedirs(pasta_img, exist_ok=True)
    tab = ler_particao(base, particao)
    amostra = sortear(tab, POSITIVOS_ENSAIO)
    amostra.to_csv(os.path.join(pasta, "amostra.csv"), index=False)
    niv, fora = niveis(base)
    niv.to_csv(os.path.join(pasta, "niveis.csv"), index=False)
    print(f"{base} | {particao}: amostra de {len(amostra)} imagens ({int(amostra['rotulo'].sum())} positivas, "
          f"{amostra['paciente'].nunique()} pacientes) | {len(niv)} níveis | fora da grade: {fora or 'nenhum'}", flush=True)
    man_arq = os.path.join(pasta, "manifesto.csv")
    feitas = set(pd.read_csv(man_arq, usecols=["id"])["id"].astype(str)) if os.path.exists(man_arq) else set()
    faltam = [i for i in amostra["id"].astype(str) if i not in feitas]
    if limite:
        faltam = faltam[:limite]
    print(f"  já prontas: {len(feitas)} | faltam {len(faltam)}", flush=True)
    alvo = pd.read_csv(BP[base]["tabela"])
    alvo = alvo[alvo[BP[base]["id"]].astype(str).isin(faltam)]
    baixar = baixador(base, alvo)
    t0, erros = time.time(), 0
    for k, ident in enumerate(faltam):
        caminho = None
        try:
            caminho = baixar(ident)
            linhas = uma_imagem(caminho, ident, niv, pasta_img, segmentar)
            pd.DataFrame(linhas).to_csv(man_arq, mode="a", header=not os.path.exists(man_arq), index=False)
        except Exception as e:
            erros += 1
            print(f"  {ident}: {type(e).__name__}: {str(e)[:100]}", flush=True)
        finally:
            if caminho:
                shutil.rmtree(os.path.dirname(caminho), ignore_errors=True)
        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(faltam)} | {(time.time() - t0) / (k + 1):.1f} s por imagem", flush=True)
    return resumo_geracao(base, particao, erros)


def resumo_geracao(base: str, particao: str, erros: int = 0) -> int:
    pasta = _pasta(base, particao)
    man = pd.read_csv(os.path.join(pasta, "manifesto.csv"))
    amostra = pd.read_csv(os.path.join(pasta, "amostra.csv"))
    niv = pd.read_csv(os.path.join(pasta, "niveis.csv"))
    print(f"\nManifesto: {man['id'].nunique()} de {len(amostra)} imagens | {len(man)} versões | erros nesta rodada: {erros}")
    print("Erro de ancoragem (mediana da medida na amostra × alvo do percentil):")
    for r in niv.itertuples():
        g = man[(man["degradacao"] == r.degradacao) & (man["nivel"] == r.percentil)]
        med = g["valor_medido"].median()
        print(f"  {r.degradacao:9s} {r.percentil:4s} | parâmetro {r.parametro:8.3f} | alvo {r.alvo:9.3f} | "
              f"mediana {med:9.3f} | erro {med - r.alvo:+8.3f}")
    completo = man["id"].nunique() == len(amostra)
    print("RESULTADO:", "geração completa" if completo else "INCOMPLETA — rode a mesma célula de novo")
    return 0 if completo else 1


def predizer_eixo_b(base: str, particao: str, modelos: str = "/out/modelos") -> int:
    from src.predicoes.pos_treino import SEMENTES, TAXA
    from src.predicoes.predizer import predizer
    from src.predicoes.schema import validar
    from src.predicoes.tabela import para_schema

    pasta = _pasta(base, particao)
    man = pd.read_csv(os.path.join(pasta, "manifesto.csv"), dtype={"id": str, "nivel": str})
    amostra = pd.read_csv(os.path.join(pasta, "amostra.csv"), dtype={"id": str, "paciente": str})
    if man["id"].nunique() != len(amostra):
        raise RuntimeError("a geração não terminou: rode --gerar até 'geração completa'")
    img = man.merge(amostra[["id", "paciente", "rotulo"]], on="id").assign(fonte=base)
    tabelas, problemas = [], []
    for m in TAXA:
        for s in SEMENTES:
            ex = f"{m}_s{s}_lr{TAXA[m]:g}"
            pred = predizer(os.path.join(modelos, ex), img)
            pred["degradacao"], pred["nivel"] = img["degradacao"].to_numpy(), img["nivel"].to_numpy()
            res = json.load(open(os.path.join(modelos, ex, "resumo.json"), encoding="utf-8"))
            t = para_schema(pred, res, os.path.join(modelos, ex), [base], dados="/out", particao=particao)
            erros = validar(t)
            if erros:
                problemas.append((ex, erros))
            tabelas.append(t)
            nat = pred[pred["degradacao"] == "nenhuma"]
            print(f"{ex}: {len(t)} linhas | prob. média natural {nat['prob'].mean():.4f}"
                  + (f" | ERROS: {erros}" if erros else ""), flush=True)
    tab = pd.concat(tabelas, ignore_index=True)
    destino = os.path.join(pasta, "tabela_eixo_b.csv")
    tab.to_csv(destino, index=False)
    print(f"\n{len(tab)} linhas em {destino}")
    print("RESULTADO:", "OK" if not problemas else f"PROBLEMAS: {problemas[:2]}")
    return 0 if not problemas else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", choices=sorted(FONTE), required=True)
    ap.add_argument("--particao", default="validacao")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--gerar", action="store_true")
    g.add_argument("--resumo", action="store_true", help="só o resumo da geração")
    g.add_argument("--predizer", action="store_true")
    ap.add_argument("--limite", type=int, default=None, help="só as primeiras N imagens (teste rápido)")
    a = ap.parse_args(argv)
    if a.gerar:
        return gerar(a.base, a.particao, a.limite)
    if a.resumo:
        return resumo_geracao(a.base, a.particao)
    return predizer_eixo_b(a.base, a.particao)


if __name__ == "__main__":
    raise SystemExit(main())
