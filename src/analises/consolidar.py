"""Consolidação das famílias de testes e T3 (protocolo §8.2, §9; F6 › Eixo B › Lados e tendência; F6 › T3).

Roda no notebook de treino, depois de:
- eixo A (``src.analises.eixo_a``) em ``/out/analises/<partição>/eixo_a_regressoes.csv``;
- eixo B de cada base enviado ao dataset ``resultados`` e baixado (``src.cache.resultados --baixar``).

Faz:
- **família 2 (Holm)**: tendência do ΔAUROC em gama, brilho e contraste em todos os pares modelo × base
  (por degradação, o menor p dos lados × número de lados, já calculado na base);
- **BH** nas secundárias do eixo B: tendências de FNR, FPR e ECE em todas as degradações e a do AUROC nas
  espaciais (não estimáveis entram com p = 1);
- **T3** por par modelo × base, degradação e lado: dose-resposta se (a) pelo menos 2 dos 3 desfechos (FNR,
  FPR, ECE) têm tendência com p ajustado < 0,05 **e** efeito máximo >= margem; (b) o controle negativo
  (p25/p75) não tem efeito acima da margem em AUROC, FNR e FPR; (c) a direção bate com o eixo A em pelo
  menos uma comparação e em todas as feitas. A direção só é comparada onde o eixo A tem efeito (p ajustado
  < 0,05) e o eixo B tem tendência acima da margem; para gama e brilho, só no FP (§8.3). Sem comparação
  possível, não há convergência A ↔ B e o T3 sai "não".

Uso (notebook "treino", CPU):
    python -m src.analises.consolidar --particao validacao
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

from src.analises.eixo_c import margem
from src.analises.estatistica import benjamini_hochberg, holm

FONTE_CURTA = {"chexpert_plus": "plus", "rsna": "rsna", "padchest": "padchest"}
MEDIDA_DA_DEGRADACAO = {"gama": "q_brilho", "brilho": "q_brilho", "contraste": "q_contraste",
                        "desfoque": "q_nitidez", "resolucao": "q_nitidez", "ruido": "q_ruido"}
SINAL_DO_LADO = {"baixo": -1, "alto": 1}                  # o nível sobe → a medida desce (baixo) ou sobe (alto)


def ler_eixo_b(pasta: str, particao: str) -> pd.DataFrame:
    arqs = sorted(glob.glob(os.path.join(pasta, f"*__{particao}__eixo_b_resultados.csv")))
    if not arqs:
        raise FileNotFoundError(f"nenhum eixo B em {pasta} (rode src.cache.resultados --baixar)")
    return pd.concat([pd.read_csv(a) for a in arqs], ignore_index=True)


def familias(b: pd.DataFrame) -> pd.DataFrame:
    b = b.copy()
    prim = b.drop_duplicates(["modelo", "fonte", "degradacao"])
    prim = prim[prim["familia"] == "primaria"]
    ajuste = dict(zip(zip(prim["modelo"], prim["fonte"], prim["degradacao"]),
                      holm(prim["p_familia2"].fillna(1.0).to_numpy())))
    b["p_familia2_holm"] = [ajuste.get((m, f, d), np.nan) for m, f, d in zip(b["modelo"], b["fonte"], b["degradacao"])]
    # secundárias (BH): FNR, FPR, ECE em todas; AUROC nas espaciais
    cols = [("fnr", None), ("fpr", None), ("ece", None), ("auroc", "secundaria")]
    pares = []
    for k, so in cols:
        for i, r in b.iterrows():
            if so is None or r["familia"] == so:
                pares.append((i, k, r[f"p_tendencia_{k}"]))
    aj = benjamini_hochberg(np.array([1.0 if pd.isna(p) else p for _, _, p in pares]))
    for (i, k, _), pa in zip(pares, aj):
        b.loc[i, f"p_{k}_bh"] = pa
    return b


def t3(b: pd.DataFrame, a: pd.DataFrame, prevalencias: dict) -> pd.DataFrame:
    linhas = []
    for _, r in b.iterrows():
        prev = prevalencias.get(r["fonte"], 0.1)
        tend = {}
        for k in ("fnr", "fpr", "ece"):
            p = r.get(f"p_{k}_bh", np.nan)
            tend[k] = bool(pd.notna(p) and p < 0.05 and abs(r[f"delta_{k}_max"]) >= margem(k, prev))
        controle = bool(r["controle_auroc_ok"] and r["controle_fnr_ok"] and r["controle_fpr_ok"])
        medida = MEDIDA_DA_DEGRADACAO[r["degradacao"]]
        # direção (F6 › T3 › Critério, revisto no ensaio): só onde A e B têm efeito; brilho e gama só no FP (§8.3)
        desfechos = [("fpr", "fp")] if medida == "q_brilho" else [("fnr", "fn"), ("fpr", "fp")]
        direcoes = {}
        for k, erro in desfechos:
            ora = a[(a["modelo"] == r["modelo"]) & (a["fonte"] == r["fonte"]) & (a["medida"] == medida) & (a["erro"] == erro)]
            orb = r.get(f"or_nivel_{k}", np.nan)
            a_tem_efeito = len(ora) and pd.notna(ora["or_1dp"].iloc[0]) and ora["p_ajustado"].iloc[0] < 0.05
            if a_tem_efeito and tend[k] and pd.notna(orb):
                efeito_b = np.sign(np.log(orb)) * SINAL_DO_LADO[r["lado"]]      # efeito da medida subir sobre o erro
                direcoes[k] = bool(efeito_b == np.sign(np.log(ora["or_1dp"].iloc[0])))
        mesma_direcao = all(direcoes.values()) if direcoes else None
        n_tend = sum(tend.values())
        resultado = n_tend >= 2 and controle and mesma_direcao is True
        linhas.append({"modelo": r["modelo"], "fonte": r["fonte"], "degradacao": r["degradacao"], "lado": r["lado"],
                       "medida_eixo_a": medida, "tendencias": ",".join(k for k, v in tend.items() if v) or "-",
                       "n_tendencias": n_tend, "controle_negativo_ok": controle,
                       "direcao_comparada": ",".join(f"{k}:{'igual' if v else 'oposta'}" for k, v in direcoes.items())
                       or "sem efeito no eixo A para comparar",
                       "mesma_direcao_eixo_a": mesma_direcao, "t3_dose_resposta": resultado})
    return pd.DataFrame(linhas)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--particao", default="validacao")
    ap.add_argument("--resultados", default="/out/resultados")
    ap.add_argument("--analises", default="/out/analises")
    a = ap.parse_args(argv)
    saida = os.path.join(a.analises, a.particao)
    b = familias(ler_eixo_b(a.resultados, a.particao))
    eixo_a = pd.read_csv(os.path.join(saida, "eixo_a_regressoes.csv"))
    tab = pd.read_csv(os.path.join(saida, "e0_pares.csv"))
    prevalencias = dict(zip(tab["fonte"], tab["prevalencia"]))
    t = t3(b, eixo_a, prevalencias)
    b.to_csv(os.path.join(saida, "eixo_b_consolidado.csv"), index=False)
    t.to_csv(os.path.join(saida, "t3.csv"), index=False)
    prim = b.drop_duplicates(["modelo", "fonte", "degradacao"])
    prim = prim[prim["familia"] == "primaria"]
    print(f"Consolidação | {a.particao} | eixo B de {b['fonte'].nunique()} bases, {len(b)} curvas")
    print(f"Família 2 (Holm, {len(prim)} testes):")
    for _, r in prim.sort_values(["fonte", "modelo", "degradacao"]).iterrows():
        print(f"  {r['modelo']} × {r['fonte']:13s} | {r['degradacao']:9s} | p {r['p_familia2']:.3g} → Holm {r['p_familia2_holm']:.3g}")
    print(f"T3 (dose-resposta; direção comparada só onde A e B têm efeito): {int(t['t3_dose_resposta'].sum())} de {len(t)} curvas")
    for _, r in t.iterrows():
        print(f"  {r['modelo']} × {r['fonte']:13s} | {r['degradacao']:9s} {r['lado']:5s} | tendências {r['tendencias']:12s} | "
              f"controle {'ok' if r['controle_negativo_ok'] else 'FORA'} | direção {r['direcao_comparada']:34s} | "
              f"T3 {'SIM' if r['t3_dose_resposta'] else 'não'}")
    print(f"RESULTADO: OK (salvo em {saida}/eixo_b_consolidado.csv e t3.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
