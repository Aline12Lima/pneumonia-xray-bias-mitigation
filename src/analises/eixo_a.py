"""Eixo A: qualidade da imagem × erros e calibração, dentro de cada base (protocolo §8.3 e §9).

Para cada par modelo × base e cada medida de qualidade:
- FN (entre positivos) e FP (entre negativos) no limiar fixado na validação da fonte (principal);
- regressão logística do erro sobre a medida padronizada em z **dentro da fonte**, ajustada por
  posição, idade, sexo e fabricante (quando a base tem mais de um), com as **5 sementes juntas** na
  mesma regressão e erro-padrão agrupado por paciente; efeito em razão de chances por 1 DP;
- família primária (§9, família 1): brilho, contraste, saturação e enquadramento × FN e FP × pares,
  com Holm (α = 0,05); medidas secundárias com Benjamini-Hochberg;
- brilho: lido principalmente no modelo de FP; no de FN, marcado com a ressalva de que pode refletir
  a extensão da opacidade (§8.3);
- ECE e inclinação de calibração por faixa da medida (Plus: 2 faixas; RSNA e PadChest: 3), média
  entre sementes.

Uso (notebook "treino", CPU):
    python -m src.analises.eixo_a --tabela /out/predicoes/tabela_validacao.csv
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from src.analises.estatistica import benjamini_hochberg, holm, logistica_agrupada, p_normal
from src.analises.metricas import calibracao, ece
from src.predicoes.pos_treino import TREINOS
from src.qualidade.medidas import PRIMARIAS, SECUNDARIAS

FAIXAS = {"chexpert_plus": 2, "rsna": 3, "padchest": 3}
MIN_FABRICANTE = 50                                       # fabricantes com menos imagens viram "outros"


def _limiares(execucoes, treinos: str) -> dict:
    return {ex: json.load(open(os.path.join(treinos, ex, "pos_treino.json"), encoding="utf-8"))["limiar"]["limiar_logit"]
            for ex in execucoes}


def _covariaveis(g: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=g.index)
    x["pa"] = (g["posicao"] == "PA").astype(float)
    x["masculino"] = (g["sexo"] == "M").astype(float)
    idade = g["idade"].astype(float)
    x["idade_z"] = (idade - idade.mean()) / idade.std(ddof=0)
    fab = g["fabricante"].fillna("sem")
    cont = fab.value_counts()
    fab = fab.where(fab.map(cont) >= MIN_FABRICANTE, "outros")
    if fab.nunique() > 1:
        x = x.join(pd.get_dummies(fab, prefix="fab", drop_first=True, dtype=float))
    return x.loc[:, x.std(ddof=0) > 0]


def regressao(g: pd.DataFrame, medida: str, erro: str) -> dict:
    """g: linhas de um par (5 sementes) com a coluna ``medida_z`` e a coluna binária ``erro``."""
    d = g.dropna(subset=[f"{medida}_z", "idade"])
    x = _covariaveis(d)
    x.insert(0, "medida_z", d[f"{medida}_z"].astype(float))
    x.insert(0, "const", 1.0)
    coef, eps = logistica_agrupada(x.to_numpy(float), d[erro].to_numpy(float), d["id_paciente"].to_numpy())
    b, ep = float(coef[1]), float(eps[1])
    if not np.isfinite(ep):
        raise RuntimeError("erro-padrão não finito (separação ou colinearidade)")
    return {"n_linhas": int(len(d)), "eventos": int(d[erro].sum()), "or_1dp": float(np.exp(b)),
            "ic_inf": float(np.exp(b - 1.96 * ep)), "ic_sup": float(np.exp(b + 1.96 * ep)),
            "p": p_normal(b, ep), "covariaveis": ",".join(c for c in x.columns if c not in ("const", "medida_z"))}


def calibracao_por_faixa(g: pd.DataFrame, medida: str, faixas: int) -> list[dict]:
    imgs = g.drop_duplicates("id_imagem").dropna(subset=[medida])
    cortes = np.quantile(imgs[medida], np.linspace(0, 1, faixas + 1))
    rotulos = pd.cut(g[medida], bins=np.unique(cortes), include_lowest=True, labels=False)
    out = []
    for f in range(faixas):
        sub = g[rotulos == f]
        if sub.empty:
            continue
        por_semente = []
        for _, s in sub.groupby("id_execucao"):
            y, z = s["rotulo"].to_numpy(float), s["logit"].to_numpy(float)
            if 0 < y.sum() < len(y):
                por_semente.append((ece(y, 1 / (1 + np.exp(-z))), calibracao(y, z)[1]))
        if por_semente:
            e, i = np.array(por_semente).T
            out.append({"faixa": f + 1, "de": float(cortes[f]), "ate": float(cortes[f + 1]),
                        "imagens": int(sub["id_imagem"].nunique()), "positivos": int(sub.drop_duplicates("id_imagem")["rotulo"].sum()),
                        "ece": float(e.mean()), "ece_dp_sementes": float(e.std(ddof=1)) if len(e) > 1 else np.nan,
                        "inclinacao": float(i.mean())})
    return out


def rodar(tab: pd.DataFrame, treinos: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    lim = _limiares(tab["id_execucao"].unique(), treinos)
    tab = tab.copy()
    tab["pred"] = tab["logit"].to_numpy() >= tab["id_execucao"].map(lim).to_numpy()
    tab["fn"] = ((tab["rotulo"] == 1) & ~tab["pred"]).astype(int)
    tab["fp"] = ((tab["rotulo"] == 0) & tab["pred"]).astype(int)
    medidas = PRIMARIAS + SECUNDARIAS
    for fonte, g in tab.groupby("fonte"):                 # z por fonte, sobre as imagens da base
        imgs = g.drop_duplicates("id_imagem")
        for m in medidas:
            mu, dp = imgs[m].mean(), imgs[m].std(ddof=0)
            tab.loc[g.index, f"{m}_z"] = (g[m] - mu) / dp
    regs, cal = [], []
    for (mod, fonte), g in tab.groupby(["modelo", "fonte"]):
        for m in medidas:
            for erro, sub in (("fn", g[g["rotulo"] == 1]), ("fp", g[g["rotulo"] == 0])):
                linha = {"modelo": mod, "fonte": fonte, "medida": m, "erro": erro,
                         "familia": "primaria" if m in PRIMARIAS else "secundaria",
                         "ressalva": "brilho no FN pode refletir a opacidade" if (m == "q_brilho" and erro == "fn") else ""}
                try:
                    linha.update(regressao(sub, m, erro))
                except Exception as e:                    # separação, colinearidade: registra e segue
                    linha["falha"] = f"{type(e).__name__}: {e}"
                regs.append(linha)
        for m in PRIMARIAS:
            for c in calibracao_por_faixa(g, m, FAIXAS.get(fonte, 3)):
                cal.append({"modelo": mod, "fonte": fonte, "medida": m, **c})
    regs = pd.DataFrame(regs)
    regs["p_ajustado"] = np.nan
    for fam, metodo in (("primaria", holm), ("secundaria", benjamini_hochberg)):
        sel = (regs["familia"] == fam) & regs["p"].notna()
        if sel.any():
            regs.loc[sel, "p_ajustado"] = metodo(regs.loc[sel, "p"].to_numpy())
    regs["significativo"] = regs["p_ajustado"] < 0.05
    return regs, pd.DataFrame(cal)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tabela", required=True)
    ap.add_argument("--treinos", default=TREINOS)
    ap.add_argument("--saida", default=None)
    a = ap.parse_args(argv)
    tab = pd.read_csv(a.tabela, dtype={"id_imagem": str, "id_paciente": str, "fabricante": "object"})
    tab = tab[tab["degradacao"] == "nenhuma"]
    particao = "+".join(sorted(tab["particao"].unique()))
    saida = a.saida or f"/out/analises/{particao}"
    os.makedirs(saida, exist_ok=True)
    regs, cal = rodar(tab, a.treinos)
    regs.to_csv(os.path.join(saida, "eixo_a_regressoes.csv"), index=False)
    cal.to_csv(os.path.join(saida, "eixo_a_calibracao_faixas.csv"), index=False)
    prim = regs[regs["familia"] == "primaria"]
    falhas = int(regs["falha"].notna().sum()) if "falha" in regs else 0
    print(f"Eixo A | partição {particao} | {len(regs)} regressões ({len(prim)} primárias, Holm; "
          f"{len(regs) - len(prim)} secundárias, BH) | falhas: {falhas}")
    for _, r in prim.iterrows():
        if pd.notna(r.get("or_1dp")):
            print(f"  {r['modelo']} × {r['fonte']:13s} | {r['medida']:16s} | {r['erro'].upper()} | OR {r['or_1dp']:.2f} "
                  f"[{r['ic_inf']:.2f}; {r['ic_sup']:.2f}] | eventos {r['eventos']:5d} | p aj. {r['p_ajustado']:.3g}"
                  + (f" | {r['ressalva']}" if r["ressalva"] else ""))
        else:
            print(f"  {r['modelo']} × {r['fonte']} | {r['medida']} | {r['erro']} | FALHA: {r.get('falha')}")
    print(f"Calibração por faixa: {len(cal)} linhas")
    print(f"RESULTADO: {'OK' if not falhas else 'OK com falhas registradas'} (salvo em {saida}/eixo_a_*.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
