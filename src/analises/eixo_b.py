"""Análise do eixo B (protocolo §8.4 e §9; F6 › Eixo B › Lados).

Lê a tabela de predições do eixo B (``src.perturbations.eixo_b --predizer``) no notebook da base.

Por modelo, degradação e lado (gama e brilho: escurecer e clarear; as outras: um lado), em cada nível
(0 = natural, 1 = p25/p75, 2 = p10/p90, 3 = p5/p95, 4 = p1/p99):
- AUROC, FNR e FPR (no limiar fixado na validação da fonte) e ECE com pesos que devolvem a prevalência
  natural da partição; média entre as 5 sementes; Δ em relação ao natural;
- tendência: FNR e FPR por regressão logística do erro de cada imagem sobre o nível, sementes juntas,
  erro-padrão agrupado por paciente; AUROC e ECE pela inclinação da curva (mínimos quadrados sobre os
  níveis), com bootstrap pareado por paciente (p bilateral); Mann-Kendall só descritivo;
- FNR e FPR com menos de 10 eventos por parâmetro ou ajuste instável: tendência não estimável, sem p
  (mesma regra do eixo A);
- controle negativo (p25/p75): |Δ| abaixo da margem da métrica;
- família 2 (Holm, na consolidação): tendência do ΔAUROC em gama, brilho e contraste; para gama e brilho,
  o menor p dos dois lados × 2 (Bonferroni dentro da degradação), mantendo 27 testes.

Uso (notebook da base, CPU):
    python -m src.analises.eixo_b --base plus --bootstrap 200                       # ensaio
    python -m src.analises.eixo_b --base padchest --particao teste                  # F8 (2.000 reamostragens)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score

from src.analises.eixo_a import EP_MAX, EPV_MIN
from src.analises.eixo_c import ece_ponderado, margem
from src.analises.estatistica import logistica_agrupada, p_normal
from src.analises.metricas import indices_bootstrap

ORDEM = {"baixo": ["p25", "p10", "p5", "p1"], "alto": ["p75", "p90", "p95", "p99"]}
LADOS = {"gama": ["baixo", "alto"], "brilho": ["baixo", "alto"], "contraste": ["baixo"],
         "desfoque": ["baixo"], "resolucao": ["baixo"], "ruido": ["alto"]}
FAMILIA_2 = ["gama", "brilho", "contraste"]
SEMENTE = 2026


def _pesos_ece(y: np.ndarray, prevalencia: float) -> np.ndarray:
    """Pesos que devolvem a prevalência natural na amostra enriquecida (positivos com peso 1)."""
    pos, neg = y.sum(), len(y) - y.sum()
    w_neg = pos * (1 - prevalencia) / (prevalencia * neg)
    return np.where(y == 1, 1.0, w_neg)


def _metricas(y, z, limiar, w_ece):
    pred = z >= limiar
    return {"auroc": roc_auc_score(y, z), "fnr": float((~pred[y == 1]).mean()), "fpr": float(pred[y == 0].mean()),
            "ece": ece_ponderado(y.astype(float), 1 / (1 + np.exp(-z)), w_ece)}


def curva(g: pd.DataFrame, niveis: list[str], limiares: dict, prevalencia: float, n_boot: int) -> dict:
    """g: linhas de um modelo e uma degradação (natural incluído); niveis: ['0', 'p25', ...] em ordem."""
    piv = g.pivot_table(index="id_imagem", columns=["nivel", "id_execucao"], values="logit")
    meta = g.drop_duplicates("id_imagem").set_index("id_imagem").loc[piv.index]
    y, pac = meta["rotulo"].to_numpy(int), meta["id_paciente"].to_numpy()
    w = _pesos_ece(y, prevalencia)
    exs = sorted(g["id_execucao"].unique())
    idx_nivel = np.arange(len(niveis), dtype=float)

    def pontos(ii):
        out = np.empty((len(niveis), 4))
        for a, nv in enumerate(niveis):
            vals = [_metricas(y[ii], piv[(nv, ex)].to_numpy(float)[ii], limiares[ex], w[ii]) for ex in exs]
            out[a] = [np.mean([v[k] for v in vals]) for k in ("auroc", "fnr", "fpr", "ece")]
        return out
    tudo = np.arange(len(y))
    pt = pontos(tudo)
    inclin = lambda col: np.polyfit(idx_nivel, col, 1)[0]                       # noqa: E731
    res = {"niveis": niveis, "pontos": pt, "inclinacao_auroc": inclin(pt[:, 0]), "inclinacao_ece": inclin(pt[:, 3])}
    if n_boot:
        bs = []
        for ii in indices_bootstrap(pac, n_boot, SEMENTE):
            if y[ii].min() == y[ii].max():
                continue
            pb = pontos(ii)
            bs.append((inclin(pb[:, 0]), inclin(pb[:, 3])))
        bs = np.array(bs)
        for j, k in enumerate(("auroc", "ece")):
            b = bs[:, j]
            res[f"p_tendencia_{k}"] = float(min(1.0, 2 * min((1 + np.sum(b <= 0)) / (len(b) + 1),
                                                              (1 + np.sum(b >= 0)) / (len(b) + 1))))
            res[f"inclinacao_{k}_ic"] = np.percentile(b, [2.5, 97.5]).tolist()
    # FNR e FPR: regressão logística do erro sobre o nível, sementes juntas, EP agrupado por paciente
    for k, classe in (("fnr", 1), ("fpr", 0)):
        sel = y == classe
        xs, ys, gs = [], [], []
        for a, nv in enumerate(niveis):
            for ex in exs:
                z = piv[(nv, ex)].to_numpy(float)[sel]
                erro = (z < limiares[ex]) if classe == 1 else (z >= limiares[ex])
                xs.append(np.full(sel.sum(), a, float)); ys.append(erro.astype(float)); gs.append(pac[sel])
        x = np.column_stack([np.ones(sum(len(v) for v in xs)), np.concatenate(xs)])
        yy = np.concatenate(ys)
        eventos = int(min(yy.sum(), len(yy) - yy.sum()))
        res[f"eventos_{k}"] = eventos
        # mesma regra do eixo A (F6 › Eixo A › Regressões não estimáveis)
        if eventos < EPV_MIN * x.shape[1]:
            res[f"or_nivel_{k}"], res[f"p_tendencia_{k}"] = np.nan, np.nan
            res[f"nao_estimavel_{k}"] = f"poucos eventos ({eventos})"
            continue
        b, ep = logistica_agrupada(x, yy, np.concatenate(gs))
        if not np.isfinite(ep[1]) or ep[1] > EP_MAX or abs(b[1]) > 10:
            res[f"or_nivel_{k}"], res[f"p_tendencia_{k}"] = np.nan, np.nan
            res[f"nao_estimavel_{k}"] = f"ajuste instável (coef {b[1]:.2f}, EP {ep[1]:.2f})"
            continue
        res[f"or_nivel_{k}"] = float(np.exp(b[1]))
        res[f"p_tendencia_{k}"] = p_normal(b[1], ep[1])
    for j, k in enumerate(("auroc", "fnr", "fpr", "ece")):
        res[f"mann_kendall_{k}"] = float(kendalltau(idx_nivel, pt[:, j]).pvalue) if len(niveis) > 2 else np.nan
    return res


def rodar(tab: pd.DataFrame, modelos: str, prevalencia: float, n_boot: int) -> pd.DataFrame:
    limiares = {ex: json.load(open(os.path.join(modelos, ex, "pos_treino.json")))["limiar"]["limiar_logit"]
                for ex in tab["id_execucao"].unique()}
    linhas = []
    for (mod, fonte), g in tab.groupby(["modelo", "fonte"]):
        nat = g[g["degradacao"] == "nenhuma"]
        for deg, lados in LADOS.items():
            gd = g[g["degradacao"] == deg]
            if gd.empty:
                continue
            for lado in lados:
                niveis = ["0"] + [n for n in ORDEM[lado] if (gd["nivel"] == n).any()]
                if len(niveis) < 3:
                    continue
                c = curva(pd.concat([nat, gd[gd["nivel"].isin(niveis)]]), niveis, limiares, prevalencia, n_boot)
                base = c["pontos"][0]
                linha = {"modelo": mod, "fonte": fonte, "degradacao": deg, "lado": lado,
                         "niveis": ",".join(niveis[1:]), "familia": "primaria" if deg in FAMILIA_2 else "secundaria"}
                for j, k in enumerate(("auroc", "fnr", "fpr", "ece")):
                    linha[f"{k}_natural"] = base[j]
                    linha[f"delta_{k}_max"] = c["pontos"][-1, j] - base[j]
                    d1 = c["pontos"][1, j] - base[j]                      # controle negativo (p25/p75)
                    linha[f"delta_{k}_controle"] = d1
                    linha[f"controle_{k}_ok"] = abs(d1) < margem(k, prevalencia)
                for k in ("auroc", "fnr", "fpr", "ece"):
                    for chave in (f"p_tendencia_{k}", f"mann_kendall_{k}"):
                        linha[chave] = c.get(chave, np.nan)
                linha.update({"inclinacao_auroc": c["inclinacao_auroc"], "inclinacao_ece": c["inclinacao_ece"],
                              "or_nivel_fnr": c["or_nivel_fnr"], "or_nivel_fpr": c["or_nivel_fpr"],
                              "eventos_fnr": c["eventos_fnr"], "eventos_fpr": c["eventos_fpr"],
                              "nao_estimavel": "; ".join(f"{k}: {c[f'nao_estimavel_{k}']}" for k in ("fnr", "fpr")
                                                         if f"nao_estimavel_{k}" in c)})
                linha["curva"] = json.dumps({n: [round(float(v), 5) for v in c["pontos"][a]] for a, n in enumerate(niveis)})
                linhas.append(linha)
    res = pd.DataFrame(linhas)
    # p do teste primário por degradação (família 2): o menor dos lados × número de lados
    res["p_familia2"] = np.nan
    for (mod, fonte, deg), g in res[res["familia"] == "primaria"].groupby(["modelo", "fonte", "degradacao"]):
        p = min(1.0, g["p_tendencia_auroc"].min() * len(g))
        res.loc[g.index, "p_familia2"] = p
    return res


def prevalencia(base: str, particao: str, listas_json: str = "data/splits/listas_sha256.json") -> float:
    """Prevalência natural da partição (regra primária: incertos fora). Validação: a lista; teste: as
    contagens congeladas (o carregador de treino não lê o teste)."""
    if particao == "validacao":
        from src.treino.carregador import ler_particao
        return float(ler_particao(base, particao, verificar_hashes=False)["rotulo"].mean())
    d = json.load(open(listas_json, encoding="utf-8"))["bases"][base]["listas"][particao]
    return d["positivos"] / (d["imagens"] - d.get("incertos", 0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", choices=["plus", "rsna", "padchest"], required=True)
    ap.add_argument("--particao", default="validacao")
    ap.add_argument("--modelos", default="/out/modelos")
    ap.add_argument("--bootstrap", type=int, default=2000)
    a = ap.parse_args(argv)
    pasta = f"/out/{a.base}/eixo_b/{a.particao}"
    tab = pd.read_csv(os.path.join(pasta, "tabela_eixo_b.csv"), dtype={"id_imagem": str, "id_paciente": str, "nivel": str})
    prev = prevalencia(a.base, a.particao)
    res = rodar(tab, a.modelos, prev, a.bootstrap)
    destino = os.path.join(pasta, "eixo_b_resultados.csv")
    res.to_csv(destino, index=False)
    print(f"Eixo B | {a.base} | {a.particao} | prevalência natural {prev:.4f} | {len(res)} curvas | bootstrap {a.bootstrap}")
    for _, r in res.iterrows():
        ok = "ok" if r["controle_auroc_ok"] and r["controle_fnr_ok"] and r["controle_fpr_ok"] else "FORA"
        print(f"  {r['modelo']} {r['degradacao']:9s} {r['lado']:5s} | ΔAUROC máx {r['delta_auroc_max']:+.3f} "
              f"(p {r['p_tendencia_auroc']:.3g}) | ΔFNR {r['delta_fnr_max']:+.3f} (p {r['p_tendencia_fnr']:.3g}) | "
              f"ΔFPR {r['delta_fpr_max']:+.3f} (p {r['p_tendencia_fpr']:.3g}) | ΔECE {r['delta_ece_max']:+.4f} | controle {ok}"
              + (f" | não estimável: {r['nao_estimavel']}" if r["nao_estimavel"] else ""))
    print(f"RESULTADO: OK (salvo em {destino})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
