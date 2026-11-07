"""E0: desempenho de referência e gap bruto (protocolo §8.1 e §9).

Para cada execução (modelo × semente) e cada base: AUROC, AUPRC (com a prevalência), FNR, FPR e PPV
no limiar fixado na validação da fonte (``pos_treino.json``), ECE (15 faixas de igual massa) bruto e
após temperatura, intercepto e inclinação de calibração, Brier.

Por par modelo × base: média e desvio-padrão entre as 5 sementes (dispersão entre sementes) e IC 95%
da média por bootstrap por paciente (as mesmas imagens reamostradas para as 5 sementes), reportados
separadamente (§9). Gap bruto por par fonte → externa, orientado para que positivo signifique piora
no externo (AUROC e AUPRC: interno − externo; FNR, FPR e ECE: externo − interno), com bootstrap
independente em cada base.

No M3, FNR e FPR também com o limiar por fonte (F6 › M3 › Limiar por fonte, sensibilidade).

Uso (notebook "treino", CPU):
    python -m src.analises.e0 --tabela /out/predicoes/tabela_validacao.csv --bootstrap 200    # ensaio
    python -m src.analises.e0 --tabela /out/predicoes/tabela_teste.csv                        # teste (2.000)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from src.analises.metricas import ece, indices_bootstrap, metricas
from src.predicoes.pos_treino import TREINOS
from sklearn.metrics import average_precision_score, roc_auc_score

FONTES_TREINO = {"M1": {"chexpert_plus"}, "M2": {"rsna"}, "M3": {"chexpert_plus", "rsna"}}
FONTE_CURTA = {"chexpert_plus": "plus", "rsna": "rsna", "padchest": "padchest"}
NO_BOOTSTRAP = ["auroc", "auprc", "fnr", "fpr", "ece"]
PIORA_SE_SOBE = {"fnr", "fpr", "ece"}


def papel(modelo: str, fonte: str) -> str:
    return "interno" if fonte in FONTES_TREINO[modelo] else "externo"


def _pos_treino(id_execucao: str, treinos: str) -> dict:
    return json.load(open(os.path.join(treinos, id_execucao, "pos_treino.json"), encoding="utf-8"))


def _rapidas(y, z, limiar):
    p = 1 / (1 + np.exp(-z))
    pred = z >= limiar
    return {"auroc": roc_auc_score(y, z), "auprc": average_precision_score(y, z),
            "fnr": float((~pred[y == 1]).mean()), "fpr": float(pred[y == 0].mean()), "ece": ece(y, p)}


def por_execucao(tab: pd.DataFrame, treinos: str) -> pd.DataFrame:
    linhas = []
    for (ex, fonte), g in tab.groupby(["id_execucao", "fonte"], sort=True):
        pt = _pos_treino(ex, treinos)
        y, z = g["rotulo"].to_numpy(int), g["logit"].to_numpy(float)
        m = metricas(y, z, pt["limiar"]["limiar_logit"], g["probabilidade_temperatura"].to_numpy(float))
        mod = g["modelo"].iloc[0]
        linha = {"id_execucao": ex, "modelo": mod, "semente": int(g["semente"].iloc[0]), "fonte": fonte,
                 "papel": papel(mod, fonte), **m}
        lpf = pt.get("limiar_por_fonte", {}).get(FONTE_CURTA[fonte])
        if lpf:                                            # M3: limiar por fonte (sensibilidade)
            pred = z >= lpf["limiar_logit"]
            linha["fnr_limiar_fonte"] = float((~pred[y == 1]).mean())
            linha["fpr_limiar_fonte"] = float(pred[y == 0].mean())
        linhas.append(linha)
    return pd.DataFrame(linhas)


def _bootstrap_par(tab: pd.DataFrame, modelo: str, fonte: str, limiares: dict, n: int, semente: int):
    """Matriz n × métricas com a média entre sementes, reamostrando pacientes da base."""
    g = tab[(tab["modelo"] == modelo) & (tab["fonte"] == fonte)]
    piv = g.pivot_table(index="id_imagem", columns="id_execucao", values="logit")
    meta = g.drop_duplicates("id_imagem").set_index("id_imagem").loc[piv.index]
    y, pac = meta["rotulo"].to_numpy(int), meta["id_paciente"].to_numpy()
    zs = {ex: piv[ex].to_numpy(float) for ex in piv.columns}
    out = np.empty((n, len(NO_BOOTSTRAP)))
    for b, idx in enumerate(indices_bootstrap(pac, n, semente)):
        if y[idx].min() == y[idx].max():
            out[b] = np.nan
            continue
        vals = [_rapidas(y[idx], z[idx], limiares[ex]) for ex, z in zs.items()]
        out[b] = [np.mean([v[k] for v in vals]) for k in NO_BOOTSTRAP]
    return out


def resumir(tab: pd.DataFrame, por_ex: pd.DataFrame, treinos: str, n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    limiares = {ex: _pos_treino(ex, treinos)["limiar"]["limiar_logit"] for ex in por_ex["id_execucao"].unique()}
    pares, boots = [], {}
    metricas_sem = ["auroc", "auprc", "fnr", "fpr", "ppv", "ece", "ece_temperatura", "intercepto", "inclinacao",
                    "brier", "prob_media", "fnr_limiar_fonte", "fpr_limiar_fonte"]
    for (mod, fonte), g in por_ex.groupby(["modelo", "fonte"]):
        linha = {"modelo": mod, "fonte": fonte, "papel": papel(mod, fonte), "sementes": len(g),
                 "n": int(g["n"].iloc[0]), "prevalencia": float(g["prevalencia"].iloc[0])}
        for k in metricas_sem:
            if k in g and g[k].notna().any():
                linha[k], linha[f"{k}_dp_sementes"] = float(g[k].mean()), float(g[k].std(ddof=1))
        if n_boot:
            bt = _bootstrap_par(tab, mod, fonte, limiares, n_boot, 2026)
            boots[(mod, fonte)] = bt
            for j, k in enumerate(NO_BOOTSTRAP):
                linha[f"{k}_ic_inf"], linha[f"{k}_ic_sup"] = np.nanpercentile(bt[:, j], [2.5, 97.5])
        pares.append(linha)
    pares = pd.DataFrame(pares)
    gaps = []
    for mod in sorted(pares["modelo"].unique()):
        p = pares[pares["modelo"] == mod]
        for _, ext in p[p["papel"] == "externo"].iterrows():
            for _, inn in p[p["papel"] == "interno"].iterrows():
                linha = {"modelo": mod, "interno": inn["fonte"], "externo": ext["fonte"]}
                for j, k in enumerate(NO_BOOTSTRAP):
                    d = ext[k] - inn[k] if k in PIORA_SE_SOBE else inn[k] - ext[k]
                    linha[f"gap_{k}"] = d
                    if n_boot:
                        bi, be = boots[(mod, inn["fonte"])][:, j], boots[(mod, ext["fonte"])][:, j]
                        db = be - bi if k in PIORA_SE_SOBE else bi - be
                        linha[f"gap_{k}_ic_inf"], linha[f"gap_{k}_ic_sup"] = np.nanpercentile(db, [2.5, 97.5])
                gaps.append(linha)
    return pares, pd.DataFrame(gaps)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tabela", required=True)
    ap.add_argument("--treinos", default=TREINOS)
    ap.add_argument("--saida", default=None, help="padrão: /out/analises/<partição>")
    ap.add_argument("--bootstrap", type=int, default=2000)
    a = ap.parse_args(argv)
    tab = pd.read_csv(a.tabela, dtype={"id_imagem": str, "id_paciente": str})
    tab = tab[tab["degradacao"] == "nenhuma"]
    particao = "+".join(sorted(tab["particao"].unique()))
    saida = a.saida or f"/out/analises/{particao}"
    os.makedirs(saida, exist_ok=True)
    por_ex = por_execucao(tab, a.treinos)
    pares, gaps = resumir(tab, por_ex, a.treinos, a.bootstrap)
    por_ex.to_csv(os.path.join(saida, "e0_por_execucao.csv"), index=False)
    pares.to_csv(os.path.join(saida, "e0_pares.csv"), index=False)
    gaps.to_csv(os.path.join(saida, "e0_gaps.csv"), index=False)
    print(f"E0 | partição {particao} | {tab['id_execucao'].nunique()} execuções | bootstrap {a.bootstrap}")
    for _, r in pares.iterrows():
        ic = f" [{r['auroc_ic_inf']:.3f}; {r['auroc_ic_sup']:.3f}]" if a.bootstrap else ""
        extra = (f" | limiar por fonte: FNR {r['fnr_limiar_fonte']:.3f}, FPR {r['fpr_limiar_fonte']:.3f}"
                 if "fnr_limiar_fonte" in r and pd.notna(r.get("fnr_limiar_fonte")) else "")
        print(f"  {r['modelo']} × {r['fonte']:13s} ({r['papel']:7s}) | AUROC {r['auroc']:.3f} ± {r['auroc_dp_sementes']:.3f}{ic}"
              f" | AUPRC {r['auprc']:.3f} (prev {r['prevalencia']:.3f}) | FNR {r['fnr']:.3f} | FPR {r['fpr']:.3f}"
              f" | ECE {r['ece']:.4f} | inclinação {r['inclinacao']:.2f}{extra}")
    for _, r in gaps.iterrows():
        ic = f" [{r['gap_auroc_ic_inf']:.3f}; {r['gap_auroc_ic_sup']:.3f}]" if a.bootstrap else ""
        print(f"  gap {r['modelo']}: {r['interno']} → {r['externo']} | ΔAUROC {r['gap_auroc']:+.3f}{ic} | "
              f"ΔFNR {r['gap_fnr']:+.3f} | ΔFPR {r['gap_fpr']:+.3f} | ΔECE {r['gap_ece']:+.4f}")
    print(f"RESULTADO: OK (salvo em {saida}/e0_*.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
