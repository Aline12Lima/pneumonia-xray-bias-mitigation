"""Eixo C: explicação da queda externa por padronização (protocolo §8.5; decisões F6 › Eixo C).

Para cada par fonte → externa (no ensaio: M1 Plus → RSNA e M2 RSNA → Plus; no teste, também o PadChest):
- D_bruto = R_interno − R_externo (AUROC; FNR, FPR e ECE orientados para que positivo = piora);
- S1: o externo reponderado para a distribuição clínica (posição, idade, sexo) da fonte;
- S2: S1 mais as medidas de qualidade comparáveis entre bases;
- pesos de importância por regressão logística (classificador de domínio fonte × externo), **dentro de
  cada classe** (positivos e negativos separados) em S1 e S2, para que a diferença entre S1 e S2 seja
  só a qualidade; pesos **truncados no percentil 99** e renormalizados para média 1 em cada classe;
- parcela clínica = (D_bruto − D_S1) / D_bruto; parcela de qualidade = (D_S1 − D_S2) / D_bruto, só
  quando |D_bruto| >= margem da métrica e o IC de D_bruto exclui zero; sem truncar as frações;
- teste da família 3 (H2): D_S1 − D_S2 > 0 na AUROC, p unilateral por bootstrap; Holm nos pares;
  as outras métricas com Benjamini-Hochberg;
- diagnóstico: tamanho efetivo da amostra, maior diferença média padronizada depois dos pesos (meta
  < 0,1), maior peso e fração de pesos truncados;
- bootstrap por paciente em cada base, **reestimando os pesos** em cada reamostragem;
- métricas por semente com os mesmos pesos (os pesos dependem só das imagens) e média entre sementes.

M3 (fonte = Plus + RSNA): R_interno = média do Plus e do RSNA; a distribuição de referência é a união
das duas bases com peso igual por fonte.

Uso (notebook "treino", CPU):
    python -m src.analises.eixo_c --tabela /out/predicoes/tabela_validacao.csv --bootstrap 200
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.analises.estatistica import benjamini_hochberg, holm
from src.analises.metricas import indices_bootstrap
from src.analises.t1_t2 import COMPARAVEIS
from src.predicoes.pos_treino import TREINOS

CLINICAS = ["pa", "idade", "masculino"]
FONTES_TREINO = {"M1": ["chexpert_plus"], "M2": ["rsna"], "M3": ["chexpert_plus", "rsna"]}
METRICAS = ["auroc", "fnr", "fpr", "ece"]
PIORA_SE_SOBE = {"fnr", "fpr", "ece"}
TRUNCAR_PERCENTIL = 99
SEMENTE = 2026


def margem(metrica: str, prevalencia: float) -> float:
    """Margens de relevância do §9."""
    return {"auroc": 0.02, "fnr": 0.03, "fpr": 0.03}.get(metrica, min(0.02, prevalencia / 4))


# ------------------------------------------------------------------ pesos e métricas ponderadas
def pesos(ref: pd.DataFrame, ext: pd.DataFrame, cols: list[str], w_ref: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Pesos de importância para as imagens de ``ext`` imitarem ``ref``, dentro de cada classe."""
    w = np.ones(len(ext))
    diag = {"truncados": 0}
    w_ref = np.ones(len(ref)) if w_ref is None else w_ref
    for classe in (0, 1):
        r, e = ref["rotulo"].to_numpy() == classe, ext["rotulo"].to_numpy() == classe
        x = np.vstack([ref.loc[r, cols].to_numpy(float), ext.loc[e, cols].to_numpy(float)])
        y = np.r_[np.ones(r.sum()), np.zeros(e.sum())]
        sw = np.r_[w_ref[r] / w_ref[r].mean(), np.ones(e.sum())]
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1e6))
        clf.fit(x, y, logisticregression__sample_weight=sw)
        pr = np.clip(clf.predict_proba(ext.loc[e, cols].to_numpy(float))[:, 1], 1e-6, 1 - 1e-6)
        wc = pr / (1 - pr)
        teto = np.percentile(wc, TRUNCAR_PERCENTIL)
        diag["truncados"] += int((wc > teto).sum())
        wc = np.minimum(wc, teto)
        w[e] = wc / wc.mean()
    diag.update({"tamanho_efetivo": float(w.sum() ** 2 / (w ** 2).sum()), "peso_maximo": float(w.max()),
                 "fracao_truncada": diag["truncados"] / len(w)})
    return w, diag


def smd_maxima(ref: pd.DataFrame, ext: pd.DataFrame, cols: list[str], w: np.ndarray, w_ref=None) -> float:
    w_ref = np.ones(len(ref)) if w_ref is None else w_ref
    pior = 0.0
    for c in cols:
        a, b = ref[c].to_numpy(float), ext[c].to_numpy(float)
        ma, mb = np.average(a, weights=w_ref), np.average(b, weights=w)
        va, vb = np.average((a - ma) ** 2, weights=w_ref), np.average((b - mb) ** 2, weights=w)
        pior = max(pior, abs(ma - mb) / np.sqrt((va + vb) / 2 + 1e-12))
    return float(pior)


def ece_ponderado(y, p, w, bins: int = 15) -> float:
    ordem = np.argsort(p, kind="stable")
    cw = np.cumsum(w[ordem]) / w.sum()
    faixa = np.minimum((cw * bins - 1e-12).astype(int), bins - 1)
    erro = 0.0
    for b in range(bins):
        g = ordem[faixa == b]
        if len(g):
            wg = w[g]
            erro += wg.sum() / w.sum() * abs(np.average(y[g], weights=wg) - np.average(p[g], weights=wg))
    return float(erro)


def metricas_ponderadas(y, z, limiar, w) -> dict:
    p = 1 / (1 + np.exp(-z))
    pred = z >= limiar
    return {"auroc": float(roc_auc_score(y, z, sample_weight=w)),
            "fnr": float(np.sum(w * ((y == 1) & ~pred)) / np.sum(w * (y == 1))),
            "fpr": float(np.sum(w * ((y == 0) & pred)) / np.sum(w * (y == 0))),
            "ece": ece_ponderado(y, p, w)}


# ------------------------------------------------------------------ um par
def _imagens(tab: pd.DataFrame, modelo: str, fonte: str) -> tuple[pd.DataFrame, dict]:
    g = tab[(tab["modelo"] == modelo) & (tab["fonte"] == fonte)]
    piv = g.pivot_table(index="id_imagem", columns="id_execucao", values="logit")
    img = g.drop_duplicates("id_imagem").set_index("id_imagem").loc[piv.index].reset_index()
    img = img.assign(pa=(img["posicao"] == "PA").astype(float), masculino=(img["sexo"] == "M").astype(float),
                     idade=img["idade"].astype(float))
    ok = img[CLINICAS + COMPARAVEIS].notna().all(axis=1).to_numpy()
    return img[ok].reset_index(drop=True), {ex: piv[ex].to_numpy(float)[ok] for ex in piv.columns}


def _media_sementes(img, zs, limiares, w) -> dict:
    y = img["rotulo"].to_numpy(int)
    vals = [metricas_ponderadas(y, z, limiares[ex], w) for ex, z in zs.items()]
    return {k: float(np.mean([v[k] for v in vals])) for k in METRICAS}


def _calcular(refs, ext, limiares):
    """refs: lista de (imagens, logits) das fontes internas; ext: (imagens, logits) do externo."""
    r_int = {k: float(np.mean([_media_sementes(i, z, limiares, np.ones(len(i)))[k] for i, z in refs])) for k in METRICAS}
    ref = pd.concat([i for i, _ in refs], ignore_index=True)
    w_ref = np.concatenate([np.full(len(i), 1 / len(i)) for i, _ in refs]) * len(ref)   # peso igual por fonte (M3)
    ie, ze = ext
    r_ext = _media_sementes(ie, ze, limiares, np.ones(len(ie)))
    w1, d1 = pesos(ref, ie, CLINICAS, w_ref)
    w2, d2 = pesos(ref, ie, CLINICAS + COMPARAVEIS, w_ref)
    r_s1, r_s2 = _media_sementes(ie, ze, limiares, w1), _media_sementes(ie, ze, limiares, w2)
    out = {}
    for k in METRICAS:
        o = (lambda a, b: b - a) if k in PIORA_SE_SOBE else (lambda a, b: a - b)
        out[k] = {"r_interno": r_int[k], "r_externo": r_ext[k], "r_s1": r_s1[k], "r_s2": r_s2[k],
                  "d_bruto": o(r_int[k], r_ext[k]), "d_s1": o(r_int[k], r_s1[k]), "d_s2": o(r_int[k], r_s2[k])}
    diag = {"s1_" + a: b for a, b in d1.items()} | {"s2_" + a: b for a, b in d2.items()}
    diag["s1_smd_max"] = smd_maxima(ref, ie, CLINICAS, w1, w_ref)
    diag["s2_smd_max"] = smd_maxima(ref, ie, CLINICAS + COMPARAVEIS, w2, w_ref)
    diag["smd_max_antes"] = smd_maxima(ref, ie, CLINICAS + COMPARAVEIS, np.ones(len(ie)), w_ref)
    return out, diag


def _reamostrar(img, zs, idx):
    return img.iloc[idx].reset_index(drop=True), {ex: z[idx] for ex, z in zs.items()}


def par(tab, modelo, externo, limiares, n_boot):
    refs = [_imagens(tab, modelo, f) for f in FONTES_TREINO[modelo]]
    ext = _imagens(tab, modelo, externo)
    est, diag = _calcular(refs, ext, limiares)
    boots = []
    rng_idx = [indices_bootstrap(i["id_paciente"].to_numpy(), n_boot, SEMENTE + k) for k, (i, _) in enumerate(refs)]
    ext_idx = indices_bootstrap(ext[0]["id_paciente"].to_numpy(), n_boot, SEMENTE + 99)
    for _ in range(n_boot):
        rb = [_reamostrar(i, z, next(it)) for (i, z), it in zip(refs, rng_idx)]
        eb = _reamostrar(*ext, next(ext_idx))
        try:
            boots.append(_calcular(rb, eb, limiares)[0])
        except ValueError:                                  # reamostra sem uma das classes
            continue
    linhas = []
    for k in METRICAS:
        e = est[k]
        b = {c: np.array([x[k][c] for x in boots]) for c in ("d_bruto", "d_s1", "d_s2")} if boots else {}
        prev = float(ext[0]["rotulo"].mean())
        linha = {"modelo": modelo, "interno": "+".join(FONTES_TREINO[modelo]), "externo": externo, "metrica": k,
                 **e, "d_clinica": e["d_bruto"] - e["d_s1"], "d_qualidade": e["d_s1"] - e["d_s2"],
                 "margem": margem(k, prev), "bootstrap_validos": len(boots), **diag}
        if boots:
            for c in ("d_bruto", "d_s1", "d_s2"):
                linha[f"{c}_ic_inf"], linha[f"{c}_ic_sup"] = np.percentile(b[c], [2.5, 97.5])
            dq = b["d_s1"] - b["d_s2"]
            linha["d_qualidade_ic_inf"], linha["d_qualidade_ic_sup"] = np.percentile(dq, [2.5, 97.5])
            linha["p_qualidade"] = float((1 + np.sum(dq <= 0)) / (len(dq) + 1))      # H2: D_S1 − D_S2 > 0
            ic_exclui_zero = linha["d_bruto_ic_inf"] > 0 or linha["d_bruto_ic_sup"] < 0
        else:
            ic_exclui_zero = False
        if abs(e["d_bruto"]) >= linha["margem"] and ic_exclui_zero:
            linha["parcela_clinica"] = linha["d_clinica"] / e["d_bruto"]
            linha["parcela_qualidade"] = linha["d_qualidade"] / e["d_bruto"]
        else:
            linha["parcela_clinica"] = linha["parcela_qualidade"] = np.nan
            linha["sem_fracao"] = "D_bruto abaixo da margem ou IC com zero"
        linhas.append(linha)
    return linhas


def rodar(tab: pd.DataFrame, treinos: str, n_boot: int) -> pd.DataFrame:
    limiares = {ex: json.load(open(os.path.join(treinos, ex, "pos_treino.json"), encoding="utf-8"))["limiar"]["limiar_logit"]
                for ex in tab["id_execucao"].unique()}
    linhas = []
    for modelo, fontes in FONTES_TREINO.items():
        if modelo not in set(tab["modelo"]):
            continue
        for ext in sorted(set(tab["fonte"]) - set(fontes)):
            linhas += par(tab, modelo, ext, limiares, n_boot)
    res = pd.DataFrame(linhas)
    if "p_qualidade" in res:
        res["p_ajustado"] = np.nan
        prim = res["metrica"] == "auroc"                   # família 3 (Holm); outras métricas: BH
        res.loc[prim, "p_ajustado"] = holm(res.loc[prim, "p_qualidade"].fillna(1).to_numpy())
        if (~prim).any():
            res.loc[~prim, "p_ajustado"] = benjamini_hochberg(res.loc[~prim, "p_qualidade"].fillna(1).to_numpy())
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tabela", required=True)
    ap.add_argument("--treinos", default=TREINOS)
    ap.add_argument("--saida", default=None)
    ap.add_argument("--bootstrap", type=int, default=2000)
    a = ap.parse_args(argv)
    tab = pd.read_csv(a.tabela, dtype={"id_imagem": str, "id_paciente": str, "fabricante": "object"})
    tab = tab[tab["degradacao"] == "nenhuma"]
    particao = "+".join(sorted(tab["particao"].unique()))
    saida = a.saida or f"/out/analises/{particao}"
    os.makedirs(saida, exist_ok=True)
    res = rodar(tab, a.treinos, a.bootstrap)
    res.to_csv(os.path.join(saida, "eixo_c.csv"), index=False)
    print(f"Eixo C | partição {particao} | {res[['modelo', 'externo']].drop_duplicates().shape[0]} pares | bootstrap {a.bootstrap}")
    for _, r in res.iterrows():
        ic = (f" [{r['d_bruto_ic_inf']:+.3f}; {r['d_bruto_ic_sup']:+.3f}]" if "d_bruto_ic_inf" in r and pd.notna(r["d_bruto_ic_inf"]) else "")
        fr = (f" | parcela clínica {r['parcela_clinica']:+.2f}, qualidade {r['parcela_qualidade']:+.2f}"
              if pd.notna(r["parcela_qualidade"]) else " | sem fração")
        pq = f" | p (qualidade) {r['p_qualidade']:.3g}, aj. {r['p_ajustado']:.3g}" if "p_qualidade" in r and pd.notna(r.get("p_qualidade")) else ""
        print(f"  {r['modelo']} {r['interno']} → {r['externo']} | {r['metrica']:5s} | D_bruto {r['d_bruto']:+.3f}{ic} | "
              f"D_S1 {r['d_s1']:+.3f} | D_S2 {r['d_s2']:+.3f}{fr}{pq}")
    d = res.drop_duplicates(["modelo", "externo"])
    for _, r in d.iterrows():
        print(f"  diagnóstico {r['modelo']} → {r['externo']} | SMD máx. antes {r['smd_max_antes']:.2f}, S1 {r['s1_smd_max']:.2f}, "
              f"S2 {r['s2_smd_max']:.2f} | tamanho efetivo S1 {r['s1_tamanho_efetivo']:.0f}, S2 {r['s2_tamanho_efetivo']:.0f} "
              f"| peso máx. S2 {r['s2_peso_maximo']:.1f} | truncados S2 {r['s2_fracao_truncada']:.3f}")
    print(f"RESULTADO: OK (salvo em {saida}/eixo_c.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
