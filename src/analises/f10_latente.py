"""Espaço latente e explicabilidade (F10, passos 10.1 a 10.6; acréscimo A2; D-019). **Exploratória**:
FDR (Benjamini-Hochberg) em todos os p desta fase, e nenhuma decisão de modelagem tirada daqui.

Duas etapas:

1. ``--extrair --base X`` (notebook da base, GPU, depois do passo 6.8; mesma trava do teste):
   - **amostra** da base: todos os positivos até 1.000 e negativos até completar 3.000 imagens
     (semente 2026), da tabela de predições do teste da base (``/out/teste/<base>/tabela_teste.csv.gz``);
   - **embeddings** (10.1): penúltima camada da DenseNet121 (ReLU de ``norm5`` + média global = 1.024
     dimensões), para as 15 execuções, em float16 → ``/out/teste/<base>/f10_embeddings.npz``;
   - **Grad-CAM** (10.5): para cada modelo, semente 0, até 30 VP, 30 FP e 30 FN sorteados (limiar da
     validação da fonte), mapa na saída de ``norm5`` para o logit; **massa dentro da ROI pulmonar**
     (máscara do mesmo modelo de segmentação das medidas, na imagem de 320 px);
   - **sanidade** (10.6; Adebayo et al., 2018): o mesmo mapa com o bloco denso 4, ``norm5`` e o
     classificador reinicializados (randomização em cascata); correlação de Spearman entre os dois mapas.
     Regra pré-especificada: mediana de Spearman ≥ 0,5 num modelo → mapas desse modelo **não interpretados**.
     → ``/out/teste/<base>/f10_gradcam.csv``.
2. ``--analisar`` (notebook "treino", CPU; arquivos das três bases baixados de ``resultados``):
   - **sonda de origem** (10.2): regressão logística multinomial (padronizada, L2, C = 1) prevendo a fonte
     a partir do embedding, validação cruzada de 5 dobras agrupadas por paciente; AUROC macro (um contra
     os outros) e acurácia balanceada; comparada com a mesma sonda nas medidas **comparáveis** do T2, nas
     mesmas imagens; p da diferença de AUROC por bootstrap por paciente (semente 0 do modelo);
   - **sonda de qualidade** (10.3): ridge (alfa = 1) prevendo brilho e contraste a partir do embedding,
     mesmas dobras; R² fora da dobra, com p por bootstrap (R² ≤ 0);
   - **projeção 2D** (10.4): UMAP (``umap-learn``; sem ele, PCA, e isso fica registrado) da semente 0,
     colorida por fonte, rótulo e tipo de erro; só descritiva;
   - **Grad-CAM**: massa na ROI por tipo de erro e por fonte (Kruskal-Wallis), só nos modelos que passam
     na sanidade;
   - BH em todos os p da fase.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

SEMENTE = 2026
N_AMOSTRA, N_POS_MAX, N_CAM = 3000, 1000, 30
LIMITE_SANIDADE = 0.5
N_BOOT = 200


# ------------------------------------------------------------------ etapa 1 (notebook da base)
def amostra(tab: pd.DataFrame, n: int = N_AMOSTRA, n_pos_max: int = N_POS_MAX) -> pd.DataFrame:
    img = tab.drop_duplicates("id_imagem")[["id_imagem", "id_paciente", "fonte", "rotulo"]]
    rng = np.random.default_rng(SEMENTE)
    pos, neg = img[img["rotulo"] == 1], img[img["rotulo"] == 0]
    pos = pos.iloc[np.sort(rng.choice(len(pos), min(len(pos), n_pos_max), replace=False))]
    neg = neg.iloc[np.sort(rng.choice(len(neg), min(len(neg), n - len(pos)), replace=False))]
    return pd.concat([pos, neg], ignore_index=True)


def _carregar_modelo(pasta: str, disp):
    import torch
    from src.treino.modelo import criar_densenet121
    m = criar_densenet121(False).to(disp)
    m.load_state_dict(torch.load(os.path.join(pasta, "melhor.pt"), map_location=disp, weights_only=True))
    return m.eval()


def embeddings(modelo, caminhos: list[str], disp, lote: int = 64, workers: int = 2) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from src.treino.carregador import ImagensCache
    ld = DataLoader(ImagensCache(pd.DataFrame({"caminho": caminhos, "rotulo": 0}), None), batch_size=lote,
                    shuffle=False, num_workers=workers)
    out = []
    with torch.no_grad():
        for x, _ in ld:
            f = F.relu(modelo.features(x.to(disp)))
            out.append(F.adaptive_avg_pool2d(f, 1).flatten(1).float().cpu().numpy())
    return np.concatenate(out).astype(np.float16)


def gradcam(modelo, x) -> np.ndarray:
    """Grad-CAM (Selvaraju et al.) na saída de norm5 para o logit; mapa 10 × 10, ReLU, sem normalizar."""
    import torch
    import torch.nn.functional as F
    modelo.zero_grad()
    a = modelo.features(x)
    a.retain_grad()
    z = modelo.classifier(F.adaptive_avg_pool2d(F.relu(a), 1).flatten(1))
    z.sum().backward()
    pesos = a.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((pesos * a).sum(1))
    return cam.detach().cpu().numpy()


def randomizar_topo(modelo, semente: int = 0):
    """Cópia do modelo com denseblock4, norm5 e classificador reinicializados (randomização em cascata)."""
    import copy
    import torch
    m = copy.deepcopy(modelo)
    torch.manual_seed(semente)
    for mod in list(m.features.denseblock4.modules()) + [m.features.norm5, m.classifier]:
        if hasattr(mod, "reset_parameters"):
            mod.reset_parameters()
    return m.eval()


def massa_na_roi(cam: np.ndarray, mascara: np.ndarray) -> float:
    import cv2
    c = cv2.resize(cam.astype(np.float32), mascara.shape[::-1], interpolation=cv2.INTER_LINEAR)
    tot = c.sum()
    return float((c * (mascara > 0)).sum() / tot) if tot > 0 else float("nan")


def extrair(base: str, modelos: str = "/out/modelos", segmentar=None) -> dict:
    import cv2
    import torch
    from scipy.stats import spearmanr
    from src.perturbations.eixo_b import tabela_da_particao
    from src.predicoes.pos_treino import SEMENTES, TAXA
    from src.predicoes.teste import CACHE
    from src.cache.hashes import nome_cache
    from src.treino.carregador import ImagensCache
    tabela_da_particao(base, "teste")                       # trava
    pasta = f"/out/teste/{base}"
    tab = pd.read_csv(os.path.join(pasta, "tabela_teste.csv.gz"), dtype={"id_imagem": str, "id_paciente": str})
    am = amostra(tab)
    am["caminho"] = [os.path.join(CACHE[base], nome_cache(i)) for i in am["id_imagem"]]
    disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb = {}
    for m in TAXA:
        for s in SEMENTES:
            ex = f"{m}_s{s}_lr{TAXA[m]:g}"
            emb[ex] = embeddings(_carregar_modelo(os.path.join(modelos, ex), disp), am["caminho"].tolist(), disp)
            print(f"  {ex}: embeddings {emb[ex].shape}", flush=True)
    np.savez_compressed(os.path.join(pasta, "f10_embeddings.npz"), id_imagem=am["id_imagem"].to_numpy(),
                        id_paciente=am["id_paciente"].to_numpy(), fonte=am["fonte"].to_numpy(),
                        rotulo=am["rotulo"].to_numpy(), **emb)
    if segmentar is None:
        from src.imagem.mascara import segmentar
    rng = np.random.default_rng(SEMENTE)
    linhas = []
    for m in TAXA:
        ex = f"{m}_s0_lr{TAXA[m]:g}"
        lim = json.load(open(os.path.join(modelos, ex, "pos_treino.json"), encoding="utf-8"))["limiar"]["limiar_logit"]
        g = tab[tab["id_execucao"] == ex]
        tipos = {"VP": g[(g["rotulo"] == 1) & (g["logit"] >= lim)], "FP": g[(g["rotulo"] == 0) & (g["logit"] >= lim)],
                 "FN": g[(g["rotulo"] == 1) & (g["logit"] < lim)]}
        modelo = _carregar_modelo(os.path.join(modelos, ex), disp)
        aleat = randomizar_topo(modelo)
        for tipo, d in tipos.items():
            d = d.iloc[np.sort(rng.choice(len(d), min(len(d), N_CAM), replace=False))] if len(d) else d
            for ident in d["id_imagem"]:
                cam_arq = os.path.join(CACHE[base], nome_cache(ident))
                x, _ = ImagensCache(pd.DataFrame({"caminho": [cam_arq], "rotulo": [0]}), None)[0]
                x = x.unsqueeze(0).to(disp).requires_grad_(False)
                c, cr = gradcam(modelo, x)[0], gradcam(aleat, x)[0]
                img8 = cv2.imread(cam_arq, cv2.IMREAD_GRAYSCALE)
                mask = segmentar(img8)
                rho = spearmanr(c.ravel(), cr.ravel()).statistic if c.std() > 0 and cr.std() > 0 else np.nan
                linhas.append({"modelo": m, "fonte": d["fonte"].iloc[0] if len(d) else base, "id_imagem": ident,
                               "tipo": tipo, "massa_roi": massa_na_roi(c, mask),
                               "massa_roi_aleatorio": massa_na_roi(cr, mask), "spearman_aleatorio": rho})
        print(f"  Grad-CAM {ex}: {sum(1 for l in linhas if l['modelo'] == m)} mapas", flush=True)
    pd.DataFrame(linhas).to_csv(os.path.join(pasta, "f10_gradcam.csv"), index=False)
    return {"amostra": len(am), "positivos": int(am["rotulo"].sum()), "mapas": len(linhas),
            "arquivos": [os.path.join(pasta, "f10_embeddings.npz"), os.path.join(pasta, "f10_gradcam.csv")]}


# ------------------------------------------------------------------ etapa 2 (notebook "treino")
def _dobras(pac: np.ndarray, k: int = 5) -> np.ndarray:
    from sklearn.model_selection import GroupKFold
    d = np.empty(len(pac), int)
    for i, (_, te) in enumerate(GroupKFold(k).split(pac, groups=pac)):
        d[te] = i
    return d


def _fora_da_dobra_fonte(X: np.ndarray, y: np.ndarray, dobras: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    classes = np.unique(y)
    p = np.zeros((len(y), len(classes)))
    for k in np.unique(dobras):
        tr = dobras != k
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X[tr], y[tr])
        p[~tr] = m.predict_proba(X[~tr])
    return p


def _auroc_macro(y, p) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p, multi_class="ovr", average="macro"))


def _bal(y, p) -> float:
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y, np.unique(y)[p.argmax(1)]))


def _r2_fora(X, alvo, dobras) -> np.ndarray:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    pr = np.empty(len(alvo))
    for k in np.unique(dobras):
        tr = dobras != k
        pr[~tr] = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X[tr], alvo[tr]).predict(X[~tr])
    return pr


def _r2(y, p) -> float:
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def sondas(emb: dict, meta: pd.DataFrame, medidas: pd.DataFrame, n_boot: int = N_BOOT) -> pd.DataFrame:
    """emb: {execução: matriz}; meta: id_imagem, id_paciente, fonte, rotulo (mesma ordem); medidas: por imagem."""
    from src.analises.metricas import indices_bootstrap
    from src.analises.t1_t2 import COMPARAVEIS
    med = medidas.set_index("id_imagem").loc[meta["id_imagem"]]
    ok = med[COMPARAVEIS].notna().all(1).to_numpy()
    y, pac = meta["fonte"].to_numpy()[ok], meta["id_paciente"].to_numpy()[ok]
    dob = _dobras(pac)
    Xq = med[COMPARAVEIS].to_numpy(float)[ok]
    pq = _fora_da_dobra_fonte(Xq, y, dob)
    auc_q = _auroc_macro(y, pq)
    boots = list(indices_bootstrap(pac, n_boot, SEMENTE))
    linhas = []
    for ex, E in emb.items():
        X = E.astype(np.float32)[ok]
        pe = _fora_da_dobra_fonte(X, y, dob)
        lin = {"execucao": ex, "modelo": ex.split("_")[0], "semente": int(ex.split("_s")[1].split("_")[0]),
               "n": int(ok.sum()), "auroc_fonte_embedding": _auroc_macro(y, pe), "acuracia_bal_embedding": _bal(y, pe),
               "auroc_fonte_medidas": auc_q, "acuracia_bal_medidas": _bal(y, pq)}
        lin["diferenca_auroc"] = lin["auroc_fonte_embedding"] - auc_q
        if lin["semente"] == 0 and n_boot:
            d = []
            for ii in boots:
                if len(np.unique(y[ii])) == len(np.unique(y)):
                    d.append(_auroc_macro(y[ii], pe[ii]) - _auroc_macro(y[ii], pq[ii]))
            d = np.array(d)
            lin["p_diferenca"] = float(min(1, 2 * min((1 + (d <= 0).sum()) / (len(d) + 1), (1 + (d >= 0).sum()) / (len(d) + 1))))
        for q in ("q_brilho", "q_contraste"):
            alvo = med[q].to_numpy(float)[ok]
            pr = _r2_fora(X, alvo, dob)
            lin[f"r2_{q}"] = _r2(alvo, pr)
            if lin["semente"] == 0 and n_boot:
                r = np.array([_r2(alvo[ii], pr[ii]) for ii in boots])
                lin[f"p_r2_{q}"] = float((1 + (r <= 0).sum()) / (len(r) + 1))
        linhas.append(lin)
    return pd.DataFrame(linhas)


def resumo_gradcam(g: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import kruskal
    linhas = []
    for m, d in g.groupby("modelo"):
        rho = float(np.nanmedian(d["spearman_aleatorio"]))
        lin = {"modelo": m, "mapas": len(d), "spearman_mediano_aleatorio": rho, "interpretavel": rho < LIMITE_SANIDADE}
        for t, dt in d.groupby("tipo"):
            lin[f"massa_roi_{t}"] = float(dt["massa_roi"].median())
        for k, col in (("tipo", "p_tipo"), ("fonte", "p_fonte")):
            grupos = [x["massa_roi"].dropna().to_numpy() for _, x in d.groupby(k) if x["massa_roi"].notna().sum() >= 3]
            lin[col] = float(kruskal(*grupos).pvalue) if lin["interpretavel"] and len(grupos) >= 2 else np.nan
        linhas.append(lin)
    return pd.DataFrame(linhas)


def projecao(emb: np.ndarray, meta: pd.DataFrame, destino: str, titulo: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(emb.astype(np.float32))
    try:
        import umap
        P, metodo = umap.UMAP(random_state=SEMENTE).fit_transform(X), "UMAP"
    except ImportError:
        from sklearn.decomposition import PCA
        P, metodo = PCA(2, random_state=SEMENTE).fit_transform(X), "PCA (umap-learn ausente)"
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for a, col in zip(ax, ("fonte", "rotulo", "tipo_erro")):
        for v, d in meta.groupby(col):
            a.scatter(P[d.index, 0], P[d.index, 1], s=3, alpha=0.5, label=str(v))
        a.set_title(f"{col}"); a.legend(markerscale=4, fontsize=8)
    fig.suptitle(f"{titulo} — {metodo}")
    fig.tight_layout(); fig.savefig(destino, dpi=120); plt.close(fig)
    return metodo


def analisar(resultados: str, tabela: str, treinos: str, saida: str) -> int:
    from src.analises.estatistica import benjamini_hochberg
    os.makedirs(saida, exist_ok=True)
    tab = pd.read_csv(tabela, dtype={"id_imagem": str, "id_paciente": str})
    arqs = sorted(glob.glob(os.path.join(resultados, "*__teste__f10_embeddings.npz")))
    zs = [np.load(a, allow_pickle=True) for a in arqs]
    meta = pd.DataFrame({k: np.concatenate([z[k] for z in zs]) for k in ("id_imagem", "id_paciente", "fonte", "rotulo")})
    meta["id_imagem"] = meta["id_imagem"].astype(str)
    execs = [k for k in zs[0].files if k not in ("id_imagem", "id_paciente", "fonte", "rotulo")]
    emb = {ex: np.concatenate([z[ex] for z in zs]) for ex in execs}
    medidas = tab.drop_duplicates("id_imagem")
    s = sondas(emb, meta, medidas)
    gc = pd.concat([pd.read_csv(a) for a in sorted(glob.glob(os.path.join(resultados, "*__teste__f10_gradcam.csv")))],
                   ignore_index=True)
    g = resumo_gradcam(gc)
    # tipo de erro para a projeção (semente 0 de cada modelo)
    metodos = {}
    for ex in [e for e in execs if "_s0_" in e]:
        lim = json.load(open(os.path.join(treinos, ex, "pos_treino.json"), encoding="utf-8"))["limiar"]["limiar_logit"]
        z = tab[tab["id_execucao"] == ex].set_index("id_imagem")["logit"].reindex(meta["id_imagem"]).to_numpy()
        pred = z >= lim
        m2 = meta.copy()
        m2["tipo_erro"] = np.where(m2["rotulo"] == 1, np.where(pred, "VP", "FN"), np.where(pred, "FP", "VN"))
        metodos[ex] = projecao(emb[ex], m2, os.path.join(saida, f"f10_projecao_{ex.split('_')[0]}.png"), ex)
    # BH em todos os p da fase
    ps = [("sonda", i, c) for i in s.index for c in s.columns if c.startswith("p_") and pd.notna(s.loc[i, c])]
    ps += [("gradcam", i, c) for i in g.index for c in ("p_tipo", "p_fonte") if pd.notna(g.loc[i, c])]
    if ps:
        aj = benjamini_hochberg(np.array([(s if t == "sonda" else g).loc[i, c] for t, i, c in ps]))
        for (t, i, c), v in zip(ps, aj):
            (s if t == "sonda" else g).loc[i, c + "_bh"] = v
    s.to_csv(os.path.join(saida, "f10_sondas.csv"), index=False)
    g.to_csv(os.path.join(saida, "f10_gradcam_resumo.csv"), index=False)
    print("Sonda de origem (média das sementes) e de qualidade:")
    print(s.groupby("modelo")[["auroc_fonte_embedding", "auroc_fonte_medidas", "diferenca_auroc",
                               "r2_q_brilho", "r2_q_contraste"]].mean().round(3).to_string())
    print("\nGrad-CAM:")
    print(g.round(3).to_string(index=False))
    print(f"Projeção: {sorted(set(metodos.values()))}")
    print(f"RESULTADO: OK ({saida}/f10_sondas.csv, f10_gradcam_resumo.csv, f10_projecao_*.png)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extrair", action="store_true")
    ap.add_argument("--analisar", action="store_true")
    ap.add_argument("--base", choices=["plus", "rsna", "padchest"])
    ap.add_argument("--enviar", action="store_true", help="com --extrair: envia ao dataset resultados")
    ap.add_argument("--resultados", default="/out/resultados")
    ap.add_argument("--tabela", default="/out/predicoes/tabela_teste.csv")
    ap.add_argument("--treinos", default="/out/modelos")
    ap.add_argument("--saida", default="/out/analises/teste")
    a = ap.parse_args(argv)
    if a.extrair:
        r = extrair(a.base)
        print(f"{a.base}: amostra {r['amostra']} ({r['positivos']} positivos) | {r['mapas']} mapas de Grad-CAM")
        if a.enviar:
            from src.cache.resultados import enviar
            enviar(r["arquivos"], origem=f"{a.base}__teste")
        print("RESULTADO: OK")
        return 0
    if a.analisar:
        return analisar(a.resultados, a.tabela, a.treinos, a.saida)
    ap.error("escolha --extrair ou --analisar")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
