"""T1 e T2: testes de diagnóstico dos dados (protocolo §8.2). Sem p-valor: AUROC com IC 95%.

Uma linha por imagem (as covariáveis e medidas da tabela de predições; o modelo não entra).

**T1 — a qualidade carrega informação do rótulo?** Regressão logística e gradient boosting com as
medidas de qualidade, posição, idade e sexo prevendo pneumonia, contra uma linha de base só com a
posição; **com e sem o brilho** (§8.3).
- interno: validação cruzada de 5 dobras por paciente dentro de cada base, todas as 8 medidas;
- cruzado: ajusta numa base e avalia na outra, só com as medidas comparáveis entre bases (sem nitidez
  e ruído, que são só dentro da base, §6).

**T2 — existe assinatura de origem?** Classificador de fonte (regressão logística e gradient boosting)
só com as medidas comparáveis, validação cruzada de 5 dobras por paciente, em todas as imagens, só nos
positivos e só nos negativos. AUROC macro um-contra-todos (2 fontes no ensaio, 3 no teste).

Imagens com medida vazia (máscara vazia) saem e são contadas. IC por bootstrap por paciente sobre as
predições fora da dobra.

Uso (notebook "treino", CPU):
    python -m src.analises.t1_t2 --tabela /out/predicoes/tabela_validacao.csv --bootstrap 200
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.analises.metricas import indices_bootstrap
from src.qualidade.medidas import PRIMARIAS, SECUNDARIAS, SO_DENTRO_DA_BASE

TODAS = PRIMARIAS + SECUNDARIAS
COMPARAVEIS = [m for m in TODAS if m not in SO_DENTRO_DA_BASE]
SEMENTE = 2026


def imagens(tab: pd.DataFrame) -> pd.DataFrame:
    img = tab.drop_duplicates(["fonte", "id_imagem"]).reset_index(drop=True)
    img = img.assign(pa=(img["posicao"] == "PA").astype(float), masculino=(img["sexo"] == "M").astype(float),
                     idade=img["idade"].astype(float))
    return img


def _modelos():
    return {"logistica": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
            "boosting": lambda: HistGradientBoostingClassifier(random_state=SEMENTE)}


def _auroc(y, p):
    if p.ndim == 1:
        return roc_auc_score(y, p)
    return roc_auc_score(y, p, multi_class="ovr", average="macro")


def _ic(y, p, pac, n):
    vals = []
    for idx in indices_bootstrap(pac, n, SEMENTE):
        if len(np.unique(y[idx])) == len(np.unique(y)):
            vals.append(_auroc(y[idx], p[idx]))
    return np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)


def _fora_da_dobra(fabrica, x, y, grupos, multiclasse=False):
    p = np.zeros((len(y), len(np.unique(y)))) if multiclasse else np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(x, y, grupos):
        m = fabrica().fit(x[tr], y[tr])
        pr = m.predict_proba(x[te])
        p[te] = pr if multiclasse else pr[:, 1]
    return p


def _limpar(d: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, int]:
    ok = d[cols].notna().all(axis=1)
    return d[ok], int((~ok).sum())


def t1(img: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    linhas = []
    fontes = sorted(img["fonte"].unique())
    for com_brilho in (True, False):
        for versao in ("interno", "cruzado"):
            medidas = TODAS if versao == "interno" else COMPARAVEIS
            medidas = [m for m in medidas if com_brilho or m != "q_brilho"]
            cols, base = medidas + ["pa", "idade", "masculino"], ["pa"]
            pares = [(f, f) for f in fontes] if versao == "interno" else [(a, b) for a in fontes for b in fontes if a != b]
            for f_ajuste, f_aval in pares:
                d_aj, fora_aj = _limpar(img[img["fonte"] == f_ajuste], cols)
                d_av, fora_av = _limpar(img[img["fonte"] == f_aval], cols)
                y, pac = d_av["rotulo"].to_numpy(int), d_av["id_paciente"].to_numpy()
                for nome, fab in _modelos().items():
                    res = {}
                    for rotulo, c in (("modelo", cols), ("linha_de_base", base)):
                        if versao == "interno":
                            p = _fora_da_dobra(fab, d_av[c].to_numpy(float), y, pac)
                        else:
                            p = fab().fit(d_aj[c].to_numpy(float), d_aj["rotulo"].to_numpy(int)).predict_proba(
                                d_av[c].to_numpy(float))[:, 1]
                        res[rotulo] = (_auroc(y, p), *(_ic(y, p, pac, n_boot) if n_boot else (np.nan, np.nan)))
                    linhas.append({"teste": "T1", "versao": versao, "com_brilho": com_brilho, "classificador": nome,
                                   "ajuste": f_ajuste, "avaliacao": f_aval, "imagens": len(d_av),
                                   "fora_por_medida_vazia": fora_av + (fora_aj if versao == "cruzado" else 0),
                                   "auroc": res["modelo"][0], "ic_inf": res["modelo"][1], "ic_sup": res["modelo"][2],
                                   "auroc_so_posicao": res["linha_de_base"][0],
                                   "ganho_sobre_posicao": res["modelo"][0] - res["linha_de_base"][0]})
    return pd.DataFrame(linhas)


def t2(img: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    linhas = []
    multi = img["fonte"].nunique() > 2
    for grupo, d0 in (("todas", img), ("positivos", img[img["rotulo"] == 1]), ("negativos", img[img["rotulo"] == 0])):
        d, fora = _limpar(d0, COMPARAVEIS)
        y = pd.factorize(d["fonte"], sort=True)[0]
        pac = (d["fonte"] + "|" + d["id_paciente"]).to_numpy()
        for nome, fab in _modelos().items():
            p = _fora_da_dobra(fab, d[COMPARAVEIS].to_numpy(float), y, pac, multiclasse=multi)
            ic = _ic(y, p, pac, n_boot) if n_boot else (np.nan, np.nan)
            acerto = float(((p.argmax(1) if multi else (p >= 0.5).astype(int)) == y).mean())
            linhas.append({"teste": "T2", "grupo": grupo, "classificador": nome, "fontes": d["fonte"].nunique(),
                           "imagens": len(d), "fora_por_medida_vazia": fora, "auroc": _auroc(y, p),
                           "ic_inf": ic[0], "ic_sup": ic[1], "acerto": acerto,
                           "acerto_classe_majoritaria": float(np.bincount(y).max() / len(y))})
    return pd.DataFrame(linhas)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tabela", required=True)
    ap.add_argument("--saida", default=None)
    ap.add_argument("--bootstrap", type=int, default=2000)
    a = ap.parse_args(argv)
    tab = pd.read_csv(a.tabela, dtype={"id_imagem": str, "id_paciente": str, "fabricante": "object"})
    tab = tab[tab["degradacao"] == "nenhuma"]
    particao = "+".join(sorted(tab["particao"].unique()))
    saida = a.saida or f"/out/analises/{particao}"
    os.makedirs(saida, exist_ok=True)
    img = imagens(tab)
    r1, r2 = t1(img, a.bootstrap), t2(img, a.bootstrap)
    r1.to_csv(os.path.join(saida, "t1.csv"), index=False)
    r2.to_csv(os.path.join(saida, "t2.csv"), index=False)
    print(f"T1 e T2 | partição {particao} | {len(img)} imagens | bootstrap {a.bootstrap}")
    for _, r in r1.iterrows():
        print(f"  T1 {r['versao']:7s} {'com' if r['com_brilho'] else 'sem'} brilho | {r['classificador']:9s} | "
              f"{r['ajuste']} → {r['avaliacao']:13s} | AUROC {r['auroc']:.3f} [{r['ic_inf']:.3f}; {r['ic_sup']:.3f}] | "
              f"só posição {r['auroc_so_posicao']:.3f} | ganho {r['ganho_sobre_posicao']:+.3f}")
    for _, r in r2.iterrows():
        print(f"  T2 {r['grupo']:9s} | {r['classificador']:9s} | {r['fontes']} fontes, {r['imagens']} imagens | AUROC "
              f"{r['auroc']:.3f} [{r['ic_inf']:.3f}; {r['ic_sup']:.3f}] | acerto {r['acerto']:.3f} "
              f"(classe majoritária {r['acerto_classe_majoritaria']:.3f})")
    print(f"RESULTADO: OK (salvo em {saida}/t1.csv e t2.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
