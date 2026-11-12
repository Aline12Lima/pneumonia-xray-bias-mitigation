"""Revisão qualitativa dos erros (F9, passos 9.5 a 9.7; acréscimo A1; D-020). Exploratória.

Três etapas, em notebooks diferentes:

1. ``--sortear`` (notebook "treino", depois de ``src.predicoes.juntar_teste``): para cada par modelo ×
   base **externa** (M1 → RSNA e PadChest; M2 → Plus e PadChest; M3 → PadChest), com as predições da
   **semente 0** e o limiar da validação da fonte: 25 FP e 25 FN de maior confiança (FP com maior
   probabilidade; FN com menor) e 25 FP e 25 FN aleatórios entre os restantes (semente 2026). Grava a
   amostra (IDs só no Redivis) e a sua SHA-256, e uma **folha de codificação por base** com um código
   sequencial por imagem (uma imagem que aparece em mais de um par é codificada uma vez só). A folha não
   tem ID, modelo, probabilidade nem tipo de erro; só o código e o rótulo de referência (necessário para
   "rótulo provavelmente errado").
2. ``--pranchas --base X`` (notebook da base): monta pranchas PNG de 4 × 5 imagens de 320 px com o
   código de cada uma, em ordem de código. As pranchas ficam só no Redivis (termos de uso).
3. Codificação (fora do código): preencher a folha com 0/1 em cada categoria do livro de códigos, mais
   ``codificador`` e ``observacao``. A folha preenchida pode ir ao repositório (``docs/f9/``): não tem ID.
4. ``--resumo`` (notebook "treino"): junta as folhas à amostra e tabula a frequência de cada categoria
   por par, tipo de erro e critério (maior confiança × aleatório). Sem teste de hipótese (exploratória).

Livro de códigos (passo 9.6, definido antes do destrancamento): ``CATEGORIAS``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

CATEGORIAS = {
    "dispositivos_tubos": "tubo, cateter, eletrodo, marca-passo ou outro dispositivo sobre o tórax",
    "marcadores_texto": "letra, marcador de lado, régua ou texto queimado na imagem",
    "rotacao_corte": "tórax rodado, cortado ou fora do quadro",
    "inspiracao_baixa": "inspiração baixa (menos de 9 arcos costais posteriores visíveis)",
    "achado_concorrente": "derrame, atelectasia, edema ou outra opacidade que não é pneumonia",
    "rotulo_provavelmente_errado": "a imagem parece contradizer o rótulo de referência",
    "exposicao_extrema": "imagem muito clara ou muito escura, ou com saturação visível",
}
PARES_EXTERNOS = {"M1": ["rsna", "padchest"], "M2": ["chexpert_plus", "padchest"], "M3": ["padchest"]}
BASE_DA_FONTE = {"chexpert_plus": "plus", "rsna": "rsna", "padchest": "padchest"}
N_POR_GRUPO = 25
SEMENTE = 2026
SAIDA = "/out/analises/teste"


def _limiar(treinos: str, execucao: str) -> float:
    return json.load(open(os.path.join(treinos, execucao, "pos_treino.json"), encoding="utf-8"))["limiar"]["limiar_logit"]


def sortear(tab: pd.DataFrame, treinos: str, n: int = N_POR_GRUPO) -> pd.DataFrame:
    rng = np.random.default_rng(SEMENTE)
    linhas = []
    for modelo, fontes in PARES_EXTERNOS.items():
        s0 = tab[(tab["modelo"] == modelo) & (tab["semente"] == 0)]
        if s0.empty:
            continue
        lim = _limiar(treinos, s0["id_execucao"].iloc[0])
        for fonte in fontes:
            g = s0[s0["fonte"] == fonte]
            for tipo, sel, asc in (("FP", (g["rotulo"] == 0) & (g["logit"] >= lim), False),
                                   ("FN", (g["rotulo"] == 1) & (g["logit"] < lim), True)):
                e = g[sel].sort_values(["probabilidade", "id_imagem"], ascending=[asc, True])
                conf = e.head(n)
                resto = e.iloc[len(conf):]
                alea = resto.iloc[np.sort(rng.choice(len(resto), min(n, len(resto)), replace=False))] if len(resto) else resto
                for crit, d in (("maior_confianca", conf), ("aleatorio", alea)):
                    linhas.append(d.assign(tipo_erro=tipo, criterio=crit)[
                        ["modelo", "fonte", "id_imagem", "id_paciente", "rotulo", "probabilidade", "tipo_erro", "criterio"]])
    amostra = pd.concat(linhas, ignore_index=True) if linhas else pd.DataFrame()
    codigos = {}
    for fonte, g in amostra.groupby("fonte", sort=True):
        for k, i in enumerate(sorted(g["id_imagem"].unique()), 1):
            codigos[(fonte, i)] = f"{BASE_DA_FONTE[fonte][:2].upper()}{k:04d}"
    amostra["codigo"] = [codigos[(f, i)] for f, i in zip(amostra["fonte"], amostra["id_imagem"])]
    return amostra


def folha(amostra: pd.DataFrame, fonte: str) -> pd.DataFrame:
    """Folha de codificação de uma base: código, rótulo de referência e categorias vazias."""
    u = amostra[amostra["fonte"] == fonte].drop_duplicates("codigo").sort_values("codigo")
    f = pd.DataFrame({"codigo": u["codigo"].to_numpy(),
                      "rotulo_referencia": u["rotulo"].map({1: "positivo", 0: "negativo"}).to_numpy()})
    for c in CATEGORIAS:
        f[c] = ""
    f["codificador"] = ""
    f["observacao"] = ""
    return f


def sha256_amostra(amostra: pd.DataFrame) -> str:
    chave = amostra.sort_values(["modelo", "fonte", "tipo_erro", "criterio", "id_imagem"])
    txt = "\n".join(f"{m}|{f}|{t}|{c}|{i}" for m, f, t, c, i in
                    chave[["modelo", "fonte", "tipo_erro", "criterio", "id_imagem"]].itertuples(index=False))
    return hashlib.sha256(txt.encode()).hexdigest()


def pranchas(amostra: pd.DataFrame, base: str, cache: str, destino: str, por_prancha: int = 20) -> list[str]:
    import cv2
    from src.cache.hashes import nome_cache
    fonte = {v: k for k, v in BASE_DA_FONTE.items()}[base]
    u = amostra[amostra["fonte"] == fonte].drop_duplicates("codigo").sort_values("codigo")
    os.makedirs(destino, exist_ok=True)
    arquivos = []
    for p in range(0, len(u), por_prancha):
        bloco = u.iloc[p:p + por_prancha]
        tela = np.full((4 * 340, 5 * 320), 255, np.uint8)
        for k, (cod, ident) in enumerate(zip(bloco["codigo"], bloco["id_imagem"])):
            img = cv2.imread(os.path.join(cache, nome_cache(ident)), cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((320, 320), np.uint8)
            y, x = (k // 5) * 340, (k % 5) * 320
            tela[y + 20:y + 340, x:x + 320] = img
            cv2.putText(tela, cod, (x + 4, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1, cv2.LINE_AA)
        arq = os.path.join(destino, f"prancha_{base}_{p // por_prancha + 1:02d}.png")
        cv2.imwrite(arq, tela)
        arquivos.append(arq)
    return arquivos


def resumo(amostra: pd.DataFrame, folhas: pd.DataFrame) -> pd.DataFrame:
    d = amostra.merge(folhas, on="codigo", how="left")
    cats = list(CATEGORIAS)
    for c in cats:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    g = d.groupby(["modelo", "fonte", "tipo_erro", "criterio"])
    out = g[cats].mean().round(3)
    out.insert(0, "n_codificadas", g[cats[0]].count())
    out.insert(0, "n", g.size())
    return out.reset_index()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sortear", action="store_true")
    ap.add_argument("--pranchas", action="store_true")
    ap.add_argument("--resumo", action="store_true")
    ap.add_argument("--base", choices=["plus", "rsna", "padchest"])
    ap.add_argument("--tabela", default="/out/predicoes/tabela_teste.csv")
    ap.add_argument("--treinos", default="/out/modelos")
    ap.add_argument("--amostra", default=os.path.join(SAIDA, "f9_amostra.csv"))
    ap.add_argument("--folhas", nargs="*", default=[], help="folhas preenchidas (CSV)")
    a = ap.parse_args(argv)
    if a.sortear:
        tab = pd.read_csv(a.tabela, dtype={"id_imagem": str, "id_paciente": str})
        am = sortear(tab, a.treinos)
        os.makedirs(SAIDA, exist_ok=True)
        am.to_csv(a.amostra, index=False)
        h = sha256_amostra(am)
        for fonte in sorted(am["fonte"].unique()):
            folha(am, fonte).to_csv(os.path.join(SAIDA, f"f9_folha_{BASE_DA_FONTE[fonte]}.csv"), index=False)
        print(am.groupby(["modelo", "fonte", "tipo_erro", "criterio"]).size().to_string())
        print(f"Imagens a codificar por base: {am.drop_duplicates('codigo').groupby('fonte').size().to_dict()}")
        print(f"SHA-256 da amostra: {h}")
        print(f"RESULTADO: OK (amostra em {a.amostra}; folhas f9_folha_<base>.csv em {SAIDA})")
    elif a.pranchas:
        from src.predicoes.teste import CACHE
        am = pd.read_csv(a.amostra, dtype={"id_imagem": str})
        arqs = pranchas(am, a.base, CACHE[a.base], f"/out/{a.base}/f9_pranchas")
        print(f"RESULTADO: {len(arqs)} pranchas em /out/{a.base}/f9_pranchas")
    elif a.resumo:
        am = pd.read_csv(a.amostra, dtype={"id_imagem": str})
        fl = pd.concat([pd.read_csv(f) for f in a.folhas], ignore_index=True)
        r = resumo(am, fl)
        r.to_csv(os.path.join(SAIDA, "f9_resumo.csv"), index=False)
        print(r.to_string(index=False))
        print(f"RESULTADO: OK ({SAIDA}/f9_resumo.csv)")
    else:
        ap.error("escolha --sortear, --pranchas ou --resumo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
