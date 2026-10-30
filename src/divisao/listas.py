"""Listas de treino, validação e teste (F4, passo 4.8; protocolo §5).

Regras (configs/base.yaml, bloco ``divisao``):

- divisão **por paciente**, 70/10/20, semente 2026;
- estratificada pelo rótulo do paciente: positivo se tiver ao menos uma imagem positiva; no Plus,
  "só incerto" é um terceiro estrato, para os incertos não se concentrarem numa partição;
- PadChest: só teste externo, a base inteira vira a lista de teste (Plano › PadChest › Papel);
- Plus: os negativos do treino são sorteados **uma vez**, 4 por positivo (F4 › Todas › Receita e
  F4 › Plus › Subamostra fixa). Os incertos ficam fora do treino (F3 › Plus › Rótulo).

Saídas no ``/out`` do notebook da base (os IDs não vão para o repositório, por causa dos termos de uso):

    /out/<base>/listas/treino.csv
    /out/<base>/listas/validacao.csv
    /out/<base>/listas/treino_subamostra.csv          (só Plus: o que o modelo usa)
    /out/<base>/listas_teste_trancado/teste.csv       (trancado até 21/11)
    /out/<base>/listas/resumo.json                    (contagens e SHA-256 de cada lista)

O SHA-256 é calculado sobre os IDs ordenados, um por linha; o mesmo conjunto dá sempre o mesmo hash.
O ``resumo.json`` (sem IDs) é o que vai para o repositório e para o diário.

Uso (no notebook da base):
    python -m src.divisao.listas --base plus
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os

import numpy as np
import pandas as pd

SEMENTE = 2026
FRACOES = {"treino": 0.70, "validacao": 0.10, "teste": 0.20}
RAZAO_NEG_POR_POS = 4

# coluna que diz qual imagem do mesmo paciente é a mais antiga (regra F3 › Todas › Duplicatas)
# Remover uma imagem não muda a divisão por paciente nem o sorteio do treino do Plus: o paciente
# continua no mesmo estrato, e o sorteio só olha os negativos do treino.
ORDEM_ANTIGUIDADE = {"rsna": "seguimento", "plus": "patient_report_date_order"}

BASES = {
    "plus": {"tabela": "/out/plus_elegiveis_v2.csv", "id": "chave_dcm", "paciente": "deid_patient_id",
             "saida": "/out/plus"},
    "rsna": {"tabela": "/out/rsna/rsna_elegiveis.csv", "id": "SOPInstanceUID", "paciente": "nih_paciente",
             "saida": "/out/rsna"},
    "padchest": {"tabela": "/out/padchest_recorte.csv", "id": "ImageID", "paciente": "PatientID",
                 "saida": "/out/padchest"},
}


# ------------------------------------------------------------------ rótulo
def rotulo(df: pd.DataFrame, base: str) -> pd.Series:
    """'positivo', 'negativo' ou 'incerto' por imagem, conforme a regra primária de cada base."""
    if base == "plus":
        return df["pn_cat"].map({"positivo": "positivo", "incerto": "incerto"}).fillna("negativo")
    col = "positivo" if base == "rsna" else "pos"
    return np.where(df[col].astype(str).str.lower().isin(["true", "1"]), "positivo", "negativo")


def sha256_ids(ids) -> str:
    texto = "".join(f"{i}\n" for i in sorted(map(str, ids)))
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ divisão
def dividir(df: pd.DataFrame, col_paciente: str, col_rotulo: str = "rotulo", semente: int = SEMENTE,
            fracoes: dict = FRACOES) -> pd.Series:
    """Partição de cada imagem, sorteada por paciente dentro do estrato do paciente."""
    ordem = {"positivo": 0, "incerto": 1, "negativo": 2}
    estrato = (df.assign(_o=df[col_rotulo].map(ordem)).groupby(col_paciente)["_o"].min()
               .map({v: k for k, v in ordem.items()}))
    rng = np.random.default_rng(semente)
    particao = {}
    for nome in ["positivo", "incerto", "negativo"]:                     # ordem fixa: reprodutível
        pacientes = np.array(sorted(estrato[estrato == nome].index.astype(str)))
        pacientes = pacientes[rng.permutation(len(pacientes))]
        n = len(pacientes)
        c1 = int(round(n * fracoes["treino"]))
        c2 = c1 + int(round(n * fracoes["validacao"]))
        for i, p in enumerate(pacientes):
            particao[p] = "treino" if i < c1 else "validacao" if i < c2 else "teste"
    return df[col_paciente].astype(str).map(particao)


def subamostrar(treino: pd.DataFrame, razao: int = RAZAO_NEG_POR_POS, semente: int = SEMENTE) -> pd.DataFrame:
    """Todos os positivos do treino + negativos sorteados (razao por positivo). Incertos ficam fora."""
    pos = treino[treino["rotulo"] == "positivo"]
    neg = treino[treino["rotulo"] == "negativo"]
    n = min(len(neg), razao * len(pos))
    sorteados = neg.sort_values("id").sample(n, random_state=semente)
    return pd.concat([pos, sorteados]).sort_values("id")


def contar(df: pd.DataFrame, col_paciente: str) -> dict:
    pos = int((df["rotulo"] == "positivo").sum())
    inc = int((df["rotulo"] == "incerto").sum())
    return {"imagens": len(df), "pacientes": int(df[col_paciente].nunique()), "positivos": pos,
            "incertos": inc, "prevalencia_incerto_excluido": round(pos / max(len(df) - inc, 1), 4),
            "sha256": sha256_ids(df["id"])}


# ------------------------------------------------------------------ execução
def excluir_duplicatas_pixel(df: pd.DataFrame, md5: pd.DataFrame, col_id: str, col_paciente: str,
                             col_ordem: str | None) -> tuple[pd.DataFrame, list[str]]:
    """Regra F3 › Todas › Duplicatas para pixels idênticos (MD5 dos pixels, da passada completa).

    Pacientes diferentes no mesmo grupo: saem todos. Mesmo paciente: fica a mais antiga (menor ``col_ordem``).
    ``md5``: colunas ``id`` e ``md5_pixels``. Devolve (tabela sem as duplicatas, IDs retirados).
    """
    m = df.merge(md5[["id", "md5_pixels"]].astype(str), left_on=df[col_id].astype(str), right_on="id", how="left")
    grupos = m[m["md5_pixels"].notna() & m.duplicated("md5_pixels", keep=False)]
    sair = []
    for _, g in grupos.groupby("md5_pixels"):
        if g[col_paciente].nunique() > 1:
            sair += g["id"].tolist()
        else:
            g = g.sort_values([col_ordem, "id"] if col_ordem else ["id"])
            sair += g["id"].iloc[1:].tolist()
    return df[~df[col_id].astype(str).isin(sair)], sair


def gerar(base: str, tabela: str | None = None, saida: str | None = None) -> dict:
    cfg = BASES[base]
    tabela, saida = tabela or cfg["tabela"], saida or cfg["saida"]
    df = pd.read_csv(tabela)
    if base == "padchest" and "PatientSex_DICOM" in df.columns:
        df = df[df["PatientSex_DICOM"].isin(["M", "F"])]                  # base final: sexo registrado
    if base == "rsna" and "idade" in df.columns:
        df = df[df["idade"].between(18, 100)]   # o CSV de elegíveis ainda tem as 5 imagens acima de 100 anos
    df = df.drop_duplicates(cfg["id"]).copy()
    retiradas = []
    blocos = glob.glob(f"{saida}/passada/blocos/*.csv")
    if base in ORDEM_ANTIGUIDADE and blocos:       # pixels idênticos encontrados na passada completa
        md5 = pd.concat([pd.read_csv(b, usecols=["id", "md5_pixels"]) for b in blocos])
        ordem = ORDEM_ANTIGUIDADE[base] if ORDEM_ANTIGUIDADE[base] in df.columns else None
        df, retiradas = excluir_duplicatas_pixel(df, md5, cfg["id"], cfg["paciente"], ordem)
    df["id"] = df[cfg["id"]].astype(str)
    df["rotulo"] = rotulo(df, base)
    pac = cfg["paciente"]

    df["particao"] = "teste" if base == "padchest" else dividir(df, pac)
    if base != "padchest":
        cruzados = df.groupby(pac)["particao"].nunique()
        assert (cruzados == 1).all(), "paciente em mais de uma partição"

    pasta, trancada = f"{saida}/listas", f"{saida}/listas_teste_trancado"
    os.makedirs(pasta, exist_ok=True)
    os.makedirs(trancada, exist_ok=True)
    cols = ["id", pac, "rotulo"]
    resumo = {"base": base, "semente": SEMENTE, "fracoes": FRACOES, "elegiveis": len(df),
              "duplicatas_pixel_retiradas": len(retiradas), "listas": {}}
    for nome in ["treino", "validacao", "teste"]:
        parte = df[df["particao"] == nome]
        if parte.empty:
            continue
        destino = f"{trancada}/teste.csv" if nome == "teste" else f"{pasta}/{nome}.csv"
        parte[cols].sort_values("id").to_csv(destino, index=False)
        resumo["listas"][nome] = contar(parte, pac)
    if base == "plus":
        sub = subamostrar(df[df["particao"] == "treino"])
        sub[cols].to_csv(f"{pasta}/treino_subamostra.csv", index=False)
        resumo["listas"]["treino_subamostra"] = {**contar(sub, pac), "razao_neg_por_pos": RAZAO_NEG_POR_POS}
    with open(f"{pasta}/resumo.json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)
    return resumo


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", choices=sorted(BASES), required=True)
    a = ap.parse_args(argv)
    r = gerar(a.base)
    print(f"{a.base}: {r['elegiveis']} imagens elegíveis | semente {r['semente']} | "
          f"duplicatas de pixel retiradas: {r['duplicatas_pixel_retiradas']}")
    print(pd.DataFrame(r["listas"]).T[["imagens", "pacientes", "positivos", "incertos",
                                       "prevalencia_incerto_excluido", "sha256"]].to_string())
    print(f"\nListas em {BASES[a.base]['saida']}/listas/ e {BASES[a.base]['saida']}/listas_teste_trancado/")
    print("Nenhum paciente em duas partições." if a.base != "padchest" else "PadChest: base inteira como teste externo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
