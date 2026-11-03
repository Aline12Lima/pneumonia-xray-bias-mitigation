"""Covariáveis de treino e validação para o notebook de treino (F5, passo 5.7; protocolo §7.4).

A tabela de predições (``configs/schema_predicoes.yaml``) precisa de estudo, posição, idade, sexo,
fabricante e das medidas de qualidade. Elas estão aqui, no notebook da base (tabela de elegíveis e
passada completa), e não no ``cache_treino``. Este módulo junta essas colunas **só para as imagens
de treino e validação** e envia o arquivo para a tabela ``<base>_listas`` do ``cache_treino``, ao lado
das listas. O teste nunca sai do notebook da base: o filtro é pelas listas de treino e validação.

Arquivo: ``/out/<base>/covariaveis.csv``, colunas ``id``, ``id_estudo``, ``posicao`` (AP/PA), ``idade``,
``sexo`` (F/M), ``fabricante`` (nome unificado; vazio no RSNA) e as 8 medidas de qualidade
(``q_brilho`` … ``q_ruido``, resolução original, da passada completa).

Uso (no notebook da base, CPU):
    python -m src.cache.covariaveis --base plus             # gera e confere
    python -m src.cache.covariaveis --base plus --enviar    # gera, confere e envia ao cache_treino
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from src.qualidade.fabricante import normalizar_fabricante
from src.qualidade.medidas import PRIMARIAS, SECUNDARIAS

MEDIDAS = PRIMARIAS + SECUNDARIAS
NOME = "covariaveis.csv"

BASES = {
    "plus": {"elegiveis": "/out/plus_elegiveis_v2.csv", "listas": ["/out/plus/listas/treino_subamostra.csv",
                                                                   "/out/plus/listas/validacao.csv"],
             "colunas": {"chave_dcm": "id", "estudo": "id_estudo", "ap_pa": "posicao", "idade": "idade",
                         "sex": "sexo"},
             "sexo": {"Male": "M", "Female": "F"}},
    "rsna": {"elegiveis": "/out/rsna/rsna_elegiveis.csv", "listas": ["/out/rsna/listas/treino.csv",
                                                                     "/out/rsna/listas/validacao.csv"],
             "colunas": {"SOPInstanceUID": "id", "StudyInstanceUID": "id_estudo", "proj": "posicao",
                         "idade": "idade", "sexo": "sexo"},
             "sexo": {"M": "M", "F": "F"}},
}


def montar(base: str, elegiveis: pd.DataFrame, passada: pd.DataFrame, ids: pd.Series) -> pd.DataFrame:
    """Covariáveis das imagens ``ids`` (treino e validação). Para se faltar alguma imagem."""
    cfg = BASES[base]
    el = elegiveis[list(cfg["colunas"])].rename(columns=cfg["colunas"])
    el["id"] = el["id"].astype(str)
    el = el.drop_duplicates("id")
    el["sexo"] = el["sexo"].map(cfg["sexo"])
    el["idade"] = pd.to_numeric(el["idade"], errors="coerce").round().astype("Int64")
    pa = passada[["id", "fabricante"] + MEDIDAS].copy()
    pa["id"] = pa["id"].astype(str)
    pa = pa.drop_duplicates("id", keep="last")
    pa["_na_passada"] = True
    pa["fabricante"] = [None if pd.isna(f) or str(f).strip() in ("", "nan") else normalizar_fabricante(f)
                        for f in pa["fabricante"]]
    ids = pd.Series(sorted(set(ids.astype(str))), name="id")
    df = ids.to_frame().merge(el, on="id", how="left").merge(pa, on="id", how="left")
    # falta de verdade = imagem sem linha nas elegíveis ou na passada. Medidas vazias com a linha presente
    # são permitidas (o schema aceita): é a máscara de pulmão vazia, marcada na passada (q_aux_roi_vazia)
    sem_el, sem_pa = int(df["posicao"].isna().sum()), int(df["_na_passada"].isna().sum())
    if sem_el or sem_pa:
        raise RuntimeError(f"{base}: {sem_el} imagens sem linha nas elegíveis e {sem_pa} sem linha na passada")
    return df[["id", "id_estudo", "posicao", "idade", "sexo", "fabricante"] + MEDIDAS]


def resumo(df: pd.DataFrame) -> dict:
    return {"imagens": len(df), "posicao": df["posicao"].value_counts().to_dict(),
            "sexo": df["sexo"].value_counts(dropna=False).to_dict(),
            "idade_min_max": [int(df["idade"].min()), int(df["idade"].max())],
            "idade_vazia": int(df["idade"].isna().sum()), "idade_fora_18_100": int((~df["idade"].between(18, 100)).sum()),
            "fabricante": df["fabricante"].value_counts(dropna=False).to_dict(),
            "medidas_vazias (máscara vazia)": int(df[[m for m in MEDIDAS if m != "q_enquadramento"]].isna().all(axis=1).sum())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", choices=sorted(BASES), required=True)
    ap.add_argument("--enviar", action="store_true", help="envia para a tabela <base>_listas do cache_treino")
    a = ap.parse_args(argv)
    from src.passada.passada_completa import juntar
    cfg = BASES[a.base]
    ids = pd.concat([pd.read_csv(l, usecols=["id"]) for l in cfg["listas"]])["id"]
    df = montar(a.base, pd.read_csv(cfg["elegiveis"], low_memory=False), juntar(a.base), ids)
    destino = f"/out/{a.base}/{NOME}"
    df.to_csv(destino, index=False)
    print(f"{a.base}: covariáveis de {len(df)} imagens de treino e validação em {destino} (nenhuma de teste)")
    for k, v in resumo(df).items():
        print(f"  {k}: {v}")
    if a.enviar:
        from src.cache.enviar_treino import _dataset, _tabela, enviar
        tab = _tabela(_dataset(), f"{a.base}_listas", f"{a.base}: listas de treino e validação e hashes do cache")
        log = f"/out/{a.base}/envio_covariaveis.txt"
        if os.path.exists(log):
            os.remove(log)                                   # reenviar substitui o arquivo anterior
        envio = f"/scratch/envio_cov_{a.base}"
        os.makedirs(envio, exist_ok=True)
        arq = os.path.join(envio, NOME)
        df.to_csv(arq, index=False)
        enviar(tab, [arq], log)
        print(f"RESULTADO: {NOME} enviado para {a.base}_listas no cache_treino")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
