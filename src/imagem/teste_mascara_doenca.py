"""Teste "máscara × doença": a máscara do pulmão deixa a opacidade de fora? (decisão aberta F3 › Todas › Máscara × doença)

Por que: o modelo de máscara tende a marcar o pulmão com ar e a deixar de fora o pulmão opacificado.
Se isso acontecer mais nos positivos, as medidas de qualidade "no pulmão" passam a refletir a doença
(eixo A, T1/T2, eixo C) e a atenção do modelo sobre a consolidação parece "fora do pulmão" (F10).

Onde roda: no notebook do workflow ``auditoria_rsna`` (tem internet para o ``git clone`` e guarda em
``/out/rsna/`` o arquivo de anotações e a tabela de cabeçalhos do treino).

Desenho (fixado antes de rodar)
-------------------------------
- Amostra: treino do Kaggle, adultos de 18 a 100 anos, 250 imagens por célula projeção × rótulo
  (AP-positivo, AP-negativo, PA-positivo, PA-negativo), semente 2026. Estratificar pela projeção
  evita confundir "doença" com "AP", que já tem máscaras menores.
- Arquivos: o nome do DICOM no Kaggle é o ``patientId`` (que identifica a imagem); as caixas do arquivo
  de anotações são indexadas pelo ``SOPInstanceUID``, lido do próprio cabeçalho do DICOM.
- Para cada imagem: máscara pelo procedimento oficial (``src/imagem/mascara.py``), área da máscara,
  razão entre os pulmões e, nos positivos, a **cobertura das caixas de opacidade** desenhadas pelos
  radiologistas (fração da área das caixas que cai dentro da máscara).

Regra de decisão (proposta; confirmar antes de rodar)
-----------------------------------------------------
A máscara é considerada **dependente da doença** se qualquer uma valer:

1. Cobertura mediana das caixas nos positivos abaixo de 0,80.
2. Em alguma projeção, a área média nos positivos é pelo menos 10% menor que nos negativos,
   com o intervalo de 95% (bootstrap por paciente) da diferença excluindo zero.
3. A fração de imagens com razão entre pulmões abaixo de 0,35 é, nos positivos, pelo menos o dobro
   da dos negativos e pelo menos 5 pontos percentuais maior.

Se for dependente: repetir com o lungVAE na mesma amostra e escolher o modelo com maior cobertura,
desde que ele também passe na inspeção de 100 imagens por base.

Uso (no notebook, depois do ``git clone`` do repositório):
    python -m src.imagem.teste_mascara_doenca
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ("Lung Opacity", "No Lung Opacity / Not Normal", "Normal")
LIMITE_COBERTURA = 0.80
LIMITE_AREA_RELATIVA = 0.10
LIMITE_RAZAO = 0.35


# ------------------------------------------------------------------ amostra e caixas
COLUNAS_NECESSARIAS = ("patientId", "classe_final", "proj", "idade")


def amostrar(cab: pd.DataFrame, n_por_celula: int = 250, semente: int = 2026) -> pd.DataFrame:
    """250 imagens por célula projeção × rótulo, entre adultos do treino do Kaggle com rótulo final."""
    faltam = [c for c in COLUNAS_NECESSARIAS if c not in cab.columns]
    if faltam:
        raise KeyError(f"faltam colunas {faltam} na tabela de cabeçalhos; colunas disponíveis: {list(cab.columns)}")
    cab = cab.drop_duplicates("patientId")
    if "nih_paciente" not in cab.columns:
        print("Aviso: sem 'nih_paciente'; o bootstrap será por imagem (patientId).")
        cab = cab.assign(nih_paciente=cab["patientId"])
    base = cab[cab["classe_final"].isin(CLASSES) & cab["proj"].isin(["AP", "PA"])
               & cab["idade"].between(18, 100)].copy()
    if "kaggle_teste" in base.columns:
        base = base[~base["kaggle_teste"].astype(str).str.lower().eq("true")]
    base["positivo"] = base["classe_final"].eq("Lung Opacity")
    partes = []
    for (proj, pos), g in base.groupby(["proj", "positivo"]):
        if len(g) < n_por_celula:
            raise ValueError(f"célula {proj}/{'positivo' if pos else 'negativo'} tem só {len(g)} imagens")
        partes.append(g.sample(n_por_celula, random_state=semente))
    return pd.concat(partes).reset_index(drop=True)


def caixas_por_imagem(caminho_json: str | Path) -> dict[str, list[tuple[float, float, float, float]]]:
    """Caixas de opacidade do rótulo final (grupo "Calculated", classe "Lung Opacity"), por SOPInstanceUID."""
    anot = json.load(open(caminho_json, encoding="utf-8"))
    grupo = next(g for g in anot["labelGroups"]
                 if any(l["name"].strip() == "Lung Opacity" for l in g["labels"]))
    ids = {l["id"] for l in grupo["labels"] if l["name"].strip() == "Lung Opacity"}
    caixas: dict[str, list] = {}
    for ds in anot["datasets"]:
        for a in ds.get("annotations", []):
            d = a.get("data") or {}
            if a.get("labelId") in ids and {"x", "y", "width", "height"} <= set(d):
                caixas.setdefault(a["SOPInstanceUID"], []).append(
                    (float(d["x"]), float(d["y"]), float(d["width"]), float(d["height"])))
    return caixas


def cobertura(roi: np.ndarray, caixas: list[tuple[float, float, float, float]]) -> float:
    """Fração da área das caixas (unidas) que fica dentro da máscara. NaN se não houver caixa."""
    if not caixas:
        return float("nan")
    alvo = np.zeros(roi.shape, dtype=bool)
    h, w = roi.shape
    for x, y, cw, ch in caixas:
        x0, y0 = max(int(round(x)), 0), max(int(round(y)), 0)
        x1, y1 = min(int(round(x + cw)), w), min(int(round(y + ch)), h)
        alvo[y0:y1, x0:x1] = True
    n = alvo.sum()
    return float((roi.astype(bool) & alvo).sum() / n) if n else float("nan")


# ------------------------------------------------------------------ resumo e regra
def _boot_dif_area(g: pd.DataFrame, n_boot: int, semente: int) -> tuple[float, float, float]:
    """Diferença relativa de área (positivos − negativos) / negativos, com IC 95% por bootstrap de paciente."""
    rng = np.random.default_rng(semente)
    pacientes = g["nih_paciente"].unique()
    por_pac = {p: d for p, d in g.groupby("nih_paciente")}

    def dif(df):
        a_pos, a_neg = df.loc[df["positivo"], "area"].mean(), df.loc[~df["positivo"], "area"].mean()
        return (a_pos - a_neg) / a_neg

    estim = dif(g)
    reps = []
    for _ in range(n_boot):
        amostra = rng.choice(pacientes, size=len(pacientes), replace=True)
        reps.append(dif(pd.concat([por_pac[p] for p in amostra])))
    lo, hi = np.nanpercentile(reps, [2.5, 97.5])
    return float(estim), float(lo), float(hi)


def resumir(res: pd.DataFrame, n_boot: int = 2000, semente: int = 2026) -> dict:
    """Aplica a regra de decisão do cabeçalho e devolve números e veredito."""
    saida: dict = {"por_projecao": {}}
    criterios = {}

    pos = res[res["positivo"]]
    cob_med = float(pos["cobertura_caixas"].median())
    saida["cobertura_mediana_positivos"] = cob_med
    saida["fracao_positivos_cobertura_abaixo_0_5"] = float((pos["cobertura_caixas"] < 0.5).mean())
    criterios["1_cobertura_baixa"] = cob_med < LIMITE_COBERTURA

    area_dependente = False
    for proj, g in res.groupby("proj"):
        estim, lo, hi = _boot_dif_area(g, n_boot, semente)
        raz_pos = float((g.loc[g["positivo"], "razao_pulmoes"] < LIMITE_RAZAO).mean())
        raz_neg = float((g.loc[~g["positivo"], "razao_pulmoes"] < LIMITE_RAZAO).mean())
        saida["por_projecao"][proj] = {
            "area_media_positivos": float(g.loc[g["positivo"], "area"].mean()),
            "area_media_negativos": float(g.loc[~g["positivo"], "area"].mean()),
            "diferenca_relativa_area": estim, "ic95": [lo, hi],
            "fracao_razao_abaixo_0_35_positivos": raz_pos,
            "fracao_razao_abaixo_0_35_negativos": raz_neg,
            "cobertura_mediana_positivos": float(g.loc[g["positivo"], "cobertura_caixas"].median()),
        }
        if estim <= -LIMITE_AREA_RELATIVA and hi < 0:
            area_dependente = True
        criterios.setdefault("3_pulmao_encolhido", False)
        if raz_pos >= 2 * raz_neg and raz_pos - raz_neg >= 0.05:
            criterios["3_pulmao_encolhido"] = True
    criterios["2_area_menor_nos_positivos"] = area_dependente

    saida["criterios"] = criterios
    saida["dependente_da_doenca"] = any(criterios.values())
    return saida


# ------------------------------------------------------------------ execução no Redivis
def _ler_dicom(pid: str, tabela, pasta: str):
    import pydicom
    locais = glob.glob(f"/scratch/rsna_dcm/**/{pid}.dcm", recursive=True)
    if locais:
        return pydicom.dcmread(locais[0])
    destino = os.path.join(pasta, pid)
    os.makedirs(destino, exist_ok=True)
    tabela.file(f"{pid}.dcm").download(destino)
    arq = glob.glob(f"{destino}/**/*.dcm", recursive=True)[0]
    ds = pydicom.dcmread(arq)
    os.remove(arq)
    return ds


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cabecalhos", default="/out/rsna/rsna_treino_cabecalhos.csv")
    ap.add_argument("--anotacoes", default="/out/rsna/pneumonia-challenge-annotations-adjudicated-kaggle_2018.json")
    ap.add_argument("--dicom", default="dicom:sjtd", help="tabela de arquivos DICOM do RSNA no Redivis")
    ap.add_argument("--n", type=int, default=250, help="imagens por célula projeção × rótulo")
    ap.add_argument("--saida", default="/out/rsna/teste_mascara_doenca")
    a = ap.parse_args(argv)

    import redivis
    from src.imagem.conversao import para_8bits
    from src.imagem.mascara import razao_pulmoes, segmentar_pulmoes

    os.makedirs(a.saida, exist_ok=True)
    cab = pd.read_csv(a.cabecalhos)
    amostra = amostrar(cab, a.n)
    caixas = caixas_por_imagem(a.anotacoes)
    tabela = redivis.table(a.dicom)
    print(f"Amostra: {len(amostra)} imagens | {amostra.groupby(['proj', 'positivo']).size().to_dict()}")

    with ThreadPoolExecutor(8) as ex:
        dicoms = list(ex.map(lambda s: _ler_dicom(s, tabela, "/scratch/teste_mascara"), amostra["patientId"]))

    linhas = []
    for row, ds in zip(amostra.itertuples(), dicoms):
        img8 = para_8bits(ds.pixel_array, monochrome1=ds.PhotometricInterpretation == "MONOCHROME1")
        pulmoes = segmentar_pulmoes(img8)
        roi = pulmoes.max(axis=0)
        cx = caixas.get(str(ds.SOPInstanceUID), [])
        if row.positivo and not cx:
            print(f"Aviso: positivo sem caixa no arquivo de anotações: {row.patientId}")
        linhas.append({"patientId": row.patientId, "nih_paciente": row.nih_paciente,
                       "proj": row.proj, "positivo": bool(row.positivo),
                       "area": float(roi.mean()), "razao_pulmoes": razao_pulmoes(pulmoes),
                       "n_caixas": len(cx), "cobertura_caixas": cobertura(roi, cx)})
    res = pd.DataFrame(linhas)
    res.to_csv(f"{a.saida}/resultado_por_imagem.csv", index=False)

    resumo = resumir(res)
    json.dump(resumo, open(f"{a.saida}/resumo.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps(resumo, indent=2, ensure_ascii=False))
    print(f"\nVeredito: a máscara {'DEPENDE' if resumo['dependente_da_doenca'] else 'NÃO depende'} da doença "
          f"pela regra pré-fixada | saída em {a.saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
