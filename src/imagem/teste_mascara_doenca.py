"""Teste "máscara × doença": a máscara do pulmão deixa a opacidade de fora? (decisão aberta F3 › Todas › Máscara × doença)

Por que: o modelo de máscara tende a marcar o pulmão com ar e a deixar de fora o pulmão opacificado.
Se isso acontecer mais nos positivos, as medidas de qualidade "no pulmão" passam a refletir a doença
(eixo A, T1/T2, eixo C) e a atenção do modelo sobre a consolidação parece "fora do pulmão" (F10).

Onde roda: no notebook do workflow ``auditoria_rsna``, na configuração gratuita (CPU). Usa os arquivos
que a auditoria já guardou em ``/out/rsna/``: ``rsna_elegiveis.csv`` (25.403 imagens de adultos do treino
do Kaggle) e o arquivo de anotações adjudicadas do MD.ai, com as caixas de opacidade.

Desenho (fixado antes de rodar)
-------------------------------
- Amostra: entre as elegíveis (adultos de 18 a 100 anos, treino do Kaggle, sem "Exclude"),
  250 imagens por célula projeção × rótulo (AP-positivo, AP-negativo, PA-positivo, PA-negativo),
  semente 2026. Estratificar pela projeção evita confundir "doença" com "AP", que já tem pulmão menor.
- Para cada imagem: máscara pelo procedimento oficial (``src/imagem/mascara.py``), área da máscara
  e razão entre os pulmões.
- Nos positivos: **cobertura das caixas** = fração da área das caixas de opacidade (rótulo final,
  grupo "Calculated") que cai dentro da máscara.
- **Referência transplantada:** todas as imagens do RSNA têm 1024 × 1024. Cada positivo é pareado,
  por sorteio (semente 2026), com um negativo da mesma projeção; as caixas do positivo são colocadas
  na máscara do negativo. Isso mede quanto a máscara cobriria da mesma região num pulmão sem opacidade,
  e corrige o fato de as caixas serem retângulos que passam da borda do pulmão.

Regra de decisão (pré-fixada)
-----------------------------
A máscara é considerada **dependente da doença** se qualquer uma valer:

1. A mediana de (cobertura no positivo ÷ cobertura da mesma caixa no negativo pareado) fica abaixo
   de 0,90, ou seja, a máscara perde mais de 10% da opacidade em relação a um pulmão sem doença.
2. Em alguma projeção, a área média nos positivos é pelo menos 10% menor que nos negativos,
   com o intervalo de 95% (bootstrap por paciente) da diferença excluindo zero.
3. Em alguma projeção, a fração de imagens com razão entre pulmões abaixo de 0,35 é, nos positivos,
   pelo menos o dobro da dos negativos e pelo menos 5 pontos percentuais maior.

Se for dependente: repetir com o lungVAE na mesma amostra e escolher o modelo com maior cobertura,
desde que ele também passe na inspeção de 100 imagens por base.

Análise descritiva (não decide): brilho e contraste calculados com a máscara e com a máscara
preenchida (envoltória convexa de cada pulmão, que devolve a opacidade deixada de fora). Mede quanto
a diferença padronizada entre positivos e negativos muda de uma máscara para a outra.

Uso (no notebook, depois do ``git clone`` do repositório):
    python -m src.imagem.teste_mascara_doenca
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

LIMITE_COBERTURA_RELATIVA = 0.90
LIMITE_AREA_RELATIVA = 0.10
LIMITE_RAZAO = 0.35
COLUNAS_NECESSARIAS = ("SOPInstanceUID", "nih_paciente", "positivo", "proj", "idade")
ID_CANDIDATOS = ("SOPInstanceUID", "subset_img_id", "img_id", "instanceId")


def _bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["true", "1"])


# ------------------------------------------------------------------ amostra, caixas e arquivos
def amostrar(cab: pd.DataFrame, n_por_celula: int = 250, semente: int = 2026) -> pd.DataFrame:
    """250 imagens por célula projeção × rótulo, entre adultos do treino do Kaggle sem "Exclude"."""
    faltam = [c for c in COLUNAS_NECESSARIAS if c not in cab.columns]
    if faltam:
        raise KeyError(f"faltam colunas {faltam} na tabela; colunas disponíveis: {list(cab.columns)}")
    base = cab.drop_duplicates("SOPInstanceUID").copy()
    base["positivo"] = _bool(base["positivo"])
    filtro = base["proj"].isin(["AP", "PA"]) & base["idade"].between(18, 100)
    for col in ("exclude", "kaggle_teste"):
        if col in base.columns:
            filtro &= ~_bool(base[col])
    base = base[filtro]
    partes = []
    for (proj, pos), g in base.groupby(["proj", "positivo"]):
        if len(g) < n_por_celula:
            raise ValueError(f"célula {proj}/{'positivo' if pos else 'negativo'} tem só {len(g)} imagens")
        partes.append(g.sample(n_por_celula, random_state=semente))
    return pd.concat(partes).reset_index(drop=True)


def parear(amostra: pd.DataFrame, semente: int = 2026) -> dict[str, str]:
    """Sorteia, dentro de cada projeção, um negativo diferente para cada positivo (SOPInstanceUID → SOPInstanceUID)."""
    rng = np.random.default_rng(semente)
    pares = {}
    for _, g in amostra.groupby("proj"):
        pos = g.loc[g["positivo"], "SOPInstanceUID"].tolist()
        neg = g.loc[~g["positivo"], "SOPInstanceUID"].tolist()
        neg = list(rng.permutation(neg))
        if len(neg) < len(pos):
            raise ValueError("menos negativos que positivos para parear")
        pares.update(dict(zip(pos, neg)))
    return pares


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


def mapear_arquivos(nomes: list[str], amostra: pd.DataFrame) -> tuple[str, dict[str, str]]:
    """Descobre qual coluna da tabela dá o nome do arquivo DICOM. Devolve (coluna, SOPInstanceUID → nome)."""
    # Path().stem cortaria o UID "1.2.840...": tirar só as extensões conhecidas
    def raiz(v: str) -> str:
        v = Path(str(v)).name
        for ext in (".dcm", ".dicom", ".png"):
            if v.lower().endswith(ext):
                return v[: -len(ext)]
        return v

    por_raiz = {raiz(n): n for n in nomes}
    for col in ID_CANDIDATOS:
        if col not in amostra.columns:
            continue
        chaves = amostra[col].map(raiz)
        achados = chaves.map(por_raiz)
        if achados.notna().mean() > 0.99:
            return col, dict(zip(amostra["SOPInstanceUID"].astype(str), achados))
    raise KeyError(f"nenhuma coluna de {ID_CANDIDATOS} bate com os nomes dos arquivos; exemplos: {nomes[:3]}")


# ------------------------------------------------------------------ geometria
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


def preencher_pulmoes(pulmoes: np.ndarray) -> np.ndarray:
    """União das envoltórias convexas de cada pulmão: devolve a opacidade que a máscara deixou de fora."""
    saida = np.zeros(pulmoes.shape[1:], dtype=np.uint8)
    for m in pulmoes:
        cont, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cont:
            cv2.fillPoly(saida, [cv2.convexHull(np.vstack(cont))], 1)
    return saida


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
    reps = [dif(pd.concat([por_pac[p] for p in rng.choice(pacientes, size=len(pacientes), replace=True)]))
            for _ in range(n_boot)]
    lo, hi = np.nanpercentile(reps, [2.5, 97.5])
    return float(estim), float(lo), float(hi)


def _dif_padronizada(g: pd.DataFrame, col: str) -> float:
    a, b = g.loc[g["positivo"], col].dropna(), g.loc[~g["positivo"], col].dropna()
    dp = np.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / dp) if dp > 0 else float("nan")


def resumir(res: pd.DataFrame, n_boot: int = 2000, semente: int = 2026) -> dict:
    """Aplica a regra de decisão do cabeçalho e devolve números e veredito."""
    saida: dict = {"por_projecao": {}}
    criterios = {"1_cobertura_relativa_baixa": False, "2_area_menor_nos_positivos": False,
                 "3_pulmao_encolhido": False}

    pos = res[res["positivo"]]
    rel = (pos["cobertura_caixas"] / pos["cobertura_transplantada"]).replace([np.inf, -np.inf], np.nan)
    saida["cobertura_mediana_positivos"] = float(pos["cobertura_caixas"].median())
    saida["cobertura_mediana_transplantada"] = float(pos["cobertura_transplantada"].median())
    saida["cobertura_relativa_mediana"] = float(rel.median())
    saida["positivos_com_cobertura_relativa"] = int(rel.notna().sum())
    saida["fracao_positivos_cobertura_abaixo_0_5"] = float((pos["cobertura_caixas"] < 0.5).mean())
    criterios["1_cobertura_relativa_baixa"] = bool(rel.median() < LIMITE_COBERTURA_RELATIVA)

    for proj, g in res.groupby("proj"):
        estim, lo, hi = _boot_dif_area(g, n_boot, semente)
        raz_pos = float((g.loc[g["positivo"], "razao_pulmoes"] < LIMITE_RAZAO).mean())
        raz_neg = float((g.loc[~g["positivo"], "razao_pulmoes"] < LIMITE_RAZAO).mean())
        gp = g[g["positivo"]]
        info = {
            "area_media_positivos": float(gp["area"].mean()),
            "area_media_negativos": float(g.loc[~g["positivo"], "area"].mean()),
            "diferenca_relativa_area": estim, "ic95": [lo, hi],
            "fracao_razao_abaixo_0_35_positivos": raz_pos,
            "fracao_razao_abaixo_0_35_negativos": raz_neg,
            "cobertura_relativa_mediana": float((gp["cobertura_caixas"] / gp["cobertura_transplantada"])
                                                .replace([np.inf, -np.inf], np.nan).median()),
        }
        for m in ("brilho", "contraste"):
            if f"q_{m}_mascara" in g.columns:
                d_masc, d_cheia = _dif_padronizada(g, f"q_{m}_mascara"), _dif_padronizada(g, f"q_{m}_preenchida")
                info[f"descritivo_{m}"] = {"dif_padronizada_mascara": d_masc,
                                           "dif_padronizada_preenchida": d_cheia,
                                           "mudanca": d_cheia - d_masc}
        saida["por_projecao"][proj] = info
        if estim <= -LIMITE_AREA_RELATIVA and hi < 0:
            criterios["2_area_menor_nos_positivos"] = True
        if raz_pos >= 2 * raz_neg and raz_pos - raz_neg >= 0.05:
            criterios["3_pulmao_encolhido"] = True

    saida["criterios"] = criterios
    saida["dependente_da_doenca"] = any(criterios.values())
    return saida


# ------------------------------------------------------------------ execução no Redivis
def baixar_arquivo(arquivo, destino: str) -> None:
    """Baixa um arquivo do Redivis sem barra de progresso (versões antigas do pacote não têm a opção)."""
    try:
        arquivo.download(destino, progress=False)
    except TypeError:
        arquivo.download(destino)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tabela", default="/out/rsna/rsna_elegiveis.csv")
    ap.add_argument("--anotacoes", default="/out/rsna/pneumonia-challenge-annotations-adjudicated-kaggle_2018.json")
    ap.add_argument("--dicom", default="dicom:sjtd", help="tabela de arquivos DICOM do RSNA no Redivis")
    ap.add_argument("--n", type=int, default=250, help="imagens por célula projeção × rótulo")
    ap.add_argument("--saida", default="/out/rsna/teste_mascara_doenca")
    ap.add_argument("--temp", default="/scratch/teste_mascara")
    a = ap.parse_args(argv)

    import pydicom
    import redivis
    from src.imagem.conversao import para_8bits
    from src.imagem.mascara import razao_pulmoes, segmentar_pulmoes
    from src.qualidade.medidas import medidas_qualidade

    os.makedirs(a.saida, exist_ok=True)
    amostra = amostrar(pd.read_csv(a.tabela), a.n)
    pares = parear(amostra)
    caixas = caixas_por_imagem(a.anotacoes)
    print(f"Amostra: {len(amostra)} imagens | {amostra.groupby(['proj', 'positivo']).size().to_dict()}")
    sem_caixa = [s for s in amostra.loc[amostra["positivo"], "SOPInstanceUID"] if s not in caixas]
    print(f"Positivos sem caixa no arquivo de anotações: {len(sem_caixa)}")

    tabela = redivis.table(a.dicom)
    # Usar os próprios objetos da listagem: tabela.file(nome) falha no RSNA, porque o caminho
    # guardado no Redivis não é igual ao nome que a listagem mostra.
    arquivos = {f.name: f for f in tabela.list_files()}
    nomes = list(arquivos)
    col, nome_de = mapear_arquivos(nomes, amostra)
    print(f"Arquivos na tabela: {len(nomes)} | nome do arquivo vem da coluna '{col}'")

    os.makedirs(a.temp, exist_ok=True)

    def baixar(sop):
        destino = os.path.join(a.temp, sop)
        for tentativa in range(4):
            achados = glob.glob(f"{destino}/**/*.dcm", recursive=True)
            if achados:
                return achados[0]
            os.makedirs(destino, exist_ok=True)
            try:
                baixar_arquivo(arquivos[nome_de[sop]], destino)
            except Exception as erro:  # rede instável: tenta de novo com espera crescente
                if tentativa == 3:
                    raise RuntimeError(f"falhou ao baixar {nome_de[sop]}") from erro
                time.sleep(2 ** tentativa)
        return glob.glob(f"{destino}/**/*.dcm", recursive=True)[0]

    with ThreadPoolExecutor(8) as ex:
        caminhos = dict(zip(amostra["SOPInstanceUID"], ex.map(baixar, amostra["SOPInstanceUID"])))
    print("Download concluído; segmentando...")

    linhas, rois = [], {}
    for i, row in enumerate(amostra.itertuples(), 1):
        ds = pydicom.dcmread(caminhos[row.SOPInstanceUID])
        img8 = para_8bits(ds.pixel_array, monochrome1=ds.PhotometricInterpretation == "MONOCHROME1")
        pulmoes = segmentar_pulmoes(img8)
        roi = pulmoes.max(axis=0)
        cheia = preencher_pulmoes(pulmoes)
        rois[row.SOPInstanceUID] = np.packbits(roi.astype(bool))
        qm, qc = medidas_qualidade(img8, roi), medidas_qualidade(img8, cheia)
        linhas.append({"SOPInstanceUID": row.SOPInstanceUID, "nih_paciente": row.nih_paciente,
                       "proj": row.proj, "positivo": bool(row.positivo), "forma": img8.shape,
                       "area": float(roi.mean()), "area_preenchida": float(cheia.mean()),
                       "razao_pulmoes": razao_pulmoes(pulmoes),
                       "n_caixas": len(caixas.get(row.SOPInstanceUID, [])),
                       "cobertura_caixas": cobertura(roi, caixas.get(row.SOPInstanceUID, [])),
                       "q_brilho_mascara": qm["q_brilho"], "q_contraste_mascara": qm["q_contraste"],
                       "q_brilho_preenchida": qc["q_brilho"], "q_contraste_preenchida": qc["q_contraste"]})
        if i % 100 == 0:
            print(f"  {i}/{len(amostra)}")
    res = pd.DataFrame(linhas)

    formas = {s: f for s, f in zip(res["SOPInstanceUID"], res["forma"])}
    transpl = []
    for sop in res["SOPInstanceUID"]:
        neg = pares.get(sop)
        if neg is None or formas[sop] != formas[neg]:
            transpl.append(np.nan)
            continue
        roi_neg = np.unpackbits(rois[neg])[: np.prod(formas[neg])].reshape(formas[neg])
        transpl.append(cobertura(roi_neg, caixas.get(sop, [])))
    res["negativo_pareado"] = res["SOPInstanceUID"].map(pares)
    res["cobertura_transplantada"] = transpl
    res.drop(columns="forma").to_csv(f"{a.saida}/resultado_por_imagem.csv", index=False)

    resumo = resumir(res)
    resumo["coluna_do_nome_do_arquivo"] = col
    json.dump(resumo, open(f"{a.saida}/resumo.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps(resumo, indent=2, ensure_ascii=False))
    shutil.rmtree(a.temp, ignore_errors=True)
    print(f"\nVeredito: a máscara {'DEPENDE' if resumo['dependente_da_doenca'] else 'NÃO depende'} da doença "
          f"pela regra pré-fixada | saída em {a.saida} | arquivos temporários apagados")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
