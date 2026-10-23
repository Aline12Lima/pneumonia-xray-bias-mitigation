"""
Auditoria CheXpert Plus 01 (F3: passos 3.4 parcial, 3.8 e 3.9; base para D-018).

Roda DENTRO de um notebook do Redivis (workflow auditoria_chexpert_plus), criado a partir
da tabela df_chexpert_plus_240401. Nenhum dado por imagem sai do Redivis (D-024):
o relatório só tem números agregados.

Uso: cole este arquivo inteiro em uma célula do notebook e rode.
Saída: /out/auditoria_chexpert_plus_01.md (e o mesmo texto impresso na tela).
"""

import io
import re

import numpy as np
import pandas as pd
import redivis

SEED = 2026
TEST_FRAC = 0.20            # divisão 70/10/20 por paciente (D-004)
TARGET = 750                # portão G3
ADULT_MIN_AGE = 18
PRIMARY = "impression"      # seção usada nas quebras por estrato (decisão em aberto)
LABEL_TABLE = "chexpert_labels:pmec"
LABEL_FILES = {"report": "report_fixed.json",
               "impression": "impression_fixed.json",
               "findings": "findings_fixed.json"}
INDEX_TABLES = {"DICOM": [["DICOM_train", "dicom_train"], ["DICOM_valid", "dicom_valid"]],
                "PNG": [["PNG_train", "png_train"], ["PNG_valid", "png_valid"]]}
PROTOCOL_PREV = (6042, 223462)   # prevalência citada no protocolo (§4.3c)

KEY_RE = re.compile(r"(patient\d+/study\d+/view\d+_[a-z]+)")
R: list[str] = ["# CheXpert Plus — auditoria 01 (metadados, sem imagens)", ""]


# ----------------------------------------------------------------------------- utilidades
def md(d, index=True):
    try:
        return d.to_markdown(index=index)
    except ImportError:
        return "```\n" + d.to_string(index=index) + "\n```"


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "—"


def key_of(s):
    m = KEY_RE.search(str(s))
    return m.group(1) if m else None


def load_table(names):
    last = None
    for n in names:
        try:
            return redivis.table(n).to_pandas_dataframe(), n
        except Exception as e:  # nome diferente no workflow
            last = e
    raise last


def label_counts(s: pd.Series) -> dict:
    v = pd.to_numeric(s, errors="coerce")
    return {"positivo (1)": int((v == 1).sum()), "incerto (-1)": int((v == -1).sum()),
            "negativo (0)": int((v == 0).sum()), "não mencionado": int(v.isna().sum())}


# ----------------------------------------------------------------------------- tabela principal
df = redivis.table("_source_").to_pandas_dataframe()
def is_text(col):
    return pd.api.types.is_object_dtype(col) or pd.api.types.is_string_dtype(col)


for c in df.columns:
    if is_text(df[c]):
        df[c] = df[c].astype(object)
df["key"] = df["path_to_image"].map(key_of)
df["study"] = df["key"].str.rsplit("/", n=1).str[0]
df["view_n"] = pd.to_numeric(df["key"].str.extract(r"view(\d+)_")[0], errors="coerce")
R += ["## 1. Tabela principal",
      f"- Linhas: {len(df):,}; pacientes: {df['deid_patient_id'].nunique():,}; estudos: {df['study'].nunique():,}",
      f"- Chave de imagem extraída em {df['key'].notna().mean() * 100:.1f}% das linhas; "
      f"chaves repetidas: {df['key'].duplicated().sum():,}", ""]

# ----------------------------------------------------------------------------- rótulos
lab_tab = redivis.table(LABEL_TABLE)
R += ["## 2. Rótulos de pneumonia (CheXpert labeler por seção do laudo)", ""]
cover = []
for src, fn in LABEL_FILES.items():
    L = pd.read_json(io.BytesIO(lab_tab.file(fn).read()), lines=True)
    L["key"] = L["path_to_image"].map(key_of)
    dups = int(L["key"].duplicated().sum())
    L = L.drop_duplicates("key").set_index("key")
    df[f"pn_{src}"] = df["key"].map(L["Pneumonia"])
    in_file = df["key"].isin(L.index)
    pos_all = int((pd.to_numeric(df[f"pn_{src}"], errors="coerce") == 1).sum())
    cover.append({"fonte": src, "linhas no JSON": len(L), "repetidas": dups,
                  "imagens da tabela cobertas": f"{in_file.sum():,} ({pct(in_file.sum(), len(df))})",
                  "positivos (todas as imagens)": pos_all,
                  "prevalência (todas)": pct(pos_all, len(df))})
R += [md(pd.DataFrame(cover), index=False), "",
      f"- Protocolo (§4.3c) cita {PROTOCOL_PREV[0]:,} / {PROTOCOL_PREV[1]:,} = "
      f"{100 * PROTOCOL_PREV[0] / PROTOCOL_PREV[1]:.2f}%. A linha acima que reproduzir esse número é a fonte usada no protocolo.",
      ""]
if "section_impression" in df:
    has_imp = df["section_impression"].astype(str).str.strip().replace({"None": "", "nan": "", "<NA>": ""}).ne("")
    R += [f"- Laudos com seção de impressão preenchida: {pct(has_imp.sum(), len(df))}", ""]

# ----------------------------------------------------------------------------- índices de arquivos
R += ["## 3. Arquivos de imagem (índices DICOM e PNG)", ""]
for fmt, groups in INDEX_TABLES.items():
    keys, total_bytes, sizes = set(), 0, {}
    for names in groups:
        try:
            ix, used = load_table(names)
        except Exception as e:
            R.append(f"- {fmt}: não consegui ler {names[0]} ({type(e).__name__})")
            continue
        str_cols = [c for c in ix.columns if is_text(ix[c])]
        best, best_share = None, 0.0
        for c in str_cols:
            share = ix[c].head(2000).map(key_of).notna().mean()
            if share > best_share:
                best, best_share = c, share
        size_col = next((c for c in ix.columns if "size" in c.lower()), None)
        R.append(f"- {used}: {len(ix):,} arquivos; colunas {list(ix.columns)}; "
                 f"coluna com o caminho: {best} ({best_share * 100:.0f}% reconhecidas)")
        if best and best_share > 0.5:
            k = ix[best].map(key_of)
            keys |= set(k.dropna())
            if size_col:
                sizes.update(dict(zip(k, pd.to_numeric(ix[size_col], errors="coerce"))))
    if keys:
        df[f"has_{fmt}"] = df["key"].isin(keys)
        df[f"bytes_{fmt}"] = df["key"].map(sizes)
        R.append(f"- **{fmt}: {df[f'has_{fmt}'].sum():,} das {len(df):,} imagens da tabela têm arquivo "
                 f"({pct(df[f'has_{fmt}'].sum(), len(df))})**")
    else:
        R.append(f"- {fmt}: não foi possível casar os arquivos com a tabela; ver colunas acima")
R.append("")

# ----------------------------------------------------------------------------- recorte
flow = [("Todas as imagens", len(df))]
cur = df[df["frontal_lateral"] == "Frontal"]
flow.append(("Frontais", len(cur)))
cur = cur[cur["ap_pa"].isin(["AP", "PA"])]
flow.append(("Projeção AP ou PA", len(cur)))
cur = cur[pd.to_numeric(cur["age"], errors="coerce") >= ADULT_MIN_AGE]
flow.append((f"Idade ≥ {ADULT_MIN_AGE}", len(cur)))
cur = cur.sort_values(["study", "view_n", "key"]).drop_duplicates("study")
flow.append(("Uma frontal por estudo (provisório: menor índice de vista)", len(cur)))
if "has_DICOM" in cur:
    flow.append(("…das quais com arquivo DICOM", int(cur["has_DICOM"].sum())))
R += ["## 4. Fluxograma de exclusões (provisório)", md(pd.DataFrame(flow, columns=["etapa", "n"]), index=False), "",
      f"- Recorte: {len(cur):,} estudos de {cur['deid_patient_id'].nunique():,} pacientes", ""]

# ----------------------------------------------------------------------------- rótulos no recorte
tab = pd.DataFrame({src: label_counts(cur[f"pn_{src}"]) for src in LABEL_FILES}).T
tab["primária: positivos"] = tab["positivo (1)"]
tab["primária: avaliados"] = tab["positivo (1)"] + tab["negativo (0)"] + tab["não mencionado"]
tab["primária: prevalência %"] = (100 * tab["primária: positivos"] / tab["primária: avaliados"]).round(2)
tab["sensibilidade (incerto=1): positivos"] = tab["positivo (1)"] + tab["incerto (-1)"]
R += ["## 5. Pneumonia no recorte, por seção do laudo",
      "Política proposta na D-018: positivo = 1; negativo = 0 ou não mencionado; incerto fora da primária "
      "e positivo na sensibilidade.", md(tab), ""]

a, b = cur["pn_report"].fillna(9), cur["pn_impression"].fillna(9)
ct = pd.crosstab(a.map({1: "pos", 0: "neg", -1: "inc", 9: "n/m"}),
                 b.map({1: "pos", 0: "neg", -1: "inc", 9: "n/m"}))
ct.index.name, ct.columns.name = "laudo inteiro", "impressão"
R += ["### 5.1 Concordância laudo inteiro × impressão (recorte)", md(ct), ""]

# ----------------------------------------------------------------------------- quebras (fonte primária)
pn = pd.to_numeric(cur[f"pn_{PRIMARY}"], errors="coerce")
cur = cur.assign(pos=pn.eq(1), unc=pn.eq(-1))
ev = cur[~cur["unc"]]
age_band = pd.cut(pd.to_numeric(ev["age"], errors="coerce"), [17, 40, 60, 75, 90]).astype(str)
R += [f"## 6. Positivos por estrato (fonte: {PRIMARY}; sem incertos)", ""]
for title, col in [("projeção", ev["ap_pa"]), ("sexo", ev["sex"]), ("faixa etária", age_band),
                   ("partição oficial", ev["split"])] + \
                  ([("tem DICOM", ev["has_DICOM"])] if "has_DICOM" in ev else []):
    t = ev.groupby(col).agg(estudos=("pos", "size"), positivos=("pos", "sum"))
    t["prevalência_%"] = (100 * t["positivos"] / t["estudos"]).round(2)
    R += [f"### {title}", md(t), ""]

# ----------------------------------------------------------------------------- teste e tamanho
pos_by_patient = ev.groupby("deid_patient_id")["pos"].sum()
patients = pos_by_patient.index.to_numpy()
rng = np.random.default_rng(SEED)
sims = []
for _ in range(500):
    test = rng.choice(patients, size=int(round(TEST_FRAC * len(patients))), replace=False)
    sims.append(int(pos_by_patient.loc[test].sum()))
sims = np.array(sims)
total_pos = int(ev["pos"].sum())
frac_needed = min(1.0, TARGET / (TEST_FRAC * total_pos)) if total_pos else float("nan")
R += ["## 7. Positivos esperados no teste e tamanho do download",
      f"- Positivos no recorte (primária, {PRIMARY}): {total_pos:,}",
      f"- Positivos no teste (20% dos pacientes, 500 sorteios, semente {SEED}): "
      f"mediana {np.median(sims):.0f}, p5 {np.percentile(sims, 5):.0f}, p95 {np.percentile(sims, 95):.0f}",
      f"- G3 (≥ {TARGET} no teste): " + ("atingido" if np.percentile(sims, 5) >= TARGET else "NÃO garantido"),
      f"- Fração de pacientes do recorte necessária para {TARGET} positivos no teste: {frac_needed * 100:.0f}%"]
for fmt in ["DICOM", "PNG"]:
    col = f"bytes_{fmt}"
    if col in cur and cur[col].notna().any():
        gb = cur[col].sum() / 1e9
        R.append(f"- {fmt} do recorte inteiro: {gb:,.0f} GB "
                 f"(média {cur[col].mean() / 1e6:.1f} MB por imagem); "
                 f"para a fração necessária: cerca de {gb * frac_needed:,.0f} GB")
R.append("")

report = "\n".join(R)
with open("/out/auditoria_chexpert_plus_01.md", "w", encoding="utf-8") as f:
    f.write(report)
print(report)
print("\nRelatório salvo em /out/auditoria_chexpert_plus_01.md")
