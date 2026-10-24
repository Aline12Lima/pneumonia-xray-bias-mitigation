"""
Auditoria preliminar do PadChest (F3: passos 3.1a, 3.7, 3.9 parcial; base para D-025).

Não precisa de nenhuma imagem. Usa:
  data/raw/padchest/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz
  data/raw/padchest/Verify_Zips_ImageCounts.csv.xlsx
  data/raw/padchest/unzip-l/*.zip.unzip-l.txt

Gera:
  results/tables/padchest/relatorio_metadados.md   (relatório para colar no chat)
  results/tables/padchest/*.csv                          (tabelas de apoio)

Uso (na raiz do repositório):
  python src/data/padchest_audit.py
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2026
TARGET_POSITIVES = 750          # portão G3
ADULT_MIN_AGE = 18
FRONTAL = ["PA", "AP", "AP_horizontal"]   # AP_horizontal = AP em decúbito; reportado à parte
N_PERMUTATIONS = 1000

LISTING_ROW = re.compile(r"^\s*(\d+)\s+\S+\s+\S+\s+(.+?)\s*$")
CANDIDATE_ZIPS = list(range(0, 12)) + [54]   # pacotes 100% rotulados por médico (auditoria de 17/10)
PHYSICAL = ["XRayTubeCurrent_DICOM", "Exposure_DICOM", "ExposureInuAs_DICOM",
            "ExposureTime", "RelativeXRayExposure_DICOM", "WindowCenter_DICOM", "WindowWidth_DICOM"]


# ----------------------------------------------------------------------------- utilidades
def parse_labels(value) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        items = ast.literal_eval(value)
        if isinstance(items, str):
            items = [items]
    except (ValueError, SyntaxError):
        items = value.strip("[]").split(",")
    return [str(i).strip().strip("'\"").strip().lower() for i in items if str(i).strip()]


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    ct = pd.crosstab(x, y)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return float("nan")
    obs = ct.to_numpy(dtype=float)
    n = obs.sum()
    expected = np.outer(obs.sum(1), obs.sum(0)) / n
    chi2 = ((obs - expected) ** 2 / expected).sum()
    return float(np.sqrt(chi2 / (n * (min(obs.shape) - 1))))


def pct(a, b) -> str:
    return f"{100 * a / b:.1f}%" if b else "—"


def md_table(df: pd.DataFrame, index: bool = True) -> str:
    try:
        return df.to_markdown(index=index)
    except ImportError:  # tabulate ausente
        return "```\n" + df.to_string(index=index) + "\n```"


# ----------------------------------------------------------------------------- leitura
def read_listings(folder: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(folder.glob("*.zip.unzip-l.txt"), key=lambda p: int(p.name.split(".")[0])):
        zip_id = int(f.name.split(".")[0])
        for line in f.read_text(errors="replace").splitlines():
            m = LISTING_ROW.match(line)
            if not m:
                continue
            size, name = int(m.group(1)), m.group(2)
            if name.lower().endswith(".png"):
                rows.append((zip_id, Path(name).name, size))
    return pd.DataFrame(rows, columns=["zip", "ImageID", "bytes"])


def _profile(d: pd.DataFrame) -> dict:
    """Perfil de um conjunto de estudos, para comparar candidato × restante × todos."""
    n = len(d)
    out = {"estudos": n, "pacientes": d["PatientID"].nunique(),
           "positivos": int(d["pneumonia"].sum()),
           "prevalência_%": 100 * d["pneumonia"].mean() if n else np.nan}
    for p in FRONTAL:
        out[f"%{p}"] = 100 * d["Projection"].eq(p).mean() if n else np.nan
    if "Manufacturer_DICOM" in d:
        out["%Philips"] = 100 * d["Manufacturer_DICOM"].astype(str).str.contains("Philips").mean() if n else np.nan
    if "PatientSex_DICOM" in d:
        out["%feminino"] = 100 * d["PatientSex_DICOM"].eq("F").mean() if n else np.nan
    if "age" in d:
        out["idade_média"] = d["age"].mean()
    if "study_year" in d:
        for lo, hi in [(2007, 2011), (2012, 2014), (2015, 2017)]:
            out[f"%{lo}–{hi}"] = 100 * d["study_year"].between(lo, hi).mean() if n else np.nan
    if "MethodLabel" in d:
        out["%rótulo médico"] = 100 * d["MethodLabel"].astype(str).str.lower().eq("physician").mean() if n else np.nan
    return out


def candidate_subset(cur: pd.DataFrame, zip_bytes: pd.Series, sel: list[int]) -> list[str]:
    """Seção 12: subconjunto candidato da D-025 (pacotes selecionados), sem rótulo 'exclude'."""
    R = [f"## 12. Subconjunto candidato da D-025 — pacotes {sel}", ""]
    zips = cur["zip"].astype(int)
    in_sel = zips.isin(sel)
    sub_all = cur[in_sel]
    sub = sub_all[~sub_all["exclude_label"]]
    rest = cur[~in_sel & ~cur["exclude_label"]]
    everything = cur[~cur["exclude_label"]]

    exact = int(sub["pneumonia"].sum())
    with_atyp = int(sub["any_pneumonia_term"].sum())
    atyp_only = int((sub["any_pneumonia_term"] & ~sub["pneumonia"]).sum())
    gb = float(zip_bytes.reindex(sel).sum())

    R += ["### 12.1 Contagens",
          f"- Download: {len(sel)} pacotes, {gb:,.1f} GB",
          f"- Estudos no recorte: {len(sub_all):,}; após remover rótulo 'exclude': {len(sub):,} "
          f"(removidos {len(sub_all) - len(sub):,})",
          f"- Pacientes: {sub['PatientID'].nunique():,}",
          f"- Positivos, só 'pneumonia': **{exact:,}** ({pct(exact, len(sub))})",
          f"- Positivos, incluindo 'atypical pneumonia': **{with_atyp:,}** (só atípica: {atyp_only})",
          f"- Portão G3 (≥ {TARGET_POSITIVES}): "
          + ("atingido" if exact >= TARGET_POSITIVES else
             ("atingido só contando atípica" if with_atyp >= TARGET_POSITIVES else
              "NÃO atingido — avaliar a regra de 500 com 2 faixas")), ""]

    if "MethodLabel" in cur:
        phys = cur["MethodLabel"].astype(str).str.lower().eq("physician") & ~cur["exclude_label"]
        out_phys = phys & ~in_sel
        R += [f"- Estudos rotulados por médico fora dos pacotes escolhidos: {int(out_phys.sum()):,} "
              f"(positivos: {int(cur.loc[out_phys, 'pneumonia'].sum())}); por pacote: "
              + (", ".join(f"{int(z)}: {n}" for z, n in cur.loc[out_phys].groupby('zip').size().items()) or "nenhum"),
              ""]

    prof = pd.DataFrame({"candidato": _profile(sub), "restante": _profile(rest),
                         "todos": _profile(everything)}).round(2)
    prof.to_csv(Path("results/tables/padchest") / "candidate_vs_rest.csv")
    R += ["### 12.2 Representatividade: candidato × restante × todos (sem 'exclude')", md_table(prof), ""]

    for col, title in [("Projection", "projeção"), ("Manufacturer_DICOM", "fabricante"),
                       ("study_year", "ano")]:
        if col in sub:
            t = sub.groupby(col).agg(estudos=("StudyID", "size"), positivos=("pneumonia", "sum"),
                                     com_atipica=("any_pneumonia_term", "sum"))
            t["prevalência_%"] = (100 * t["positivos"] / t["estudos"]).round(2)
            R += [f"### 12.3 Positivos no candidato por {title}", md_table(t), ""]

    # parâmetros físicos: preenchido e não zero, por fabricante (eixo A físico)
    phys_cols = [c for c in PHYSICAL if c in sub]
    if phys_cols and "Manufacturer_DICOM" in sub:
        rows = {}
        for man, g in sub.groupby("Manufacturer_DICOM"):
            for c in phys_cols:
                v = pd.to_numeric(g[c], errors="coerce")
                rows[(man, c)] = {"preenchido_%": 100 * v.notna().mean(),
                                  "zero_%": 100 * v.eq(0).mean(),
                                  "útil_%": 100 * (v.notna() & v.ne(0)).mean(),
                                  "mediana": v[v.ne(0)].median()}
        tab = pd.DataFrame(rows).T.round(1)
        tab.index.names = ["fabricante", "campo"]
        tab.to_csv(Path("results/tables/padchest") / "candidate_physical_fields.csv")
        R += ["### 12.4 Parâmetros físicos no candidato, por fabricante (zeros tratados como ausentes)",
              md_table(tab), ""]
    return R


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/padchest")
    ap.add_argument("--csv", default="PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz")
    ap.add_argument("--report", default="results/tables/padchest/relatorio_metadados.md")
    ap.add_argument("--tables", default="results/tables/padchest")
    ap.add_argument("--zips", default=",".join(map(str, CANDIDATE_ZIPS)),
                    help="pacotes do subconjunto candidato (D-025), separados por vírgula")
    args = ap.parse_args()

    root = Path(args.root)
    out_tables = Path(args.tables)
    out_tables.mkdir(parents=True, exist_ok=True)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    R: list[str] = ["# PadChest — auditoria preliminar (sem imagens)", ""]

    csv_path = root / args.csv
    if not csv_path.exists():
        print(f"ERRO: não encontrei {csv_path}. Baixe o CSV de rótulos (16 MB) para essa pasta.")
        return 1

    # --- CSV
    df = pd.read_csv(csv_path, low_memory=False)
    R += ["## 1. CSV", f"- Linhas: {len(df):,}", f"- Colunas ({df.shape[1]}): " + ", ".join(df.columns), ""]
    needed = ["ImageID", "ImageDir", "PatientID", "StudyID", "Projection", "Labels"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        R.append(f"**Colunas esperadas ausentes:** {missing}")
        Path(args.report).write_text("\n".join(R), encoding="utf-8")
        print("\n".join(R))
        return 1

    df["labels_list"] = df["Labels"].map(parse_labels)
    df["pneumonia"] = df["labels_list"].map(lambda l: "pneumonia" in l)
    df["any_pneumonia_term"] = df["labels_list"].map(lambda l: any("pneumonia" in x for x in l))
    df["exclude_label"] = df["labels_list"].map(lambda l: "exclude" in l)
    if "StudyDate_DICOM" in df:
        df["study_year"] = pd.to_numeric(df["StudyDate_DICOM"].astype(str).str[:4], errors="coerce")
    if "PatientBirth" in df and "study_year" in df:
        df["age"] = df["study_year"] - pd.to_numeric(df["PatientBirth"], errors="coerce")

    # --- listagens dos zips
    lst = read_listings(root / "unzip-l")
    R += ["## 2. Pacotes (listagens unzip -l)",
          f"- Listagens lidas: {lst['zip'].nunique()} pacotes; PNG listados: {len(lst):,}",
          f"- Pacotes encontrados: {sorted(lst['zip'].unique().tolist())}",
          f"- Tamanho total listado: {lst['bytes'].sum() / 1e9:,.1f} GB; "
          f"média por imagem: {lst['bytes'].mean() / 1e6:.1f} MB", ""]
    dup = lst["ImageID"].duplicated(keep=False)
    if dup.any():
        R.append(f"- **Atenção:** {dup.sum()} entradas com ImageID repetido entre pacotes")

    # --- conferência CSV × listagens
    zip_of = lst.drop_duplicates("ImageID").set_index("ImageID")["zip"]
    df["zip_listed"] = df["ImageID"].map(zip_of)
    df["ImageDir_num"] = pd.to_numeric(df["ImageDir"], errors="coerce")
    found = df["zip_listed"].notna()
    agree = found & (df["zip_listed"] == df["ImageDir_num"])
    R += ["## 3. Conferência CSV × listagens",
          f"- Imagens do CSV encontradas em alguma listagem: {found.sum():,} ({pct(found.sum(), len(df))})",
          f"- Com ImageDir igual ao pacote da listagem: {agree.sum():,} ({pct(agree.sum(), found.sum())} das encontradas)",
          f"- PNG listados que não estão no CSV: {(~lst['ImageID'].isin(df['ImageID'])).sum():,}", ""]
    df["zip"] = df["zip_listed"].fillna(df["ImageDir_num"])

    # --- planilha de verificação
    xlsx = root / "Verify_Zips_ImageCounts.csv.xlsx"
    if xlsx.exists():
        try:
            vx = pd.read_excel(xlsx)
            R += ["## 4. Verify_Zips_ImageCounts (primeiras linhas)", md_table(vx.head(60), index=False), ""]
        except Exception as e:  # openpyxl ausente etc.
            R += [f"## 4. Verify_Zips_ImageCounts: não li ({e})", ""]

    # --- campos DICOM (passo 3.7)
    dicom_cols = [c for c in df.columns if c.endswith("_DICOM") or c in ("ExposureTime",)]
    comp = pd.DataFrame({
        "preenchido_%": [100 * df[c].notna().mean() for c in dicom_cols],
        "n_valores": [df[c].nunique() for c in dicom_cols],
        "exemplos": [", ".join(map(str, df[c].dropna().astype(str).value_counts().index[:4])) for c in dicom_cols],
    }, index=dicom_cols).round(1)
    comp.to_csv(out_tables / "dicom_fields_completeness.csv")
    R += ["## 5. Campos DICOM no CSV (passo 3.7) — todas as imagens", md_table(comp), ""]

    # --- fluxograma de exclusões (provisório: regra de 'primeira frontal' ainda é decisão da F4)
    flow = [("Todas as imagens", len(df))]
    cur = df
    if "Projection" in cur:
        R += ["## 6. Projeções (todas as imagens)", md_table(cur["Projection"].value_counts(dropna=False).to_frame("n")), ""]
    cur = cur[cur["Projection"].isin(FRONTAL)]
    flow.append((f"Frontais {FRONTAL}", len(cur)))
    if "Pediatric" in cur:
        cur = cur[cur["Pediatric"].astype(str).str.upper() != "PED"]
        flow.append(("Sem marca Pediatric=PED", len(cur)))
    if "age" in cur:
        cur = cur[cur["age"] >= ADULT_MIN_AGE]
        flow.append((f"Idade ≥ {ADULT_MIN_AGE} (ano do estudo − ano de nascimento)", len(cur)))
    cur = cur[cur["labels_list"].map(len) > 0]
    flow.append(("Com rótulo", len(cur)))
    cur = cur[cur["zip"].notna()]
    flow.append(("Com pacote identificado", len(cur)))
    # provisório: uma frontal por estudo, preferindo PA > AP > AP_horizontal, depois ImageID
    order = {p: i for i, p in enumerate(FRONTAL)}
    cur = (cur.assign(_o=cur["Projection"].map(order))
              .sort_values(["StudyID", "_o", "ImageID"])
              .drop_duplicates("StudyID"))
    flow.append(("Uma frontal por estudo (provisório: PA > AP > AP_horizontal)", len(cur)))
    R += ["## 7. Fluxograma de exclusões (provisório)",
          md_table(pd.DataFrame(flow, columns=["etapa", "n"]), index=False), ""]

    # --- rótulos (passo 3.9 parcial)
    n, pos = len(cur), int(cur["pneumonia"].sum())
    other_terms = pd.Series([x for l in cur["labels_list"] for x in l if "pneumonia" in x]).value_counts()
    R += ["## 8. Pneumonia no recorte",
          f"- Estudos: {n:,}; pacientes: {cur['PatientID'].nunique():,}",
          f"- Positivos (rótulo exato 'pneumonia'): {pos:,} ({pct(pos, n)})",
          f"- Com qualquer termo contendo 'pneumonia': {int(cur['any_pneumonia_term'].sum()):,}",
          f"- Rótulo 'exclude' presente: {int(cur['exclude_label'].sum()):,}",
          "- Termos com 'pneumonia':", md_table(other_terms.to_frame("n")), ""]
    if "MethodLabel" in cur:
        ml = cur.groupby("MethodLabel").agg(estudos=("StudyID", "size"), positivos=("pneumonia", "sum"))
        ml["prevalencia_%"] = (100 * ml["positivos"] / ml["estudos"]).round(2)
        R += ["- Por método de rotulagem (controle do passo 7.7):", md_table(ml), ""]
    for col, title in [("Projection", "projeção"), ("Manufacturer_DICOM", "fabricante"),
                       ("PatientSex_DICOM", "sexo")]:
        if col in cur:
            t = cur.groupby(col).agg(estudos=("StudyID", "size"), positivos=("pneumonia", "sum"))
            t["prevalencia_%"] = (100 * t["positivos"] / t["estudos"]).round(2)
            R += [f"- Por {title}:", md_table(t.sort_values("estudos", ascending=False).head(12)), ""]
    if "study_year" in cur:
        R += ["- Por ano do estudo:", md_table(cur.groupby("study_year").agg(
            estudos=("StudyID", "size"), positivos=("pneumonia", "sum"))), ""]

    # --- pacotes: equilíbrio (base para D-025)
    zip_bytes = lst.groupby("zip")["bytes"].sum() / 1e9
    per_zip = cur.groupby("zip").agg(
        estudos=("StudyID", "size"), pacientes=("PatientID", "nunique"),
        positivos=("pneumonia", "sum"))
    per_zip["prev_%"] = (100 * per_zip["positivos"] / per_zip["estudos"]).round(2)
    if "study_year" in cur:
        per_zip["ano_min"] = cur.groupby("zip")["study_year"].min()
        per_zip["ano_max"] = cur.groupby("zip")["study_year"].max()
    per_zip["%PA"] = (100 * cur.assign(pa=cur["Projection"].eq("PA")).groupby("zip")["pa"].mean()).round(1)
    if "MethodLabel" in cur:
        per_zip["%medico"] = (100 * cur.assign(m=cur["MethodLabel"].astype(str).str.lower().eq("physician"))
                              .groupby("zip")["m"].mean()).round(1)
    if "age" in cur:
        per_zip["idade_media"] = cur.groupby("zip")["age"].mean().round(1)
    per_zip["GB"] = zip_bytes.reindex(per_zip.index).round(1)
    per_zip.index = per_zip.index.astype(int)
    per_zip.to_csv(out_tables / "per_zip_summary.csv")
    R += ["## 9. Resumo por pacote (recorte)", md_table(per_zip), ""]

    assoc = {}
    zips = cur["zip"].astype(int)
    for col in ["study_year", "pneumonia", "Projection", "Manufacturer_DICOM",
                "MethodLabel", "PatientSex_DICOM"]:
        if col in cur:
            assoc[col] = round(cramers_v(zips, cur[col].astype(str)), 3)
    if "age" in cur:
        assoc["faixa_etaria"] = round(cramers_v(zips, pd.cut(cur["age"], [17, 40, 60, 75, 120]).astype(str)), 3)
    R += ["## 10. Associação pacote × características (V de Cramér)",
          "Referência: < 0,10 desprezível; 0,10–0,30 fraca; > 0,30 forte.",
          md_table(pd.Series(assoc).to_frame("V")), ""]

    multi = cur.groupby("PatientID")["zip"].nunique()
    R += [f"- Pacientes com estudos em mais de um pacote: {(multi > 1).sum():,} ({pct((multi > 1).sum(), len(multi))})", ""]

    # --- quantos pacotes para 750 positivos (sorteio de pacotes inteiros)
    rng = np.random.default_rng(SEED)
    pz = per_zip["positivos"].astype(int)
    gb = per_zip["GB"].fillna(per_zip["GB"].mean())
    ids = pz.index.to_numpy()
    k_needed, gb_needed = [], []
    for _ in range(N_PERMUTATIONS):
        perm = rng.permutation(ids)
        cum = pz.loc[perm].cumsum().to_numpy()
        k = int(np.searchsorted(cum, TARGET_POSITIVES) + 1) if cum[-1] >= TARGET_POSITIVES else np.nan
        k_needed.append(k)
        gb_needed.append(gb.loc[perm[:k]].sum() if k == k else np.nan)
    k_arr, gb_arr = np.array(k_needed, float), np.array(gb_needed, float)
    R += [f"## 11. Pacotes necessários para {TARGET_POSITIVES} positivos ({N_PERMUTATIONS} sorteios, semente {SEED})"]
    if pos < TARGET_POSITIVES:
        R.append(f"- **O recorte inteiro tem só {pos} positivos: não atinge {TARGET_POSITIVES}.** "
                 "Avaliar a regra de 500 com 2 faixas do G3.")
    else:
        R += [f"- Pacotes: mediana {np.nanmedian(k_arr):.0f}, p90 {np.nanpercentile(k_arr, 90):.0f}, máx {np.nanmax(k_arr):.0f}",
              f"- GB: mediana {np.nanmedian(gb_arr):.0f}, p90 {np.nanpercentile(gb_arr, 90):.0f}",
              f"- Positivos por pacote: média {pz.mean():.1f}, mín {pz.min()}, máx {pz.max()}"]
    R.append("")

    R += candidate_subset(cur, zip_bytes, sorted(int(z) for z in args.zips.split(",")))

    report = "\n".join(R)
    Path(args.report).write_text(report, encoding="utf-8")
    print(report)
    print(f"\nRelatório salvo em {args.report}; tabelas em {out_tables}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())