"""Junta das predições do teste das três bases e caminho das análises com o PadChest (dados sintéticos)."""
import json
import os

import numpy as np
import pandas as pd

from src.cache.covariaveis import MEDIDAS
from src.predicoes.juntar_teste import juntar
from src.predicoes.pos_treino import SEMENTES, TAXA
from src.predicoes.schema import carregar_schema

BASES = {"plus": ("chexpert_plus", 300, 0.06, 40), "rsna": ("rsna", 200, 0.2, 0), "padchest": ("padchest", 250, 0.05, 0)}


def _tabelas(tmp_path):
    rng = np.random.default_rng(0)
    ref, treinos = {}, tmp_path / "treinos"
    for b, (fonte, n, prev, n_inc) in BASES.items():
        img = pd.DataFrame({"id_imagem": [f"{b}{i}" for i in range(n + n_inc)],
                            "id_paciente": [f"{b}p{i // 2}" for i in range(n + n_inc)],
                            "id_estudo": [f"{b}e{i}" for i in range(n + n_inc)], "fonte": fonte, "particao": "teste",
                            "rotulo": (rng.random(n + n_inc) < prev).astype(int), "posicao": rng.choice(["AP", "PA"], n + n_inc),
                            "idade": rng.integers(18, 95, n + n_inc), "sexo": rng.choice(["F", "M"], n + n_inc),
                            "fabricante": "X" if b != "rsna" else None})
        for m in MEDIDAS:
            img[m] = rng.normal(0, 1, len(img))
        img.loc[n:, "rotulo"] = 1                           # incertos (só Plus) entram com rótulo 1
        linhas = []
        for m in TAXA:
            for s in SEMENTES:
                ex = f"{m}_s{s}_lr{TAXA[m]:g}"
                z = -2.5 + 2 * img["rotulo"] + rng.normal(0, 1, len(img))
                linhas.append(img.assign(modelo=m, semente=s, id_execucao=ex, degradacao="nenhuma", nivel="0", logit=z,
                                         probabilidade=1 / (1 + np.exp(-z)), probabilidade_temperatura=1 / (1 + np.exp(-z))))
                os.makedirs(treinos / ex, exist_ok=True)
                json.dump({"limiar": {"limiar_logit": -1.0}, "temperatura": 1.0}, open(treinos / ex / "pos_treino.json", "w"))
        t = pd.concat(linhas, ignore_index=True)[list(carregar_schema()["colunas"]) + MEDIDAS]
        prim = t[~t["id_imagem"].isin(img["id_imagem"].iloc[n:])]
        prim.to_csv(tmp_path / f"{b}__teste__tabela_teste.csv.gz", index=False, compression="gzip")
        if n_inc:
            t[t["id_imagem"].isin(img["id_imagem"].iloc[n:])].to_csv(
                tmp_path / f"{b}__teste__tabela_teste_incertos.csv.gz", index=False, compression="gzip")
        u = img.iloc[:n]
        ref[b] = {"listas": {"teste": {"imagens": n + n_inc, "positivos": int(u["rotulo"].sum()),
                                       "pacientes": img["id_paciente"].nunique(), **({"incertos": n_inc} if n_inc else {})}}}
    hl = tmp_path / "listas.json"
    hl.write_text(json.dumps({"bases": ref}))
    return str(hl), str(treinos)


def test_junta_e_confere(tmp_path):
    hl, _ = _tabelas(tmp_path)
    tab, sens, erros = juntar(str(tmp_path), hl)
    assert erros == []
    assert tab["fonte"].nunique() == 3 and len(sens) - len(tab) == 40 * 15


def test_recusa_contagem_diferente(tmp_path):
    hl, _ = _tabelas(tmp_path)
    d = json.load(open(hl))
    d["bases"]["padchest"]["listas"]["teste"]["positivos"] += 1
    json.dump(d, open(hl, "w"))
    _, _, erros = juntar(str(tmp_path), hl)
    assert any("padchest" in e and "positivos" in e for e in erros)


def test_e0_com_padchest_como_externo(tmp_path):
    from src.analises import e0
    hl, treinos = _tabelas(tmp_path)
    tab, _, _ = juntar(str(tmp_path), hl)
    arq = tmp_path / "tabela_teste.csv"
    tab.to_csv(arq, index=False)
    assert e0.main(["--tabela", str(arq), "--treinos", treinos, "--saida", str(tmp_path / "a"), "--bootstrap", "5"]) == 0
    gaps = pd.read_csv(tmp_path / "a" / "e0_gaps.csv") if os.path.exists(tmp_path / "a" / "e0_gaps.csv") else None
    pares = pd.read_csv(tmp_path / "a" / "e0_pares.csv")
    assert {"padchest"} <= set(pares["fonte"])
    if gaps is not None:
        assert (gaps["externo"] == "padchest").sum() == 4      # M1, M2 e M3 (duas fontes internas)
