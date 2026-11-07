"""Eixo A (§8.3): regressão do erro sobre a qualidade, sementes juntas, EP agrupado por paciente."""
import json
import os

import numpy as np
import pandas as pd

from src.analises.eixo_a import rodar
from src.qualidade.medidas import PRIMARIAS, SECUNDARIAS


def test_eixo_a_detecta_efeito_do_brilho_no_fp(tmp_path):
    rng = np.random.default_rng(0)
    n = 1500
    img = pd.DataFrame({"id_imagem": [f"i{i}" for i in range(n)], "id_paciente": [f"p{i // 2}" for i in range(n)],
                        "rotulo": (rng.random(n) < 0.2).astype(int), "posicao": rng.choice(["AP", "PA"], n),
                        "sexo": rng.choice(["M", "F"], n), "idade": rng.integers(18, 90, n),
                        "fabricante": rng.choice(["Agfa", "GE"], n)})
    for m in PRIMARIAS + SECUNDARIAS:
        img[m] = rng.normal(0, 1, n)
    linhas = []
    for s in range(2):
        ex = f"M1_s{s}"
        os.makedirs(tmp_path / ex)
        json.dump({"limiar": {"limiar_logit": 0.0}}, open(tmp_path / ex / "pos_treino.json", "w"))
        z = 2.0 * img["rotulo"] - 1 + 1.0 * img["q_brilho"] + rng.normal(0, 1, n)   # brilho alto → mais FP
        linhas.append(img.assign(id_execucao=ex, modelo="M1", semente=s, fonte="chexpert_plus", logit=z))
    tab = pd.concat(linhas, ignore_index=True)
    regs, cal = rodar(tab, str(tmp_path))
    assert len(regs) == 8 * 2 and regs["p_ajustado"].notna().all()
    fp = regs[(regs["medida"] == "q_brilho") & (regs["erro"] == "fp")].iloc[0]
    assert fp["or_1dp"] > 1.5 and fp["significativo"]
    fn = regs[(regs["medida"] == "q_brilho") & (regs["erro"] == "fn")].iloc[0]
    assert fn["ressalva"] and fn["or_1dp"] < 1                         # brilho alto → menos FN
    assert "fab_GE" in fp["covariaveis"]
    assert set(cal["faixa"]) == {1, 2}                                  # Plus: 2 faixas
