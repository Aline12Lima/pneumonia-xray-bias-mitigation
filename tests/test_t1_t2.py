"""T1 e T2 (§8.2) num conjunto sintético."""
import numpy as np
import pandas as pd

from src.analises.t1_t2 import imagens, t1, t2
from src.qualidade.medidas import PRIMARIAS, SECUNDARIAS


def _tab():
    rng = np.random.default_rng(0)
    partes = []
    for fonte, desloc in (("chexpert_plus", 0.0), ("rsna", 1.5)):
        n = 600
        d = pd.DataFrame({"id_imagem": [f"{fonte}{i}" for i in range(n)], "id_paciente": [f"{fonte}p{i // 2}" for i in range(n)],
                          "fonte": fonte, "rotulo": (rng.random(n) < 0.25).astype(int),
                          "posicao": rng.choice(["AP", "PA"], n), "sexo": rng.choice(["M", "F"], n),
                          "idade": rng.integers(18, 90, n), "degradacao": "nenhuma"})
        for m in PRIMARIAS + SECUNDARIAS:
            d[m] = rng.normal(desloc, 1, n)
        d["q_brilho"] += 1.0 * d["rotulo"]                                  # brilho carrega o rótulo
        d.loc[0, "q_contraste"] = np.nan                                     # máscara vazia
        partes.append(pd.concat([d.assign(id_execucao="M1_s0"), d.assign(id_execucao="M1_s1")]))
    return pd.concat(partes, ignore_index=True)


def test_t1_brilho_carrega_rotulo_e_t2_detecta_fonte():
    img = imagens(_tab())
    assert len(img) == 1200
    r1 = t1(img, n_boot=20)
    lr = r1[(r1["versao"] == "interno") & (r1["classificador"] == "logistica")]
    com = lr[lr["com_brilho"]]["ganho_sobre_posicao"].mean()
    sem = lr[~lr["com_brilho"]]["ganho_sobre_posicao"].mean()
    assert com > 0.15 and com > sem + 0.1
    assert (r1["fora_por_medida_vazia"] >= 1).any()
    assert set(r1[r1["versao"] == "cruzado"]["ajuste"]) == {"chexpert_plus", "rsna"}
    r2 = t2(img, n_boot=20)
    assert (r2["auroc"] > 0.9).all() and set(r2["grupo"]) == {"todas", "positivos", "negativos"}
