"""Consolidação: Holm na família 2, BH nas secundárias e T3."""
import numpy as np
import pandas as pd

from src.analises.consolidar import familias, t3


def _b():
    base = {"modelo": "M1", "fonte": "rsna", "controle_auroc_ok": True, "controle_fnr_ok": True, "controle_fpr_ok": True,
            "delta_fnr_max": 0.0, "delta_fpr_max": 0.0, "delta_ece_max": 0.0, "or_nivel_fnr": np.nan, "or_nivel_fpr": np.nan,
            "p_tendencia_auroc": 0.5, "p_tendencia_fnr": 0.9, "p_tendencia_fpr": 0.9, "p_tendencia_ece": 0.9}
    linhas = [
        {**base, "degradacao": "brilho", "lado": "alto", "familia": "primaria", "p_familia2": 0.001,
         "p_tendencia_fpr": 1e-6, "delta_fpr_max": 0.10, "or_nivel_fpr": 1.5,
         "p_tendencia_ece": 1e-6, "delta_ece_max": 0.05},
        {**base, "degradacao": "brilho", "lado": "baixo", "familia": "primaria", "p_familia2": 0.001},
        {**base, "degradacao": "contraste", "lado": "baixo", "familia": "primaria", "p_familia2": 0.4},
        {**base, "degradacao": "ruido", "lado": "alto", "familia": "secundaria", "p_familia2": np.nan,
         "p_tendencia_fnr": np.nan},
    ]
    return pd.DataFrame(linhas)


def test_familias_e_t3():
    b = familias(_b())
    assert b.loc[0, "p_familia2_holm"] == b.loc[1, "p_familia2_holm"] == 0.002       # 2 testes primários
    assert b.loc[3, "p_fnr_bh"] == 1.0                                                # não estimável → p = 1
    a = pd.DataFrame({"modelo": ["M1"], "fonte": ["rsna"], "medida": ["q_brilho"], "erro": ["fp"], "or_1dp": [1.3],
                      "p_ajustado": [0.001]})
    t = t3(b, a, {"rsna": 0.2})
    r = t.iloc[0]
    assert r["n_tendencias"] == 2 and r["mesma_direcao_eixo_a"] and r["t3_dose_resposta"]
    a.loc[0, "or_1dp"] = 0.7                                                          # eixo A na direção oposta
    assert not t3(b, a, {"rsna": 0.2}).iloc[0]["t3_dose_resposta"]



def test_t3_sem_efeito_no_eixo_a_nao_compara():
    b = familias(_b())
    a = pd.DataFrame({"modelo": ["M1"], "fonte": ["rsna"], "medida": ["q_brilho"], "erro": ["fp"], "or_1dp": [1.3],
                      "p_ajustado": [0.4]})                                       # A sem efeito
    r = t3(b, a, {"rsna": 0.2}).iloc[0]
    assert r["mesma_direcao_eixo_a"] is None and not r["t3_dose_resposta"]
    assert r["direcao_comparada"] == "sem efeito no eixo A para comparar"
