"""F10 (exploratória): amostra, embeddings, Grad-CAM com sanidade, sondas e resumo."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.analises import f10_latente as F


def test_amostra_limita_positivos():
    tab = pd.DataFrame({"id_imagem": [str(i) for i in range(5000)], "id_paciente": [str(i) for i in range(5000)],
                        "fonte": "rsna", "rotulo": [1] * 1500 + [0] * 3500})
    a = F.amostra(tab)
    assert len(a) == 3000 and a["rotulo"].sum() == 1000 and a["id_imagem"].is_unique
    assert a.equals(F.amostra(tab))


def test_embeddings_e_gradcam(tmp_path):
    import cv2
    from src.treino.modelo import criar_densenet121
    torch.manual_seed(0)
    m = criar_densenet121(False).eval()
    caminhos = []
    for i in range(3):
        p = tmp_path / f"{i}.png"
        cv2.imwrite(str(p), (np.random.default_rng(i).random((320, 320)) * 255).astype(np.uint8))
        caminhos.append(str(p))
    E = F.embeddings(m, caminhos, torch.device("cpu"), lote=2, workers=0)
    assert E.shape == (3, 1024) and E.dtype == np.float16
    from src.treino.carregador import ImagensCache
    x, _ = ImagensCache(pd.DataFrame({"caminho": caminhos[:1], "rotulo": [0]}), None)[0]
    c = F.gradcam(m, x.unsqueeze(0))[0]
    assert c.shape == (10, 10) and (c >= 0).all()
    r = F.randomizar_topo(m)
    assert not torch.equal(r.classifier.weight, m.classifier.weight)
    assert torch.equal(r.features.conv0.weight, m.features.conv0.weight)       # só o topo muda
    mask = np.zeros((320, 320), np.uint8); mask[:, :160] = 1
    assert F.massa_na_roi(np.ones((10, 10)), mask) == pytest.approx(0.5, abs=0.02)


def test_sondas_detectam_origem():
    from src.analises.t1_t2 import COMPARAVEIS
    rng = np.random.default_rng(0)
    n = 600
    fonte = np.repeat(["chexpert_plus", "rsna", "padchest"], n // 3)
    meta = pd.DataFrame({"id_imagem": [str(i) for i in range(n)], "id_paciente": [str(i // 2) for i in range(n)],
                         "fonte": fonte, "rotulo": rng.integers(0, 2, n)})
    off = pd.Series(fonte).map({"chexpert_plus": 0, "rsna": 1.5, "padchest": -1.5}).to_numpy()
    emb = {"M1_s0_lr0.0001": (rng.normal(0, 1, (n, 1024)) + 3 * off[:, None] * (np.arange(1024) < 20)).astype(np.float16)}
    med = pd.DataFrame({"id_imagem": meta["id_imagem"], **{c: rng.normal(0, 1, n) for c in COMPARAVEIS}})
    med["q_brilho"] = emb["M1_s0_lr0.0001"][:, 100].astype(float) * 3 + rng.normal(0, 0.1, n)
    s = F.sondas(emb, meta, med, n_boot=60)
    r = s.iloc[0]
    assert r["auroc_fonte_embedding"] > 0.9 and r["auroc_fonte_medidas"] < 0.7 and r["p_diferenca"] < 0.05
    assert r["r2_q_brilho"] > 0.3 and r["p_r2_q_brilho"] < 0.05


def test_resumo_gradcam_respeita_sanidade():
    g = pd.DataFrame({"modelo": ["M1"] * 9 + ["M2"] * 9, "fonte": "rsna", "tipo": ["VP", "FP", "FN"] * 6,
                      "massa_roi": np.linspace(0.2, 0.9, 18), "spearman_aleatorio": [0.1] * 9 + [0.9] * 9})
    r = F.resumo_gradcam(g).set_index("modelo")
    assert r.loc["M1", "interpretavel"] and not r.loc["M2", "interpretavel"]
    assert np.isfinite(r.loc["M1", "p_tipo"]) and np.isnan(r.loc["M2", "p_tipo"])
