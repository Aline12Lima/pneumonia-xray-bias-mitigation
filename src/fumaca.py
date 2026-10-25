"""Teste de fumaça da infraestrutura (plano de execução, F2 passo 7).

Prova que o encanamento roda do zero com um comando, com 100 imagens FICTÍCIAS
(nenhum dado real é lido):

    imagens 16 bits → conversão para 8 bits → 320 px → máscara do pulmão
    → medidas de qualidade (§6) → divisão por paciente → 1 época de DenseNet121
    → predição → correção de prior → tabela de predições conferida pelo schema

Uso (na raiz do repositório):
    python -m src.fumaca
    python -m src.fumaca --mascara central --sem-imagenet     # sem internet

Saída em results/fumaca/<data_hora>/: predicoes.csv, config_resolvida.yaml, relatorio.json.
Termina com código 1 se alguma verificação falhar.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.imagem.conversao import para_8bits, redimensionar
from src.imagem import mascara as masc
from src.predicoes.schema import validar
from src.qualidade.medidas import COLUNAS as COLUNAS_QUALIDADE, medidas_qualidade
from src.treino.dados import ImagensRX
from src.treino.modelo import correcao_prior, criar_densenet121, fixar_sementes

RAIZ = Path(__file__).resolve().parents[1]
FONTES = ["chexpert_plus", "rsna", "padchest"]


# ------------------------------------------------------------------ imagens fictícias
def gerar_ficticias(n: int, semente: int = 2026) -> tuple[list[np.ndarray], pd.DataFrame]:
    """Radiografias sintéticas de 16 bits: corpo claro, dois pulmões escuros, 'opacidade' nos positivos."""
    rng = np.random.default_rng(semente)
    imagens, linhas = [], []
    n_pac = int(n * 0.6)
    for i in range(n):
        h, w = [(1024, 1250), (768, 768), (1414, 1160)][i % 3]   # tamanhos variados, não quadrados
        yy, xx = np.mgrid[0:h, 0:w]
        corpo = ((yy - h / 2) / (h * 0.48)) ** 2 + ((xx - w / 2) / (w * 0.45)) ** 2 < 1
        img = rng.normal(9000, 400, (h, w)) + 20000 * corpo
        positivo = int(rng.random() < 0.3)
        for cx in (0.32, 0.68):
            pulmao = ((yy - h * 0.48) / (h * 0.30)) ** 2 + ((xx - w * cx) / (w * 0.14)) ** 2 < 1
            img[pulmao] -= 14000
        if positivo:
            op = ((yy - h * 0.6) / (h * 0.08)) ** 2 + ((xx - w * 0.32) / (w * 0.07)) ** 2 < 1
            img[op] += 9000
        mono1 = (i % 7 == 0)
        img = np.clip(img, 0, 65535)
        if mono1:
            img = 65535 - img
        imagens.append(img.astype(np.uint16))
        fonte = FONTES[i % 3]
        linhas.append({
            "id_imagem": f"fict_{i:03d}",
            "id_paciente": f"pac_{int(rng.integers(0, n_pac)):03d}",
            "id_estudo": f"est_{i:03d}",
            "fonte": fonte,
            "rotulo": positivo,
            "posicao": ["AP", "PA"][int(rng.integers(0, 2))],
            "idade": int(rng.integers(18, 101)),
            "sexo": ["F", "M"][int(rng.integers(0, 2))],
            "fabricante": None if fonte == "rsna" else ["FabA", "FabB"][int(rng.integers(0, 2))],
            "monochrome1": mono1,
        })
    return imagens, pd.DataFrame(linhas)


def dividir_por_paciente(meta: pd.DataFrame, prop: dict, semente: int) -> pd.Series:
    pacientes = np.array(sorted(meta["id_paciente"].unique()))
    np.random.default_rng(semente).shuffle(pacientes)
    n = len(pacientes)
    n_tr, n_va = round(n * prop["treino"]), round(n * prop["validacao"])
    mapa = {p: "treino" for p in pacientes[:n_tr]}
    mapa.update({p: "validacao" for p in pacientes[n_tr:n_tr + n_va]})
    mapa.update({p: "teste" for p in pacientes[n_tr + n_va:]})
    return meta["id_paciente"].map(mapa)


def commit_atual() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=RAIZ,
                              capture_output=True, text=True).stdout.strip() or "desconhecido"
    except Exception:
        return "desconhecido"


# ------------------------------------------------------------------ principal
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(RAIZ / "configs" / "base.yaml"))
    ap.add_argument("--saida", default=str(RAIZ / "results" / "fumaca"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--mascara", choices=["modelo", "central"], default="modelo")
    ap.add_argument("--sem-imagenet", action="store_true", help="não baixar pesos ImageNet (só sem internet)")
    ap.add_argument("--lote", type=int, default=None, help="sobrescreve o lote do YAML (máquinas com pouca RAM)")
    a = ap.parse_args(argv)

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    semente, lr = 0, cfg["treino"]["busca_taxa"][0]
    if a.lote:
        cfg["treino"]["lote"] = a.lote
    fixar_sementes(semente)
    pasta = Path(a.saida) / datetime.now().strftime("%Y%m%d_%H%M%S")
    pasta.mkdir(parents=True, exist_ok=True)
    tempos, checks = {}, {}

    def etapa(nome):
        tempos[nome] = time.time()

    def fim(nome):
        tempos[nome] = round(time.time() - tempos[nome], 2)
        print(f"  {nome}: {tempos[nome]} s", flush=True)

    print(f"Teste de fumaça | dispositivo {dev} | commit {commit_atual()} | {a.n} imagens fictícias")

    etapa("gerar_imagens")
    brutas, meta = gerar_ficticias(a.n)
    fim("gerar_imagens")

    etapa("conversao_8bits_320px")
    ci = cfg["imagem"]["conversao_8bits"]
    img8_orig, img320 = [], []
    for px, m1 in zip(brutas, meta["monochrome1"]):
        g = para_8bits(px, ci["p_baixo"], ci["p_alto"], bool(m1))
        img8_orig.append(g)
        img320.append(redimensionar(g, cfg["imagem"]["tamanho_px"]))
    fim("conversao_8bits_320px")
    checks["conversao_uint8_0_255"] = all(g.dtype == np.uint8 and g.min() == 0 and g.max() == 255 for g in img8_orig)
    checks["todas_320x320"] = all(g.shape == (320, 320) for g in img320)

    etapa("mascara_pulmao")
    rois = []
    for g in img8_orig:
        rois.append(masc.segmentar(g, dispositivo=dev) if a.mascara == "modelo" else masc.mascara_central(g.shape))
    fim("mascara_pulmao")
    frac_vazia = float(np.mean([not r.any() for r in rois]))
    checks["mascara_rodou"] = len(rois) == a.n

    etapa("medidas_qualidade")
    q = pd.DataFrame([medidas_qualidade(g, r, b) for g, r, b in zip(img8_orig, rois, brutas)])
    fim("medidas_qualidade")
    checks["medidas_qualidade_completas"] = list(q.columns) == COLUNAS_QUALIDADE
    del brutas, img8_orig

    meta["particao"] = dividir_por_paciente(meta, cfg["divisao"]["proporcoes"], cfg["divisao"]["semente"])
    checks["nenhum_paciente_em_duas_particoes"] = bool((meta.groupby("id_paciente")["particao"].nunique() == 1).all())

    etapa("treino_1_epoca")
    tr = meta.index[meta["particao"] == "treino"]
    ds_tr = ImagensRX([img320[i] for i in tr], meta.loc[tr, "rotulo"].tolist(), cfg["aumento"], semente)
    dl_tr = DataLoader(ds_tr, batch_size=cfg["treino"]["lote"], shuffle=True,
                       generator=torch.Generator().manual_seed(semente), num_workers=0)
    modelo = criar_densenet121(not a.sem_imagenet).to(dev)
    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=cfg["treino"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(len(dl_tr), 1))
    amp = dev == "cuda" and cfg["treino"]["precisao_mista"]
    scaler = torch.amp.GradScaler(enabled=amp)
    perda_fn, perdas = torch.nn.BCEWithLogitsLoss(), []
    modelo.train()
    for x, y in dl_tr:
        x, y = x.to(dev), y.to(dev)
        with torch.autocast(dev, enabled=amp):
            perda = perda_fn(modelo(x), y)
        opt.zero_grad()
        scaler.scale(perda).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        perdas.append(float(perda))
    if dev == "cuda":
        torch.cuda.synchronize()
    fim("treino_1_epoca")
    checks["perda_finita"] = bool(np.isfinite(perdas).all())
    torch.save({"modelo": modelo.state_dict(), "otimizador": opt.state_dict(), "epoca": 1}, pasta / "checkpoint.pt")
    checks["checkpoint_salvo"] = (pasta / "checkpoint.pt").exists()

    etapa("predicao")
    av = meta.index[meta["particao"].isin(["validacao", "teste"])]
    dl_av = DataLoader(ImagensRX([img320[i] for i in av], meta.loc[av, "rotulo"].tolist()),
                       batch_size=cfg["treino"]["lote"], shuffle=False, num_workers=0)
    modelo.eval()
    logits = []
    with torch.no_grad(), torch.autocast(dev, enabled=amp):
        for x, _ in dl_av:
            logits.append(modelo(x.to(dev)).float().cpu().numpy().ravel())
    fim("predicao")
    logit = np.concatenate(logits) if logits else np.array([])

    pi_tr = float(meta.loc[tr, "rotulo"].mean())
    pi_nat = float(meta["rotulo"].mean())
    logit_c = correcao_prior(logit, pi_nat, pi_tr)
    checks["correcao_prior_neutra_sem_subamostragem"] = bool(np.allclose(correcao_prior(logit, 0.3, 0.3), logit))

    id_exec = f"fumaca_M1_s{semente}_lr{lr:g}"
    pred = meta.loc[av, ["id_imagem", "id_paciente", "id_estudo", "fonte", "particao", "rotulo",
                         "posicao", "idade", "sexo", "fabricante"]].reset_index(drop=True)
    pred.insert(5, "modelo", "M1")
    pred.insert(6, "semente", semente)
    pred.insert(7, "id_execucao", id_exec)
    pred["degradacao"], pred["nivel"] = "nenhuma", "0"
    pred["logit"] = logit_c.astype("float64")
    pred["probabilidade"] = 1.0 / (1.0 + np.exp(-pred["logit"]))
    pred["probabilidade_temperatura"] = np.nan
    pred["fabricante"] = pred["fabricante"].astype("object")
    pred = pd.concat([pred, q.loc[av].reset_index(drop=True)], axis=1)

    erros = validar(pred)
    checks["tabela_valida_no_schema"] = not erros
    pred.to_csv(pasta / "predicoes.csv", index=False)

    resolvida = dict(cfg)
    resolvida["execucao"] = {"id": id_exec, "tipo": "teste_de_fumaca", "modelo": "M1", "semente": semente,
                             "taxa_aprendizado": lr, "dispositivo": dev, "commit": commit_atual(),
                             "mascara": a.mascara, "pesos_imagenet": not a.sem_imagenet,
                             "data": datetime.now().isoformat(timespec="seconds")}
    with open(pasta / "config_resolvida.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(resolvida, f, allow_unicode=True, sort_keys=False)

    relatorio = {
        "ok": all(checks.values()), "verificacoes": checks, "erros_schema": erros,
        "tempos_s": tempos, "dispositivo": dev,
        "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "sem GPU",
        "imagens": a.n, "por_particao": meta["particao"].value_counts().to_dict(),
        "imagens_treino": len(tr), "lote": cfg["treino"]["lote"],
        "treino_img_por_s": round(len(tr) / tempos["treino_1_epoca"], 2) if tempos["treino_1_epoca"] else None,
        "fracao_mascara_vazia": round(frac_vazia, 3), "perda_media": round(float(np.mean(perdas)), 4),
        "saida": str(pasta),
    }
    json.dump(relatorio, open(pasta / "relatorio.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print("\nVerificações:")
    for k, v in checks.items():
        print(f"  [{'OK' if v else 'FALHOU'}] {k}")
    for e in erros:
        print(f"  schema: {e}")
    print(f"\nMáscara vazia em {frac_vazia:.0%} das imagens fictícias (informativo)")
    print(f"Resultado: {'PASSOU' if relatorio['ok'] else 'FALHOU'} | saída em {pasta}")
    return 0 if relatorio["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
