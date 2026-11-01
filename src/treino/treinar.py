"""Treino de um modelo com uma semente e uma taxa de aprendizado (F5 passos 5.3 e 5.4; protocolo §7.2).

Receita (``configs/base.yaml``): DenseNet121 com pesos ImageNet, saída de 1 logit; entropia cruzada
binária; AdamW (weight decay 1e-4); lote 32; cosseno; até 30 épocas; parada antecipada com
paciência 5 pela AUPRC na validação da fonte; precisão mista na GPU; checkpoint a cada época.

Só treino e validação. O teste não está neste workflow.

Saída em ``/out/treinos/<modelo>_s<semente>_lr<taxa>/``:

- ``curvas.csv``: por época, perda de treino e de validação, AUROC, AUPRC, ECE e tempo;
- ``melhor.pt`` (pesos da melhor época) e ``ultimo.pt`` (para retomar se a sessão cair);
- ``predicoes_validacao.csv``: logit, logit com a correção de prior e probabilidade, por imagem;
- ``resumo.json``: configuração usada, época de parada, tempo por época, máquina e a verificação
  da correção de prior (passo 5.3).

Correção de prior (protocolo §7.2): logit + ln(π_nat/(1−π_nat)) − ln(π_treino/(1−π_treino)).
π_nat vem da partição de treino **inteira** (``data/splits/listas_sha256.json``, incertos fora), não
da validação: a validação só verifica se a correção funcionou.

Uso (notebook do ``treino_modelos``, com GPU):
    python -m src.treino.treinar --modelo M1 --semente 0 --taxa 1e-4
    python -m src.treino.treinar --modelo M1 --semente 0 --taxa 1e-4 --retomar
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.treino.carregador import ImagensCache, ler_particao, resumo
from src.treino.modelo import correcao_prior, criar_densenet121, fixar_sementes

RAIZ_DADOS = "/out/dados"
BASES_DO_MODELO = {"M1": ["plus"], "M2": ["rsna"]}           # M3 (as duas, balanceado) entra na F6


# ------------------------------------------------------------------ métricas
def metricas(y: np.ndarray, p: np.ndarray, bins: int = 15) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
            "ece": ece_igual_massa(y, p, bins), "prob_media": float(p.mean()), "prevalencia": float(y.mean())}


def ece_igual_massa(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    """ECE com faixas de igual tamanho (protocolo §9)."""
    ordem = np.argsort(p, kind="stable")
    erro = 0.0
    for grupo in np.array_split(ordem, bins):
        if len(grupo):
            erro += len(grupo) / len(p) * abs(y[grupo].mean() - p[grupo].mean())
    return float(erro)


def prevalencias(base: str, listas_json: str = "data/splits/listas_sha256.json") -> tuple[float, float]:
    """(π natural da partição de treino inteira, π do treino usado pelo modelo)."""
    d = json.load(open(listas_json, encoding="utf-8"))["bases"][base]["listas"]
    tr = d["treino"]
    pi_nat = tr["positivos"] / (tr["imagens"] - tr.get("incertos", 0))
    usado = d.get("treino_subamostra", tr)
    pi_tr = usado["positivos"] / (usado["imagens"] - usado.get("incertos", 0))
    return pi_nat, pi_tr


# ------------------------------------------------------------------ uma época
def _logits(modelo, loader, disp) -> tuple[np.ndarray, np.ndarray, float]:
    modelo.eval()
    perda, n, saidas, rot = 0.0, 0, [], []
    fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
    with torch.no_grad(), torch.autocast(device_type=disp.type, dtype=torch.float16, enabled=disp.type == "cuda"):
        for x, y in loader:
            x, y = x.to(disp, non_blocking=True), y.to(disp)
            z = modelo(x).float()
            perda += fn(z, y).item()
            n += len(y)
            saidas.append(z.squeeze(1).cpu().numpy())
            rot.append(y.squeeze(1).cpu().numpy())
    return np.concatenate(saidas), np.concatenate(rot), perda / n


def treinar(modelo_nome: str, semente: int, taxa: float, cfg: dict, saida: str, raiz: str = RAIZ_DADOS,
            pesos_imagenet: bool = True, epocas_max: int | None = None, limite: int | None = None,
            retomar: bool = False, workers: int | None = None, rotulos_embaralhados: bool = False) -> dict:
    disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fixar_sementes(semente)
    os.makedirs(saida, exist_ok=True)
    t_cfg, a_cfg = cfg["treino"], cfg["aumento"]
    epocas_max = epocas_max or int(t_cfg["epocas_max"])
    paciencia = int(t_cfg["parada_antecipada"]["paciencia"])
    workers = min(8, os.cpu_count() or 2) if workers is None else workers

    bases = BASES_DO_MODELO[modelo_nome]
    if len(bases) != 1:
        raise NotImplementedError("M3 (duas bases, amostragem balanceada) entra na F6")
    base = bases[0]
    tr = ler_particao(base, "treino", raiz=raiz)
    va = ler_particao(base, "validacao", raiz=raiz)
    if limite:                                           # só para testes rápidos
        tr, va = tr.head(limite), va.head(limite)
    if rotulos_embaralhados:                             # sanidade (passo 5.5): só o treino, a validação fica real
        tr = tr.assign(rotulo=np.random.default_rng(semente).permutation(tr["rotulo"].to_numpy()))
    pi_nat, pi_tr = prevalencias(base)

    ds_tr = ImagensCache(tr, a_cfg, semente=semente)
    ds_va = ImagensCache(va, None)
    lote = int(t_cfg["lote"])
    ld_va = DataLoader(ds_va, batch_size=lote * 2, shuffle=False, num_workers=workers,
                       pin_memory=disp.type == "cuda")

    modelo = criar_densenet121(pesos_imagenet).to(disp)
    otim = torch.optim.AdamW(modelo.parameters(), lr=taxa, weight_decay=float(t_cfg["weight_decay"]))
    agenda = torch.optim.lr_scheduler.CosineAnnealingLR(otim, T_max=epocas_max)
    escala = torch.amp.GradScaler("cuda", enabled=disp.type == "cuda")
    fn = torch.nn.BCEWithLogitsLoss()

    inicio, melhor, sem_melhora, curvas = 0, -1.0, 0, []
    ult = os.path.join(saida, "ultimo.pt")
    if retomar and os.path.exists(ult):
        ck = torch.load(ult, map_location=disp, weights_only=False)
        modelo.load_state_dict(ck["modelo"]); otim.load_state_dict(ck["otim"])
        agenda.load_state_dict(ck["agenda"]); escala.load_state_dict(ck["escala"])
        inicio, melhor, sem_melhora, curvas = ck["epoca"] + 1, ck["melhor"], ck["sem_melhora"], ck["curvas"]
        torch.set_rng_state(ck["rng_cpu"])
        if disp.type == "cuda" and ck.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(ck["rng_cuda"])
        print(f"retomando da época {inicio}", flush=True)

    print(f"{modelo_nome} | semente {semente} | taxa {taxa:g} | {disp} | treino {resumo(tr)} | "
          f"validação {resumo(va)} | workers {workers}", flush=True)
    for epoca in range(inicio, epocas_max):
        t0 = time.time()
        ds_tr.definir_epoca(epoca)
        g = torch.Generator().manual_seed(semente * 100_003 + epoca)
        ld_tr = DataLoader(ds_tr, batch_size=lote, shuffle=True, generator=g, num_workers=workers,
                           pin_memory=disp.type == "cuda", drop_last=False)
        modelo.train()
        soma, n = 0.0, 0
        for x, y in ld_tr:
            x, y = x.to(disp, non_blocking=True), y.to(disp)
            otim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=disp.type, dtype=torch.float16, enabled=disp.type == "cuda"):
                perda = fn(modelo(x).float(), y)
            escala.scale(perda).backward()
            escala.step(otim)
            escala.update()
            soma += perda.item() * len(y)
            n += len(y)
        agenda.step()
        t_treino = time.time() - t0
        z, yv, perda_va = _logits(modelo, ld_va, disp)
        m = metricas(yv, 1 / (1 + np.exp(-z)))
        linha = {"epoca": epoca, "perda_treino": soma / n, "perda_validacao": perda_va, "auroc": m["auroc"],
                 "auprc": m["auprc"], "ece_sem_correcao": m["ece"], "taxa": agenda.get_last_lr()[0],
                 "seg_treino": t_treino, "seg_total": time.time() - t0}
        curvas.append(linha)
        pd.DataFrame(curvas).to_csv(os.path.join(saida, "curvas.csv"), index=False)
        if m["auprc"] > melhor:
            melhor, sem_melhora = m["auprc"], 0
            torch.save(modelo.state_dict(), os.path.join(saida, "melhor.pt"))
        else:
            sem_melhora += 1
        torch.save({"modelo": modelo.state_dict(), "otim": otim.state_dict(), "agenda": agenda.state_dict(),
                    "escala": escala.state_dict(), "epoca": epoca, "melhor": melhor, "sem_melhora": sem_melhora,
                    "curvas": curvas, "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all() if disp.type == "cuda" else None}, ult)
        print(f"  época {epoca:2d} | perda {linha['perda_treino']:.4f} / {perda_va:.4f} | AUROC {m['auroc']:.4f} | "
              f"AUPRC {m['auprc']:.4f} | {linha['seg_total']:.0f} s{' *' if sem_melhora == 0 else ''}", flush=True)
        if sem_melhora >= paciencia:
            print(f"  parada antecipada: {paciencia} épocas sem melhorar a AUPRC", flush=True)
            break

    # ---------------------------------------------------------- melhor época, correção de prior
    modelo.load_state_dict(torch.load(os.path.join(saida, "melhor.pt"), map_location=disp, weights_only=True))
    z, yv, _ = _logits(modelo, ld_va, disp)
    zc = correcao_prior(z, pi_nat, pi_tr)
    p_bruta, p_corr = 1 / (1 + np.exp(-z)), 1 / (1 + np.exp(-zc))
    pd.DataFrame({"id": va["id"], "paciente": va["paciente"], "rotulo": yv.astype(int), "logit": z,
                  "logit_corrigido": zc, "prob": p_corr}).to_csv(os.path.join(saida, "predicoes_validacao.csv"),
                                                                    index=False)
    cur = pd.DataFrame(curvas)
    res = {"modelo": modelo_nome, "semente": semente, "taxa": taxa, "base": base,
           "rotulos_embaralhados": rotulos_embaralhados,
           "epoca_melhor": int(cur.loc[cur["auprc"].idxmax(), "epoca"]), "epocas_rodadas": int(len(cur)),
           "seg_por_epoca_mediana": float(cur["seg_total"].median()),
           "pi_natural_treino": pi_nat, "pi_treino_usado": pi_tr,
           "validacao_sem_correcao": metricas(yv, p_bruta), "validacao_com_correcao": metricas(yv, p_corr),
           "maquina": {"dispositivo": str(disp), "gpu": torch.cuda.get_device_name(0) if disp.type == "cuda" else None,
                       "torch": torch.__version__, "python": platform.python_version()},
           "receita": {"treino": t_cfg, "aumento": a_cfg, "workers": workers}}
    json.dump(res, open(os.path.join(saida, "resumo.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    sc, cc = res["validacao_sem_correcao"], res["validacao_com_correcao"]
    print(f"\nMelhor época: {res['epoca_melhor']} | AUROC {cc['auroc']:.4f} | AUPRC {cc['auprc']:.4f}")
    print(f"Correção de prior (π natural {pi_nat:.4f}, π do treino {pi_tr:.4f}):")
    print(f"  sem correção: probabilidade média {sc['prob_media']:.4f} | ECE {sc['ece']:.4f}")
    print(f"  com correção: probabilidade média {cc['prob_media']:.4f} | ECE {cc['ece']:.4f} | "
          f"prevalência observada {cc['prevalencia']:.4f}")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", choices=sorted(BASES_DO_MODELO), required=True)
    ap.add_argument("--semente", type=int, required=True)
    ap.add_argument("--taxa", type=float, required=True)
    ap.add_argument("--retomar", action="store_true", help="continua do ultimo.pt, se existir")
    ap.add_argument("--epocas", type=int, default=None, help="limite de épocas (só para testes)")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))
    if a.taxa not in [float(t) for t in cfg["treino"]["busca_taxa"]]:
        raise SystemExit(f"taxa {a.taxa} fora da grade do protocolo {cfg['treino']['busca_taxa']}")
    saida = f"/out/treinos/{a.modelo}_s{a.semente}_lr{a.taxa:g}"
    treinar(a.modelo, a.semente, a.taxa, cfg, saida, retomar=a.retomar, epocas_max=a.epocas)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
