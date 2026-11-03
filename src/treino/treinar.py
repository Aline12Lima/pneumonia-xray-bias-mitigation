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
    python -m src.treino.treinar --modelo M1 --semente 0 --taxa 1e-4 --repeticao 1 --retomar   # passo 5.6
    python -m src.treino.treinar --modelo M3 --semente 0 --taxa 1e-4 --retomar                 # F6

M3 (F6 › M3, decisões de 3/11): Plus + RSNA; cada época usa todas as 12.100 imagens de treino do Plus e
12.100 do RSNA sorteadas sem reposição (por semente e época); parada antecipada pela média simples das
AUPRCs das validações das duas fontes; correção de prior 3a (única, da mistura) e 3c (pela fonte de
treino da imagem) calculadas na validação, e a principal é a de menor ECE médio das duas validações.
Tudo o que o treino imprime também vai para ``log.txt`` na pasta da execução.
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
BASES_DO_MODELO = {"M1": ["plus"], "M2": ["rsna"], "M3": ["plus", "rsna"]}


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


def correcoes_m3(bases=("plus", "rsna"), listas_json: str = "data/splits/listas_sha256.json") -> dict:
    """Correções de prior do M3 (F6 › M3 › Correção de prior), em unidades de logit.

    3c: cada imagem recebe a correção da fonte de treino a que pertence (Plus: a do M1; RSNA: zero).
    3a: uma correção única, da mistura balanceada: π do treino = média dos π de treino das fontes,
    π natural = média dos π naturais. A imagem externa (PadChest) recebe a 3a nas duas regras.
    """
    lo = lambda p: math.log(p / (1 - p))                                    # noqa: E731
    pis = {b: prevalencias(b, listas_json) for b in bases}
    por_fonte = {b: lo(pn) - lo(pt) for b, (pn, pt) in pis.items()}
    pi_nat_mix = float(np.mean([pis[b][0] for b in bases]))
    pi_tr_mix = float(np.mean([pis[b][1] for b in bases]))
    unica = lo(pi_nat_mix) - lo(pi_tr_mix)
    return {"pis": {b: {"natural": pn, "treino": pt} for b, (pn, pt) in pis.items()},
            "3c": por_fonte, "3a": unica, "externa": unica,
            "pi_natural_mistura": pi_nat_mix, "pi_treino_mistura": pi_tr_mix}


def indices_balanceados(fontes: np.ndarray, semente: int, epoca: int) -> list[int]:
    """Época do M3 (F6 › M3 › Época balanceada): todas as imagens da menor fonte e o mesmo número da
    outra, sorteadas sem reposição; sorteio e ordem fixados por (semente, época)."""
    g = torch.Generator().manual_seed(semente * 100_003 + epoca)
    grupos = [np.flatnonzero(fontes == f) for f in sorted(set(fontes))]
    n = min(len(i) for i in grupos)
    escolhidos = []
    for i in grupos:
        escolhidos.append(i if len(i) == n else i[torch.randperm(len(i), generator=g)[:n].numpy()])
    todos = np.concatenate(escolhidos)
    return todos[torch.randperm(len(todos), generator=g).numpy()].tolist()


def metricas_por_fonte(y: np.ndarray, p: np.ndarray, fontes: np.ndarray) -> dict:
    """Métricas de cada fonte e a média simples entre as fontes (critério de parada do M3)."""
    por = {f: metricas(y[fontes == f], p[fontes == f]) for f in sorted(set(fontes))}
    return {"por_fonte": por, "auroc": float(np.mean([m["auroc"] for m in por.values()])),
            "auprc": float(np.mean([m["auprc"] for m in por.values()])),
            "ece": float(np.mean([m["ece"] for m in por.values()]))}


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
    multi = len(bases) > 1                               # M3: Plus + RSNA, época balanceada por fonte
    if multi and rotulos_embaralhados:
        raise ValueError("a sanidade com rótulos embaralhados é só para M1 e M2")
    base = "+".join(bases)
    trs, vas = [], []
    for b in bases:
        tr_b, va_b = ler_particao(b, "treino", raiz=raiz), ler_particao(b, "validacao", raiz=raiz)
        if limite:                                       # só para testes rápidos
            tr_b, va_b = tr_b.head(limite), va_b.head(limite)
        trs.append(tr_b); vas.append(va_b)
    tr = pd.concat(trs, ignore_index=True) if multi else trs[0]
    va = pd.concat(vas, ignore_index=True) if multi else vas[0]
    if rotulos_embaralhados:                             # sanidade (passo 5.5): só o treino, a validação fica real
        tr = tr.assign(rotulo=np.random.default_rng(semente).permutation(tr["rotulo"].to_numpy()))
    if multi:
        corr = correcoes_m3(bases)
        pi_nat, pi_tr = corr["pi_natural_mistura"], corr["pi_treino_mistura"]
    else:
        pi_nat, pi_tr = prevalencias(base)
    fontes_tr, fontes_va = tr["fonte"].to_numpy(), va["fonte"].to_numpy()

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
        # carregado na CPU: o estado do gerador aleatório precisa ser um ByteTensor da CPU (com
        # map_location=cuda ele ia para a GPU e o set_rng_state falhava); pesos e otimizador vão
        # para a GPU sozinhos no load_state_dict
        ck = torch.load(ult, map_location="cpu", weights_only=False)
        modelo.load_state_dict(ck["modelo"]); otim.load_state_dict(ck["otim"])
        agenda.load_state_dict(ck["agenda"]); escala.load_state_dict(ck["escala"])
        inicio, melhor, sem_melhora, curvas = ck["epoca"] + 1, ck["melhor"], ck["sem_melhora"], ck["curvas"]
        torch.set_rng_state(ck["rng_cpu"].cpu())
        if disp.type == "cuda" and ck.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all([e.cpu() for e in ck["rng_cuda"]])
        print(f"retomando da época {inicio}", flush=True)

    print(f"{modelo_nome} | semente {semente} | taxa {taxa:g} | {disp} | treino {resumo(tr)} | "
          f"validação {resumo(va)} | workers {workers}", flush=True)
    if multi:
        for b, tr_b, va_b in zip(bases, trs, vas):
            print(f"  {b}: treino {resumo(tr_b)} | validação {resumo(va_b)}", flush=True)
        print(f"  época balanceada: {min(len(t) for t in trs)} imagens de cada fonte, sorteadas sem reposição",
              flush=True)
    if sem_melhora >= paciencia:                         # a sessão caiu depois da parada antecipada
        print(f"  parada antecipada já atingida na época {inicio - 1}: só a avaliação final", flush=True)
        inicio = epocas_max
    for epoca in range(inicio, epocas_max):
        t0 = time.time()
        ds_tr.definir_epoca(epoca)
        if multi:
            ld_tr = DataLoader(ds_tr, batch_size=lote, sampler=indices_balanceados(fontes_tr, semente, epoca),
                               num_workers=workers, pin_memory=disp.type == "cuda", drop_last=False)
        else:                                            # M1 e M2: exatamente como no piloto (determinismo)
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
        if multi:                                        # média simples das fontes (F6 › M3 › Parada antecipada)
            m = metricas_por_fonte(yv, 1 / (1 + np.exp(-z)), fontes_va)
        else:
            m = metricas(yv, 1 / (1 + np.exp(-z)))
        linha = {"epoca": epoca, "perda_treino": soma / n, "perda_validacao": perda_va, "auroc": m["auroc"],
                 "auprc": m["auprc"], "ece_sem_correcao": m["ece"], "taxa": agenda.get_last_lr()[0],
                 "seg_treino": t_treino, "seg_total": time.time() - t0}
        if multi:
            for f, mf in m["por_fonte"].items():
                linha[f"auroc_{f}"], linha[f"auprc_{f}"] = mf["auroc"], mf["auprc"]
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
        fontes_txt = "".join(f" [{f}: {mf['auroc']:.4f} / {mf['auprc']:.4f}]" for f, mf in m["por_fonte"].items()) \
            if multi else ""
        print(f"  época {epoca:2d} | perda {linha['perda_treino']:.4f} / {perda_va:.4f} | AUROC {m['auroc']:.4f} | "
              f"AUPRC {m['auprc']:.4f}{fontes_txt} | {linha['seg_total']:.0f} s{' *' if sem_melhora == 0 else ''}",
              flush=True)
        if sem_melhora >= paciencia:
            print(f"  parada antecipada: {paciencia} épocas sem melhorar a AUPRC", flush=True)
            break

    # ---------------------------------------------------------- melhor época, correção de prior
    modelo.load_state_dict(torch.load(os.path.join(saida, "melhor.pt"), map_location=disp, weights_only=True))
    z, yv, _ = _logits(modelo, ld_va, disp)
    p_bruta = 1 / (1 + np.exp(-z))
    pred = {"id": va["id"], "paciente": va["paciente"], "fonte": va["fonte"], "rotulo": yv.astype(int), "logit": z}
    extra = {}
    if multi:
        # F6 › M3 › Correção de prior: 3a e 3c calculadas; principal = menor ECE médio das validações das fontes
        z3c = z + np.array([corr["3c"][f] for f in fontes_va])
        z3a = z + corr["3a"]
        sig = lambda v: 1 / (1 + np.exp(-v))                               # noqa: E731
        m3a, m3c = metricas_por_fonte(yv, sig(z3a), fontes_va), metricas_por_fonte(yv, sig(z3c), fontes_va)
        principal = "3c" if m3c["ece"] <= m3a["ece"] else "3a"
        zc = z3c if principal == "3c" else z3a
        pred.update({"logit_corrigido_3a": z3a, "logit_corrigido_3c": z3c})
        extra = {"correcao_prior": corr, "correcao_principal": principal,
                 "validacao_3a": m3a, "validacao_3c": m3c,
                 "validacao_sem_correcao_por_fonte": metricas_por_fonte(yv, p_bruta, fontes_va)}
    else:
        zc = correcao_prior(z, pi_nat, pi_tr)
        extra = {"correcao_prior": {base: correcao_prior(np.zeros(1), pi_nat, pi_tr)[0].item(),
                                    "externa": correcao_prior(np.zeros(1), pi_nat, pi_tr)[0].item()}}
    p_corr = 1 / (1 + np.exp(-zc))
    pred.update({"logit_corrigido": zc, "prob": p_corr})
    pd.DataFrame(pred).to_csv(os.path.join(saida, "predicoes_validacao.csv"), index=False)
    cur = pd.DataFrame(curvas)
    res = {"modelo": modelo_nome, "semente": semente, "taxa": taxa, "base": base, "bases": bases,
           "rotulos_embaralhados": rotulos_embaralhados,
           "epoca_melhor": int(cur.loc[cur["auprc"].idxmax(), "epoca"]), "epocas_rodadas": int(len(cur)),
           "seg_por_epoca_mediana": float(cur["seg_total"].median()),
           "pi_natural_treino": pi_nat, "pi_treino_usado": pi_tr,
           "validacao_sem_correcao": metricas(yv, p_bruta), "validacao_com_correcao": metricas(yv, p_corr),
           **extra,
           "maquina": {"dispositivo": str(disp), "gpu": torch.cuda.get_device_name(0) if disp.type == "cuda" else None,
                       "torch": torch.__version__, "python": platform.python_version()},
           "receita": {"treino": t_cfg, "aumento": a_cfg, "workers": workers}}
    if multi:                                            # para o M3, a métrica da fonte é a média das fontes
        res["validacao_com_correcao_por_fonte"] = metricas_por_fonte(yv, p_corr, fontes_va)
    json.dump(res, open(os.path.join(saida, "resumo.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    sc, cc = res["validacao_sem_correcao"], res["validacao_com_correcao"]
    if multi:
        mel = cur.loc[cur["auprc"].idxmax()]
        txt = " | ".join(f"{f}: AUROC {mel[f'auroc_{f}']:.4f}, AUPRC {mel[f'auprc_{f}']:.4f}" for f in bases)
        print(f"\nMelhor época: {res['epoca_melhor']} | média das fontes: AUROC {mel['auroc']:.4f}, "
              f"AUPRC {mel['auprc']:.4f} | {txt}")
        for nome, mm in (("sem correção", extra["validacao_sem_correcao_por_fonte"]), ("3a (única)", m3a),
                         ("3c (pela fonte)", m3c)):
            txt = " | ".join(f"{f}: prob média {v['prob_media']:.4f}, ECE {v['ece']:.4f}, prevalência "
                             f"{v['prevalencia']:.4f}" for f, v in mm["por_fonte"].items())
            print(f"  {nome:16s} ECE médio {mm['ece']:.4f} | {txt}")
        print(f"Correção principal (menor ECE médio): {principal}")
    else:
        print(f"\nMelhor época: {res['epoca_melhor']} | AUROC {cc['auroc']:.4f} | AUPRC {cc['auprc']:.4f}")
        print(f"Correção de prior (π natural {pi_nat:.4f}, π do treino {pi_tr:.4f}):")
        print(f"  sem correção: probabilidade média {sc['prob_media']:.4f} | ECE {sc['ece']:.4f}")
        print(f"  com correção: probabilidade média {cc['prob_media']:.4f} | ECE {cc['ece']:.4f} | "
              f"prevalência observada {cc['prevalencia']:.4f}")
    return res


class _Log:
    """Copia tudo o que o treino imprime para um arquivo (acompanhar pelo Files do Redivis)."""

    def __init__(self, caminho: str):
        self.caminho = caminho

    def __enter__(self):
        import sys
        self._orig, self._f = sys.stdout, open(self.caminho, "a", encoding="utf-8")
        self._f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        orig, f = self._orig, self._f

        class _T:
            def write(self, t):
                orig.write(t); f.write(t); f.flush()

            def flush(self):
                orig.flush(); f.flush()

        sys.stdout = _T()
        return self

    def __exit__(self, *exc):
        import sys
        sys.stdout = self._orig
        self._f.close()
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", choices=sorted(BASES_DO_MODELO), required=True)
    ap.add_argument("--semente", type=int, required=True)
    ap.add_argument("--taxa", type=float, required=True)
    ap.add_argument("--retomar", action="store_true", help="continua do ultimo.pt, se existir")
    ap.add_argument("--epocas", type=int, default=None, help="limite de épocas (só para testes)")
    ap.add_argument("--repeticao", type=int, default=0,
                    help="determinismo (passo 5.6): repete a mesma configuração numa pasta _rep<N>")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))
    if a.taxa not in [float(t) for t in cfg["treino"]["busca_taxa"]]:
        raise SystemExit(f"taxa {a.taxa} fora da grade do protocolo {cfg['treino']['busca_taxa']}")
    saida = f"/out/treinos/{a.modelo}_s{a.semente}_lr{a.taxa:g}" + (f"_rep{a.repeticao}" if a.repeticao else "")
    os.makedirs(saida, exist_ok=True)
    with _Log(os.path.join(saida, "log.txt")):          # a tela do notebook às vezes para de atualizar
        treinar(a.modelo, a.semente, a.taxa, cfg, saida, retomar=a.retomar, epocas_max=a.epocas)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
