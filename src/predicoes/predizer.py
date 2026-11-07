"""Predições dos 15 modelos finais em todas as bases de validação (F6, passo 6.5: ensaio geral).

O ensaio geral (protocolo §7.4) roda todas as análises sobre a validação antes de abrir o teste.
Para isso, cada modelo final é aplicado à validação do Plus **e** à do RSNA: para o M1, o RSNA faz
o papel de "externo"; para o M2, o Plus. São números de depuração do código, não resultados (o
PadChest não tem validação; o caminho do externo é exercitado pelo cruzamento Plus ↔ RSNA).

A mesma função ``predizer`` será usada no passo 6.8, no teste, com a mesma regra de correção:
- M1 e M2: a correção de prior do treino do modelo, igual para qualquer imagem (M2: zero);
- M3: a correção principal escolhida na validação (F6 › M3 › Correção de prior). Na 3c, a imagem de
  uma fonte de treino recebe a correção dessa fonte; a de fonte externa, a correção da mistura;
- temperatura de ``pos_treino.json`` (passo 6.4).

Saída: ``/out/predicoes/validacao/<execução>.csv`` (formato do schema, validado) e
``/out/predicoes/tabela_validacao.csv`` (as 15 execuções juntas: 15 × 19.016 linhas).

Uso (notebook "treino", GPU):
    python -m src.predicoes.predizer
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.predicoes.pos_treino import SEMENTES, TAXA, pasta_final
from src.predicoes.schema import validar
from src.predicoes.tabela import para_schema
from src.treino.carregador import ImagensCache, ler_particao
from src.treino.modelo import criar_densenet121
from src.treino.treinar import RAIZ_DADOS, _logits, correcoes_m3, prevalencias

SAIDA = "/out/predicoes"
BASES = ["plus", "rsna"]


def correcao_da_imagem(res: dict, fonte: str) -> float:
    """Correção de prior (em logit) que o modelo ``res`` aplica a uma imagem da ``fonte``."""
    lo = lambda p: math.log(p / (1 - p))                                    # noqa: E731
    bases = res.get("bases", [res["base"]])
    if len(bases) == 1:                                   # M1, M2: correção do próprio treino
        pn, pt = prevalencias(bases[0])
        return lo(pn) - lo(pt)
    c = correcoes_m3(bases)
    if res.get("correcao_principal") == "3a":
        return c["3a"]
    return c["3c"][fonte] if fonte in c["3c"] else c["externa"]


def predizer(pasta: str, imagens: pd.DataFrame, workers: int = 4, lote: int = 32) -> pd.DataFrame:
    """Aplica o modelo de ``pasta`` (melhor.pt) às ``imagens`` (id, paciente, rotulo, caminho, fonte)."""
    res = json.load(open(os.path.join(pasta, "resumo.json"), encoding="utf-8"))
    disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = criar_densenet121(False).to(disp)
    modelo.load_state_dict(torch.load(os.path.join(pasta, "melhor.pt"), map_location=disp, weights_only=True))
    ld = DataLoader(ImagensCache(imagens, None), batch_size=lote, shuffle=False, num_workers=workers,
                    pin_memory=disp.type == "cuda")
    z, y, _ = _logits(modelo, ld, disp)
    corr = np.array([correcao_da_imagem(res, f) for f in imagens["fonte"]])
    zc = z + corr
    return pd.DataFrame({"id": imagens["id"].astype(str).to_numpy(), "paciente": imagens["paciente"].astype(str).to_numpy(),
                         "fonte": imagens["fonte"].to_numpy(), "rotulo": y.astype(int), "logit_bruto": z,
                         "logit_corrigido": zc, "prob": 1 / (1 + np.exp(-zc))})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    a = ap.parse_args(argv)
    imagens = pd.concat([ler_particao(b, "validacao", raiz=RAIZ_DADOS) for b in BASES], ignore_index=True)
    os.makedirs(os.path.join(SAIDA, "validacao"), exist_ok=True)
    tabelas, problemas = [], []
    for m in TAXA:
        for s in SEMENTES:
            pasta = pasta_final(m, s)
            res = json.load(open(os.path.join(pasta, "resumo.json"), encoding="utf-8"))
            pred = predizer(pasta, imagens, workers=a.workers)
            # conferência: na validação da fonte, o logit corrigido tem de bater com o do treino
            orig = pd.read_csv(os.path.join(pasta, "predicoes_validacao.csv"), dtype={"id": str})
            if "fonte" not in orig.columns:
                orig["fonte"] = res["base"]
            jun = pred.merge(orig[["id", "fonte", "logit_corrigido"]], on=["id", "fonte"], suffixes=("", "_treino"))
            dif = float(np.abs(jun["logit_corrigido"] - jun["logit_corrigido_treino"]).max())
            t = para_schema(pred, res, pasta, BASES)
            erros = validar(t)
            t.to_csv(os.path.join(SAIDA, "validacao", f"{os.path.basename(pasta)}.csv"), index=False)
            tabelas.append(t)
            txt = " | ".join(f"{f}: prob média {g['prob'].mean():.4f}" for f, g in pred.groupby("fonte"))
            print(f"{m} s{s}: {len(t)} linhas | maior diferença para o treino {dif:.2e} | {txt}"
                  + (f" | ERROS: {erros}" if erros else ""), flush=True)
            if erros or dif > 1e-3:
                problemas.append(os.path.basename(pasta))
    tab = pd.concat(tabelas, ignore_index=True)
    tab.to_csv(os.path.join(SAIDA, "tabela_validacao.csv"), index=False)
    print(f"\n{len(tabelas)} execuções, {len(tab)} linhas em {SAIDA}/tabela_validacao.csv")
    print("RESULTADO:", "OK" if not problemas else f"PROBLEMAS em {problemas}")
    return 0 if not problemas else 1


if __name__ == "__main__":
    raise SystemExit(main())
