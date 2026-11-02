"""Sanidade do pipeline de treino (F5, passo 5.5 do plano de execução).

Três verificações, cada uma com um critério de aprovação fixado aqui, antes de rodar:

1. ``pacientes``: nenhum paciente em duas partições. No workflow de treino só existem treino e
   validação (o teste está fisicamente fora), então a conferência é treino × validação, por base.
   Também confere o SHA-256 das listas contra ``data/splits/listas_sha256.json`` (são as listas
   congeladas) e que nenhum arquivo de teste está em ``/out/dados``. Sem GPU.
   A conferência com o teste já foi feita na geração das listas (``src/divisao/listas.py``).
2. ``sobreajuste``: um lote fixo de 32 imagens do treino do Plus (16 positivas, 16 negativas),
   sem aumento, treinado 200 passos com a receita (AdamW, precisão mista). A perda deve chegar
   perto de zero: critério perda final < 0,01. Mostra que gradiente, perda e rótulos estão ligados.
3. ``embaralhado``: o M1 treinado com os rótulos do treino permutados (mesma prevalência), receita
   inteira, avaliado na validação com os rótulos reais. Sem sinal, a AUROC deve ficar perto de 0,5.
   Mostra que o desempenho do piloto não vem de vazamento.

   Critério (revisto em 01/11, antes de rodar as sementes 1 e 2): AUROC **média de todas as épocas**,
   |média − 0,5| < 0,05, em cada uma das três permutações (sementes 0, 1 e 2). O critério original
   (|AUROC − 0,5| < 0,05 só na melhor época) foi reprovado na semente 0 (0,5552, melhor época 0) e
   fica registrado: escolher a melhor de várias épocas sem sinal puxa o valor para longe de 0,5.
   A semente 0 é reavaliada pelo ``curvas.csv`` que já existe (``--so-avaliar``), sem novo treino.

Saída em ``/out/sanidade/``: ``pacientes.json``, ``sobreajuste.json``, ``embaralhado.json`` (semente 0,
critério original), ``embaralhado_s<semente>.json`` (critério revisto) e as pastas dos treinos
(``M1_s<semente>_lr0.0001_embaralhado/``).

Uso (notebook "treino" do workflow treino_modelos):
    python -m src.treino.sanidade --verificacao pacientes
    python -m src.treino.sanidade --verificacao sobreajuste        # GPU, menos de 1 minuto
    python -m src.treino.sanidade --verificacao embaralhado --semente 0 --so-avaliar   # sem GPU
    python -m src.treino.sanidade --verificacao embaralhado --semente 1                # GPU, um treino curto
    python -m src.treino.sanidade --verificacao embaralhado --semente 2
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
import yaml

from src.divisao.listas import sha256_ids
from src.treino.carregador import PARTICOES, ImagensCache, ler_particao
from src.treino.modelo import criar_densenet121, fixar_sementes
from src.treino.treinar import RAIZ_DADOS, treinar

SAIDA = "/out/sanidade"
LISTAS_JSON = "data/splits/listas_sha256.json"
CHAVE_TREINO = {"plus": "treino_subamostra", "rsna": "treino"}   # nome da lista de treino no listas_sha256.json
LIMITE_SOBREAJUSTE = 0.01
LIMITE_EMBARALHADO = 0.05


def _gravar(nome: str, dados: dict, saida: str) -> None:
    os.makedirs(saida, exist_ok=True)
    json.dump(dados, open(os.path.join(saida, nome), "w", encoding="utf-8"), indent=2, ensure_ascii=False)


# ------------------------------------------------------------------ 1. pacientes
def conferir_pacientes(bases=("plus", "rsna"), raiz: str = RAIZ_DADOS, listas_json: str | None = LISTAS_JSON) -> dict:
    ref = json.load(open(listas_json, encoding="utf-8"))["bases"] if listas_json else None
    res = {"aprovado": True, "bases": {}}
    for base in bases:
        cfg = PARTICOES[base]
        r = {}
        pacientes = {}
        for part in ("treino", "validacao"):
            df = pd.read_csv(cfg[part].replace("/out", raiz, 1))
            col = [c for c in df.columns if c not in ("id", "rotulo")][0]
            pacientes[part] = set(df[col].astype(str))
            r[part] = {"imagens": int(len(df)), "pacientes": len(pacientes[part])}
            if ref is not None:
                chave = CHAVE_TREINO[base] if part == "treino" else part
                esperado = ref[base]["listas"][chave]["sha256"]
                r[part]["sha256_confere"] = sha256_ids(df["id"]) == esperado
        comuns = pacientes["treino"] & pacientes["validacao"]
        r["pacientes_em_duas_particoes"] = len(comuns)
        r["exemplo"] = sorted(comuns)[:3]
        r["arquivos_de_teste"] = sorted(glob.glob(os.path.join(raiz, base, "**", "*teste*"), recursive=True))
        ok = not comuns and not r["arquivos_de_teste"] and all(r[p].get("sha256_confere", True)
                                                                 for p in ("treino", "validacao"))
        r["aprovado"] = ok
        res["aprovado"] &= ok
        res["bases"][base] = r
    return res


# ------------------------------------------------------------------ 2. sobreajuste de um lote
def sobreajustar_lote(cfg: dict, base: str = "plus", n: int = 32, passos: int = 200, taxa: float = 1e-4,
                      semente: int = 0, raiz: str = RAIZ_DADOS, pesos_imagenet: bool = True) -> dict:
    disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fixar_sementes(semente)
    tr = ler_particao(base, "treino", raiz=raiz)
    lote = pd.concat([tr[tr["rotulo"] == 1].sample(n // 2, random_state=semente),
                      tr[tr["rotulo"] == 0].sample(n - n // 2, random_state=semente)]).reset_index(drop=True)
    ds = ImagensCache(lote, None)                        # sem aumento: o mesmo lote em todos os passos
    x = torch.stack([ds[i][0] for i in range(len(ds))]).to(disp)
    y = torch.stack([ds[i][1] for i in range(len(ds))]).to(disp)

    modelo = criar_densenet121(pesos_imagenet).to(disp).train()
    otim = torch.optim.AdamW(modelo.parameters(), lr=taxa, weight_decay=float(cfg["treino"]["weight_decay"]))
    escala = torch.amp.GradScaler("cuda", enabled=disp.type == "cuda")
    fn = torch.nn.BCEWithLogitsLoss()
    perdas = []
    for _ in range(passos):
        otim.zero_grad(set_to_none=True)
        with torch.autocast(device_type=disp.type, dtype=torch.float16, enabled=disp.type == "cuda"):
            perda = fn(modelo(x).float(), y)
        escala.scale(perda).backward()
        escala.step(otim)
        escala.update()
        perdas.append(float(perda.item()))
    return {"base": base, "imagens": len(lote), "positivos": int(lote["rotulo"].sum()), "passos": passos,
            "taxa": taxa, "perda_inicial": perdas[0], "perda_final": perdas[-1],
            "perdas_a_cada_20": perdas[::20], "criterio": f"perda final < {LIMITE_SOBREAJUSTE}",
            "aprovado": perdas[-1] < LIMITE_SOBREAJUSTE, "dispositivo": str(disp)}


# ------------------------------------------------------------------ 3. rótulos embaralhados
def pasta_embaralhado(semente: int, saida_raiz: str = SAIDA) -> str:
    return os.path.join(saida_raiz, f"M1_s{semente}_lr0.0001_embaralhado")


def avaliar_embaralhado(saida: str) -> dict:
    """Critério revisto: AUROC média de todas as épocas, a partir do curvas.csv do treino."""
    cur = pd.read_csv(os.path.join(saida, "curvas.csv"))
    res = json.load(open(os.path.join(saida, "resumo.json"), encoding="utf-8"))
    media = float(cur["auroc"].mean())
    return {"semente": int(res["semente"]), "rotulos_embaralhados": bool(res.get("rotulos_embaralhados", False)),
            "auroc_por_epoca": [round(float(a), 4) for a in cur["auroc"]], "auroc_media": media,
            "auroc_melhor_epoca": res["validacao_com_correcao"]["auroc"], "epoca_melhor": res["epoca_melhor"],
            "auprc_melhor_epoca": res["validacao_com_correcao"]["auprc"],
            "prevalencia_validacao": res["validacao_com_correcao"]["prevalencia"],
            "epocas_rodadas": int(len(cur)), "criterio": f"|AUROC média das épocas - 0,5| < {LIMITE_EMBARALHADO}",
            "aprovado": abs(media - 0.5) < LIMITE_EMBARALHADO}


def treinar_embaralhado(cfg: dict, semente: int = 0, saida_raiz: str = SAIDA, so_avaliar: bool = False, **kw) -> dict:
    """Treina o M1 com os rótulos do treino permutados (semente da permutação = semente do modelo)."""
    saida = pasta_embaralhado(semente, saida_raiz)
    if not so_avaliar:
        treinar("M1", semente, 1e-4, cfg, saida, rotulos_embaralhados=True, retomar=True, **kw)
    r = avaliar_embaralhado(saida)
    if not r["rotulos_embaralhados"]:
        raise RuntimeError(f"{saida} não é um treino com rótulos embaralhados")
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verificacao", choices=["pacientes", "sobreajuste", "embaralhado"], required=True)
    ap.add_argument("--semente", type=int, default=0, help="embaralhado: semente da permutação e do modelo")
    ap.add_argument("--so-avaliar", action="store_true", help="embaralhado: só reavalia um treino já feito")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))
    if a.verificacao == "pacientes":
        r = conferir_pacientes()
        for base, b in r["bases"].items():
            print(f"{base}: treino {b['treino']} | validação {b['validacao']} | pacientes em duas partições: "
                  f"{b['pacientes_em_duas_particoes']} | arquivos de teste em /out/dados: {len(b['arquivos_de_teste'])}")
    elif a.verificacao == "sobreajuste":
        r = sobreajustar_lote(cfg)
        print(f"lote de {r['imagens']} imagens ({r['positivos']} positivas), {r['passos']} passos, {r['dispositivo']}")
        print("perda a cada 20 passos:", " ".join(f"{p:.4f}" for p in r["perdas_a_cada_20"]))
        print(f"perda inicial {r['perda_inicial']:.4f} → final {r['perda_final']:.6f} ({r['criterio']})")
    else:
        r = treinar_embaralhado(cfg, semente=a.semente, so_avaliar=a.so_avaliar)
        print(f"\nRótulos embaralhados, semente {r['semente']}: AUROC por época {r['auroc_por_epoca']}")
        print(f"AUROC média {r['auroc_media']:.4f} | melhor época {r['epoca_melhor']} (AUROC "
              f"{r['auroc_melhor_epoca']:.4f}, AUPRC {r['auprc_melhor_epoca']:.4f}, prevalência "
              f"{r['prevalencia_validacao']:.4f}) | {r['criterio']}")
    nome = f"embaralhado_s{a.semente}.json" if a.verificacao == "embaralhado" else f"{a.verificacao}.json"
    _gravar(nome, r, SAIDA)
    print("RESULTADO:", "APROVADO" if r["aprovado"] else "REPROVADO", f"(salvo em {SAIDA}/{nome})")
    return 0 if r["aprovado"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
