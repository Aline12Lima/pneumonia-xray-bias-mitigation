"""Conferência do carregador (F5, passo 5.2), antes de qualquer treino.

Roda na CPU gratuita e verifica, numa base:

1. as listas de treino e validação com o rótulo principal (contagens e prevalência);
2. o hash de cada arquivo do cache contra o passo 5.1;
3. a forma e a escala dos tensores (3 × 320 × 320, normalização do ImageNet);
4. o determinismo do aumento: a mesma imagem na mesma época dá o mesmo tensor, com 0 ou 2 workers;
   em épocas diferentes, o aumento muda;
5. uma figura com 8 imagens do treino, sem aumento e com o aumento de duas épocas, para conferir a olho
   que o aumento é só geométrico e leve.

Uso:
    python -m src.treino.conferir_carregador --base plus
"""
from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.treino.carregador import ImagensCache, ler_particao, resumo

AUMENTO = {"escala": [0.9, 1.0], "rotacao_graus": 5}


def _para_img(x: torch.Tensor) -> np.ndarray:
    from src.treino.dados import DP_IMAGENET, MEDIA_IMAGENET
    v = (x * DP_IMAGENET + MEDIA_IMAGENET)[0].clamp(0, 1).numpy()
    return (v * 255).round().astype(np.uint8)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", choices=["plus", "rsna"], required=True)
    a = ap.parse_args(argv)

    t0 = time.time()
    tr = ler_particao(a.base, "treino")
    va = ler_particao(a.base, "validacao")
    print(f"=== {a.base}: listas e hashes conferidos em {time.time() - t0:.0f} s")
    print(json.dumps({"treino": resumo(tr), "validacao": resumo(va)}, indent=2, ensure_ascii=False))
    print("Pacientes em comum entre treino e validação:", len(set(tr["paciente"]) & set(va["paciente"])))

    ds = ImagensCache(tr, AUMENTO, semente=0)
    sem = ImagensCache(tr, None)
    x, y = ds[0]
    print(f"\n=== Tensor: forma {tuple(x.shape)}, tipo {x.dtype}, mínimo {x.min():.2f}, máximo {x.max():.2f}, rótulo {y.tolist()}")
    lote = torch.stack([sem[i][0] for i in range(64)])
    print(f"Média por canal (64 imagens, sem aumento): {lote.mean(dim=(0, 2, 3)).numpy().round(3)} "
          f"| desvio: {lote.std(dim=(0, 2, 3)).numpy().round(3)}")

    igual = torch.equal(ds[5][0], ds[5][0])
    ds.definir_epoca(1)
    muda = not torch.equal(ds[5][0], ImagensCache(tr, AUMENTO, semente=0)[5][0])
    ds.definir_epoca(0)
    l0 = next(iter(DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)))[0]
    l2 = next(iter(DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)))[0]
    print(f"\n=== Determinismo: mesma época repete: {igual} | época seguinte muda: {muda} | "
          f"0 e 2 workers iguais: {torch.equal(l0, l2)}")

    linhas = []
    for i in range(8):
        a0 = _para_img(sem[i][0])
        ds.definir_epoca(0); e0 = _para_img(ds[i][0])
        ds.definir_epoca(1); e1 = _para_img(ds[i][0])
        linhas.append(np.hstack([a0, e0, e1]))
    destino = f"/out/{a.base}/f5_carregador.png"
    cv2.imwrite(destino, np.vstack([np.hstack(linhas[:4]), np.hstack(linhas[4:])]))
    print(f"\nFigura: {destino} (cada trio: sem aumento | época 0 | época 1)")
    ok = igual and muda and torch.equal(l0, l2) and tuple(x.shape) == (3, 320, 320)
    print("\nRESULTADO:", "carregador aprovado" if ok else "ALGO FALHOU — me mande a saída")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
