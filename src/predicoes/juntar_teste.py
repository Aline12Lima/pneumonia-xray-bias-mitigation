"""Junta as predições do teste das três bases no notebook de treino (F6, passo 6.8 → F7 a F11).

Cada notebook de base envia ao dataset ``resultados`` a sua tabela (``src.predicoes.teste --enviar``).
No notebook de treino, depois de ``python -m src.cache.resultados --baixar``, este script:

- lê ``<base>__teste__tabela_teste.csv.gz`` das três bases (e ``plus__teste__tabela_teste_incertos.csv.gz``);
- confere cada base contra ``data/splits/listas_sha256.json``: número de imagens (regra primária e
  incertos), de positivos e de pacientes, e as 15 execuções em cada imagem;
- valida o schema e grava ``/out/predicoes/tabela_teste.csv`` (regra primária, a que as análises leem) e
  ``/out/predicoes/tabela_teste_sensibilidade_incertos.csv`` (a mesma, mais os incertos do Plus com
  rótulo 1: sensibilidade de D-018).

As análises (E0, eixo A, T1/T2, eixo C) rodam depois com ``--tabela /out/predicoes/tabela_teste.csv``,
com o bootstrap padrão (2.000).

Uso (notebook "treino", CPU):
    python -m src.predicoes.juntar_teste
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from src.predicoes.teste import HASHES_LISTAS

BASES = ["plus", "rsna", "padchest"]
FONTE = {"plus": "chexpert_plus", "rsna": "rsna", "padchest": "padchest"}
EXECUCOES = 15


def conferir_base(base: str, tab: pd.DataFrame, inc: pd.DataFrame | None, ref: dict) -> list[str]:
    erros = []
    n_img, n_inc = tab["id_imagem"].nunique(), 0 if inc is None else inc["id_imagem"].nunique()
    esperado_inc = ref.get("incertos", 0)
    if n_img != ref["imagens"] - esperado_inc:
        erros.append(f"{base}: {n_img} imagens na regra primária, esperado {ref['imagens'] - esperado_inc}")
    if n_inc != esperado_inc:
        erros.append(f"{base}: {n_inc} incertas, esperado {esperado_inc}")
    pos = int(tab.drop_duplicates("id_imagem")["rotulo"].sum())
    if pos != ref["positivos"]:
        erros.append(f"{base}: {pos} positivos, esperado {ref['positivos']}")
    pacs = pd.concat([tab["id_paciente"], inc["id_paciente"] if inc is not None else pd.Series(dtype=str)]).nunique()
    if pacs != ref["pacientes"]:
        erros.append(f"{base}: {pacs} pacientes, esperado {ref['pacientes']}")
    por_img = tab.groupby("id_imagem")["id_execucao"].nunique()
    if tab["id_execucao"].nunique() != EXECUCOES or (por_img != EXECUCOES).any():
        erros.append(f"{base}: nem toda imagem tem as {EXECUCOES} execuções")
    if set(tab["fonte"]) != {FONTE[base]} or set(tab["particao"]) != {"teste"}:
        erros.append(f"{base}: fonte ou partição inesperada ({set(tab['fonte'])}, {set(tab['particao'])})")
    return erros


def juntar(pasta: str = "/out/resultados", hashes_listas: str = HASHES_LISTAS) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    from src.predicoes.schema import validar
    ref = json.load(open(hashes_listas, encoding="utf-8"))["bases"]
    tipos = {"id_imagem": str, "id_paciente": str, "id_estudo": str, "fabricante": "object", "nivel": str}
    tabs, incs, erros = [], [], []
    for b in BASES:
        arq = os.path.join(pasta, f"{b}__teste__tabela_teste.csv.gz")
        if not os.path.exists(arq):
            erros.append(f"{b}: {os.path.basename(arq)} não está em {pasta} (rode --enviar no notebook da base)")
            continue
        t = pd.read_csv(arq, dtype=tipos)
        arq_inc = os.path.join(pasta, f"{b}__teste__tabela_teste_incertos.csv.gz")
        inc = pd.read_csv(arq_inc, dtype=tipos) if os.path.exists(arq_inc) else None
        erros += conferir_base(b, t, inc, ref[b]["listas"]["teste"])
        tabs.append(t)
        if inc is not None:
            incs.append(inc)
    if not tabs:
        return pd.DataFrame(), pd.DataFrame(), erros
    tab = pd.concat(tabs, ignore_index=True)
    sens = pd.concat([tab] + incs, ignore_index=True)
    erros += [f"schema (primária): {e}" for e in validar(tab)]
    erros += [f"schema (sensibilidade): {e}" for e in validar(sens)]
    return tab, sens, erros


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resultados", default="/out/resultados")
    ap.add_argument("--saida", default="/out/predicoes")
    a = ap.parse_args(argv)
    tab, sens, erros = juntar(a.resultados)
    if erros:
        print("Problemas:\n- " + "\n- ".join(erros))
        print("RESULTADO: NÃO JUNTADO")
        return 1
    os.makedirs(a.saida, exist_ok=True)
    tab.to_csv(os.path.join(a.saida, "tabela_teste.csv"), index=False)
    sens.to_csv(os.path.join(a.saida, "tabela_teste_sensibilidade_incertos.csv"), index=False)
    for f, g in tab.groupby("fonte"):
        u = g.drop_duplicates("id_imagem")
        print(f"  {f:13s} | {len(u)} imagens, {int(u['rotulo'].sum())} positivos, {u['id_paciente'].nunique()} pacientes | "
              f"{g['id_execucao'].nunique()} execuções")
    print(f"Primária: {len(tab)} linhas | sensibilidade (com incertos do Plus = 1): {len(sens)} linhas")
    print(f"RESULTADO: OK (salvo em {a.saida}/tabela_teste.csv e tabela_teste_sensibilidade_incertos.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
