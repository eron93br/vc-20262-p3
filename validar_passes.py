#!/usr/bin/env python3
"""
Validação dos passes detectados contra uma anotação manual (ground truth).

Assista ao trecho e anote os passes em um CSV (passes_anotados.csv):

    t_s,time,sucesso
    3.4,0,1
    7.9,0,0
    ...

t_s = instante do passe em segundos, contado a partir do INÍCIO DO CLIPE
processado (o mesmo relógio do vídeo rastreado.mp4); time = 0/1 conforme as
cores do rastreado.mp4 (T0 = vermelho, T1 = azul); sucesso = 1 se completado.

Um passe detectado "casa" com um anotado se estiver a até --tolerancia segundos
e for do mesmo time. Reporta precisão, revocação e F1 da detecção de passes.

Uso
---
python validar_passes.py --saida ../saidas/jogo1 --anotado passes_anotados.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def casar(det: pd.DataFrame, gt: pd.DataFrame, tol: float):
    """Casamento guloso por proximidade temporal (cada anotação casa no máx. uma vez)."""
    usados, pares = set(), []
    for i, d in det.sort_values("t_ini_s").iterrows():
        cand = gt[(gt["time"] == d["time"]) & ((gt["t_s"] - d["t_ini_s"]).abs() <= tol)]
        cand = cand[~cand.index.isin(usados)]
        if len(cand):
            j = (cand["t_s"] - d["t_ini_s"]).abs().idxmin()
            usados.add(j)
            pares.append((i, j))
    return pares


def main():
    ap = argparse.ArgumentParser(description="Precisão/revocação da detecção de passes")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--anotado", required=True)
    ap.add_argument("--tolerancia", type=float, default=1.0, help="Segundos")
    a = ap.parse_args()

    det = pd.read_csv(Path(a.saida) / "passes.csv")
    gt = pd.read_csv(a.anotado)
    pares = casar(det, gt, a.tolerancia)

    vp = len(pares)
    precisao = vp / len(det) if len(det) else 0.0
    revocacao = vp / len(gt) if len(gt) else 0.0
    f1 = 2 * precisao * revocacao / (precisao + revocacao) if vp else 0.0
    acordo = sum(det.loc[i, "sucesso"] == gt.loc[j, "sucesso"] for i, j in pares) / vp if vp else 0.0

    print(f"Detectados: {len(det)} | Anotados: {len(gt)} | Casados: {vp}")
    print(f"Precisão  : {precisao:.2f}")
    print(f"Revocação : {revocacao:.2f}")
    print(f"F1        : {f1:.2f}")
    print(f"Acordo certo/errado nos casados: {acordo:.2f}")


if __name__ == "__main__":
    main()
