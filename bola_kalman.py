#!/usr/bin/env python3
"""
Suavização e preenchimento da trajetória da bola com filtro de Kalman.

Motivação (Ahmad Yousef et al., 2025): a bola é pequena, rápida e frequentemente
oculta; sem um modelo de movimento, o F1 da bola ficou em ~0,31, e com o filtro
de Kalman subiu para ~0,96. Aqui usamos uma versão didática: modelo de
velocidade constante em pixels, com
  - PREDIÇÃO nos quadros sem detecção (até --max-lacuna quadros);
  - GATING: detecções muito longe da predição são tratadas como falso positivo
    (ex.: cabeça de jogador, marca do pênalti, bola reserva na lateral).

Estado x = [px, py, vx, vy]ᵀ   (posição e velocidade em pixels/quadro)
Medição z = [px, py]ᵀ

Uso
---
python bola_kalman.py --saida ../saidas/jogo1

Entrada : <saida>/tracks.csv (linhas com classe == "bola"; até 5 candidatas por quadro)
Saída   : <saida>/bola.csv  colunas frame, px, py, fonte  (detectada | kalman)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


class KalmanBola:
    """Filtro de Kalman linear com modelo de velocidade constante."""

    def __init__(self, ruido_processo: float = 1.0, ruido_medicao: float = 4.0):
        dt = 1.0
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=float)    # transição de estado
        self.Hm = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]], dtype=float)   # o que medimos (só posição)
        self.Q = np.eye(4) * ruido_processo               # incerteza do modelo de movimento
        self.R = np.eye(2) * ruido_medicao ** 2           # incerteza da detecção (pixels²)
        self.x = None
        self.P = np.eye(4) * 500.0

    def iniciar(self, px: float, py: float):
        self.x = np.array([px, py, 0.0, 0.0])
        self.P = np.eye(4) * 500.0

    def prever(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2]

    def distancia_mahalanobis(self, z: np.ndarray) -> float:
        """Quão 'surpreendente' é a medição z dada a predição atual."""
        S = self.Hm @ self.P @ self.Hm.T + self.R
        y = z - self.Hm @ self.x
        return float(np.sqrt(y @ np.linalg.inv(S) @ y))

    def corrigir(self, z: np.ndarray):
        S = self.Hm @ self.P @ self.Hm.T + self.R
        K = self.P @ self.Hm.T @ np.linalg.inv(S)          # ganho de Kalman
        self.x = self.x + K @ (z - self.Hm @ self.x)
        self.P = (np.eye(4) - K @ self.Hm) @ self.P


def filtrar_bola(det: pd.DataFrame, n_frames: int, max_lacuna: int, gate: float) -> pd.DataFrame:
    """det: DataFrame com frame, px, py, conf (pode haver vários candidatos por quadro)."""
    por_frame = {f: g.sort_values("conf", ascending=False)[["px", "py"]].to_numpy(float)
                 for f, g in det.groupby("frame")}
    kf = KalmanBola()
    linhas, lacuna = [], 0

    for f in range(n_frames):
        cand = por_frame.get(f)

        if kf.x is None:                       # ainda não vimos a bola
            if cand is not None:
                kf.iniciar(*cand[0])           # começa pelo candidato mais confiável
                linhas.append((f, *cand[0], "detectada"))
            continue

        pred = kf.prever()
        # Associação: candidato mais coerente com a predição (menor Mahalanobis)
        z, dist = None, np.inf
        if cand is not None:
            d = [kf.distancia_mahalanobis(c) for c in cand]
            k = int(np.argmin(d))
            z, dist = cand[k], d[k]

        # Gating: após uma lacuna a incerteza cresce e o gate fica naturalmente mais largo
        if z is not None and dist < gate:
            kf.corrigir(z)
            lacuna = 0
            linhas.append((f, kf.x[0], kf.x[1], "detectada"))
        else:
            lacuna += 1
            if lacuna <= max_lacuna:
                linhas.append((f, pred[0], pred[1], "kalman"))
            elif cand is not None:             # perdemos a bola: reinicia no melhor candidato
                kf.iniciar(*cand[0])
                lacuna = 0
                linhas.append((f, *cand[0], "detectada"))

    return pd.DataFrame(linhas, columns=["frame", "px", "py", "fonte"])


def main():
    p = argparse.ArgumentParser(description="Filtro de Kalman para a trajetória da bola")
    p.add_argument("--saida", required=True, help="Pasta com tracks.csv e meta.json")
    p.add_argument("--max-lacuna", type=int, default=15, help="Quadros máximos preditos sem detecção")
    p.add_argument("--gate", type=float, default=6.0, help="Limiar de Mahalanobis para aceitar detecção")
    a = p.parse_args()

    pasta = Path(a.saida)
    meta = json.loads((pasta / "meta.json").read_text())
    tracks = pd.read_csv(pasta / "tracks.csv")
    det = tracks[tracks["classe"] == "bola"][["frame", "px", "py", "conf"]]

    bola = filtrar_bola(det, meta["n_frames"], a.max_lacuna, a.gate)
    bola.to_csv(pasta / "bola.csv", index=False)

    n = meta["n_frames"]
    n_det = (bola["fonte"] == "detectada").sum()
    n_kal = (bola["fonte"] == "kalman").sum()
    n_q = det["frame"].nunique()
    print(f"Bola detectada pelo YOLO : {n_q:5d} / {n} quadros ({100*n_q/n:.1f}%), {len(det)} candidatas")
    print(f"Aceitas após gating      : {n_det:5d}")
    print(f"Preenchidas pelo Kalman  : {n_kal:5d}")
    print(f"Cobertura final          : {100*(n_det+n_kal)/n:.1f}%  →  {pasta/'bola.csv'}")


if __name__ == "__main__":
    main()
