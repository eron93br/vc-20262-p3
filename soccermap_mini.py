#!/usr/bin/env python3
"""
DESAFIO EXTRA — SoccerMap "mini" em PyTorch (Fernández & Bornn, 2020; Gomes et al., 2026).

Em vez de resumir o estado do jogo em 6 números (xpass.py), codificamos o campo
como uma IMAGEM de C canais e deixamos uma rede totalmente convolucional (FCN)
produzir a superfície de eXP inteira. A perda é calculada só no pixel de destino
do passe observado — a rede aprende a superfície a partir de um único rótulo.

Diferenças em relação ao artigo (17 canais, grade 104×68):
  - grade reduzida 52×34 (célula de ~2 m) para treinar com poucos passes;
  - 7 canais derivados do nosso rastreamento (sem velocidades da bola).

Atenção: algumas dezenas de passes NÃO bastam para treinar uma FCN. Use este
arquivo juntando as pastas de análise de vários grupos/trechos, e compare
com o baseline e a regressão logística (mesmo protocolo de avaliação).

Uso
---
python soccermap_mini.py --saidas ../saidas/jogo1 ../saidas/jogo2 --epocas 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from campo import COMPRIMENTO, GOL_DIREITA, LARGURA
from metricas import carregar_jogadores
from xpass import estado_no_quadro, normalizar

GW, GH = 52, 34                      # largura × altura da grade
CANAIS = ["atacantes", "defensores", "dist_bola", "dist_gol",
          "cos_bola_gol", "sin_bola_gol", "espaco"]


def tensor_estado(origem, atacantes, defensores) -> np.ndarray:
    """Codifica um instante do jogo em (C, GH, GW). Coordenadas já normalizadas."""
    xs = (np.arange(GW) + 0.5) * COMPRIMENTO / GW
    ys = (np.arange(GH) + 0.5) * LARGURA / GH
    gx, gy = np.meshgrid(xs, ys)
    T = np.zeros((len(CANAIS), GH, GW), np.float32)

    def ocupacao(pts):
        m = np.zeros((GH, GW), np.float32)
        for x, y in pts:
            i, j = int(np.clip(y / LARGURA * GH, 0, GH - 1)), int(np.clip(x / COMPRIMENTO * GW, 0, GW - 1))
            m[i, j] += 1
        return m

    T[0], T[1] = ocupacao(atacantes), ocupacao(defensores)
    T[2] = np.hypot(gx - origem[0], gy - origem[1]) / COMPRIMENTO
    T[3] = np.hypot(gx - GOL_DIREITA[0], gy - GOL_DIREITA[1]) / COMPRIMENTO
    ang = np.arctan2(GOL_DIREITA[1] - gy, GOL_DIREITA[0] - gx) - np.arctan2(origem[1] - gy, origem[0] - gx)
    T[4], T[5] = np.cos(ang), np.sin(ang)
    if len(defensores):
        d = np.min([np.hypot(gx - x, gy - y) for x, y in defensores], axis=0)
        T[6] = np.minimum(d, 10) / 10
    return T


class SoccerMapMini(nn.Module):
    """FCN em 3 escalas (1×, 1/2×, 1/4×) com fusão por upsampling, como na SoccerMap."""

    def __init__(self, c_in: int = len(CANAIS)):
        super().__init__()

        def bloco(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 5, padding=2), nn.ReLU(),
                                 nn.Conv2d(co, co, 5, padding=2), nn.ReLU())

        self.f1, self.f2, self.f3 = bloco(c_in, 32), bloco(32, 64), bloco(64, 64)
        self.p1, self.p2, self.p3 = (nn.Conv2d(c, 1, 1) for c in (32, 64, 64))  # predições por escala
        self.fusao12 = nn.Conv2d(2, 1, 1)
        self.fusao23 = nn.Conv2d(2, 1, 1)

    def forward(self, x):
        h1 = self.f1(x)                               # 1×
        h2 = self.f2(F.max_pool2d(h1, 2))             # 1/2×
        h3 = self.f3(F.max_pool2d(h2, 2))             # 1/4×
        y3 = self.p3(h3)
        y2 = self.fusao23(torch.cat([self.p2(h2), F.interpolate(y3, size=h2.shape[-2:])], 1))
        y1 = self.fusao12(torch.cat([self.p1(h1), F.interpolate(y2, size=h1.shape[-2:])], 1))
        return y1                                     # logits (B, 1, GH, GW)


def montar_dataset(pastas: list[Path]):
    X, idx, y = [], [], []
    for pasta in pastas:
        direcao = json.loads((pasta / "metricas_times.json").read_text())["direcao_ataque"]
        jog = carregar_jogadores(pasta)
        for r in pd.read_csv(pasta / "passes.csv").itertuples():
            ataque = direcao[str(r.time)]
            est = estado_no_quadro(jog, r.frame_ini)
            atk = normalizar(est.loc[est["time"] == r.time, ["x", "y"]].to_numpy(), ataque)
            dfs = normalizar(est.loc[est["time"] == 1 - r.time, ["x", "y"]].to_numpy(), ataque)
            o = normalizar([r.x0, r.y0], ataque)
            d = normalizar([r.x1, r.y1], ataque)
            X.append(tensor_estado(o, atk.reshape(-1, 2), dfs.reshape(-1, 2)))
            i = int(np.clip(d[1] / LARGURA * GH, 0, GH - 1))
            j = int(np.clip(d[0] / COMPRIMENTO * GW, 0, GW - 1))
            idx.append(i * GW + j)
            y.append(r.sucesso)
    return torch.tensor(np.stack(X)), torch.tensor(idx), torch.tensor(y, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser(description="SoccerMap mini — superfície de eXP com FCN")
    ap.add_argument("--saidas", nargs="+", required=True, help="Pastas de análise (vários trechos)")
    ap.add_argument("--epocas", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()

    X, idx, y = montar_dataset([Path(p) for p in a.saidas])
    print(f"{len(y)} passes, {y.mean()*100:.1f}% certos, tensor {tuple(X.shape)}")
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(y), generator=g)
    n_val = max(1, len(y) // 5)
    val, tr = perm[:n_val], perm[n_val:]

    rede = SoccerMapMini()
    opt = torch.optim.Adam(rede.parameters(), lr=a.lr)
    for ep in range(a.epocas):
        rede.train()
        logit = rede(X[tr]).flatten(1).gather(1, idx[tr, None]).squeeze(1)   # só o pixel de destino
        perda = F.binary_cross_entropy_with_logits(logit, y[tr])
        opt.zero_grad()
        perda.backward()
        opt.step()
        rede.eval()
        with torch.no_grad():
            lv = rede(X[val]).flatten(1).gather(1, idx[val, None]).squeeze(1)
            pv = F.binary_cross_entropy_with_logits(lv, y[val])
        if ep % 5 == 0 or ep == a.epocas - 1:
            print(f"época {ep:3d}  perda treino {perda.item():.3f}  perda val {pv.item():.3f}")
    torch.save(rede.state_dict(), "soccermap_mini.pt")
    print("Pesos salvos em soccermap_mini.pt")


if __name__ == "__main__":
    main()
