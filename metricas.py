#!/usr/bin/env python3
"""
Métricas de jogo a partir das trajetórias no campo (metros).

Calcula:
  1. distância percorrida e velocidade por jogador (track_id);
  2. posse de bola por proximidade jogador–bola (Ahmad Yousef et al., 2025):
     o dono da bola é o jogador mais próximo, se estiver a menos de --raio-posse;
  3. passes: troca de dono entre jogadores → certo (mesmo time) ou errado (outro time);
  4. figuras: mapa de calor por time, linha do tempo de posse, mapa de passes.

Uso
---
python metricas.py --saida ../saidas/jogo1
python metricas.py --saida ../saidas/jogo1 --raio-posse 2.5 --ataque-time0 esquerda
python metricas.py --saida ../saidas/jogo1 --pixels      # sem homografia (aproximado!)

Entradas : tracks_campo.csv, bola_campo.csv, meta.json
Saídas   : metricas_jogadores.csv, posse_por_quadro.csv, passes.csv,
           metricas_times.json, figuras/*.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from campo import COMPRIMENTO, LARGURA, desenhar_campo

CORES_TIME = {0: "#d62828", 1: "#1d4ed8"}
VEL_MAX_HUMANA = 12.0  # m/s — acima disso é erro de rastreamento/projeção


# ── Carregamento ───────────────────────────────────────────────────────────

def carregar_jogadores(pasta: Path, pixels: bool = False, janela: int = 5) -> pd.DataFrame:
    """Lê as trajetórias, define o time de cada track por votação e suaviza posições."""
    if pixels:
        df = pd.read_csv(pasta / "tracks.csv")
        df = df[df["classe"] != "bola"].rename(columns={"px": "x", "py": "y"})
    else:
        df = pd.read_csv(pasta / "tracks_campo.csv").rename(columns={"x_m": "x", "y_m": "y"})
    df = df.dropna(subset=["x", "y"])

    # Time do track = moda das classificações válidas (ignora -1 do período de calibração)
    validos = df[df["time"] != -1]
    moda = validos.groupby("track_id")["time"].agg(lambda s: s.value_counts().idxmax())
    df["time"] = df["track_id"].map(moda)
    df = df.dropna(subset=["time"])
    df["time"] = df["time"].astype(int)

    # Mediana móvel por track reduz o "tremor" do bounding box
    df = df.sort_values(["track_id", "frame"])
    for c in ("x", "y"):
        df[c] = df.groupby("track_id")[c].transform(
            lambda s: s.rolling(janela, center=True, min_periods=1).median())
    return df


def carregar_bola(pasta: Path, pixels: bool = False) -> pd.DataFrame:
    if pixels:
        b = pd.read_csv(pasta / "bola.csv").rename(columns={"px": "x", "py": "y"})
    else:
        b = pd.read_csv(pasta / "bola_campo.csv").rename(columns={"x_m": "x", "y_m": "y"})
    return b.dropna(subset=["x", "y"])[["frame", "x", "y", "fonte"]]


# ── 1. Distância e velocidade ──────────────────────────────────────────────

def metricas_fisicas(jog: pd.DataFrame, fps: float, pixels: bool) -> pd.DataFrame:
    linhas = []
    for tid, g in jog[jog["time"] >= 0].groupby("track_id"):
        g = g.sort_values("frame")
        df_ = g["frame"].diff()
        passo = np.hypot(g["x"].diff(), g["y"].diff())
        dt = df_ / fps
        vel = passo / dt
        ok = (df_ <= 3) & (pixels | (vel <= VEL_MAX_HUMANA))   # ignora saltos e lacunas longas
        vel_suave = vel.where(ok).rolling(max(int(fps), 1), min_periods=3).mean()
        linhas.append({
            "track_id": tid, "time": int(g["time"].iloc[0]), "quadros": len(g),
            "tempo_s": len(g) / fps,
            "distancia": passo[ok].sum(),
            "vel_media": passo[ok].sum() / max(dt[ok].sum(), 1e-6),
            "vel_max": vel_suave.max(),
        })
    m = pd.DataFrame(linhas)
    if not pixels and len(m):
        m["vel_max_kmh"] = m["vel_max"] * 3.6
    return m.sort_values("distancia", ascending=False)


# ── 2. Posse de bola ───────────────────────────────────────────────────────

def posse_por_quadro(jog, bola, raio, min_quadros) -> pd.DataFrame:
    """Para cada quadro com bola: jogador mais próximo e, se perto, o dono da bola."""
    jt = jog[jog["time"] >= 0][["frame", "track_id", "time", "x", "y"]]
    m = jt.merge(bola[["frame", "x", "y"]], on="frame", suffixes=("", "_bola"))
    m["dist"] = np.hypot(m["x"] - m["x_bola"], m["y"] - m["y_bola"])
    perto = m.loc[m.groupby("frame")["dist"].idxmin()].set_index("frame")

    p = bola.set_index("frame")[["x", "y"]].rename(columns={"x": "x_bola", "y": "y_bola"})
    p = p.join(perto[["track_id", "time", "dist"]])
    p["dono"] = p["track_id"].where(p["dist"] < raio)
    p["time_dono"] = p["time"].where(p["dist"] < raio)

    # Remove "posses" muito curtas (bola passando perto de um jogador)
    run = (p["dono"] != p["dono"].shift()).cumsum()
    tam = p.groupby(run)["dono"].transform("size")
    curto = tam < min_quadros
    p.loc[curto, ["dono", "time_dono"]] = np.nan

    # Convenção usual: a posse continua com o time até o adversário tocar na bola
    p["time_posse"] = p["time_dono"].ffill()
    return p.reset_index()


def segmentos_posse(p: pd.DataFrame) -> pd.DataFrame:
    """Agrupa quadros consecutivos com o mesmo dono em segmentos (toques)."""
    d = p.dropna(subset=["dono"]).copy()
    d["seg"] = ((d["dono"] != d["dono"].shift()) | (d["frame"].diff() > 1)).cumsum()
    seg = d.groupby("seg").agg(dono=("dono", "first"), time=("time_dono", "first"),
                               f_ini=("frame", "min"), f_fim=("frame", "max"))
    # Une segmentos do mesmo dono separados por lacunas (bola oculta, condução)
    grupo = (seg["dono"] != seg["dono"].shift()).cumsum()
    seg = seg.groupby(grupo).agg(dono=("dono", "first"), time=("time", "first"),
                                 f_ini=("f_ini", "min"), f_fim=("f_fim", "max"))
    return seg.reset_index(drop=True)


# ── 3. Passes ──────────────────────────────────────────────────────────────

def detectar_passes(p, jog, fps, dist_min, dur_max) -> pd.DataFrame:
    seg = segmentos_posse(p)
    bola = p.set_index("frame")[["x_bola", "y_bola"]]
    linhas = []
    for a, b in zip(seg.itertuples(), seg.iloc[1:].itertuples()):
        duracao = (b.f_ini - a.f_fim) / fps
        x0, y0 = bola.loc[a.f_fim]
        x1, y1 = bola.loc[b.f_ini]
        dist = float(np.hypot(x1 - x0, y1 - y0))
        # Filtros: a bola precisa se deslocar (evita troca de ID do mesmo jogador)
        # e o passe não pode demorar demais (provavelmente perdemos a bola no meio)
        if dist < dist_min or duracao > dur_max:
            continue
        sucesso = int(a.time == b.time)
        receptor = int(b.dono)
        if not sucesso:
            # Receptor pretendido ≈ companheiro do passador mais próximo do ponto de interceptação
            comp = jog[(jog["frame"] == b.f_ini) & (jog["time"] == a.time) & (jog["track_id"] != a.dono)]
            if len(comp):
                receptor = int(comp.iloc[np.argmin(np.hypot(comp["x"] - x1, comp["y"] - y1))]["track_id"])
            else:
                receptor = -1
        linhas.append({
            "frame_ini": int(a.f_fim), "frame_fim": int(b.f_ini), "t_ini_s": a.f_fim / fps,
            "passador": int(a.dono), "time": int(a.time),
            "receptor": receptor, "tocou_em": int(b.dono), "time_tocou": int(b.time),
            "sucesso": sucesso, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "distancia": dist, "duracao_s": duracao,
        })
    return pd.DataFrame(linhas)


def direcao_ataque(jog: pd.DataFrame, forcar_time0: str | None) -> dict[int, str]:
    """Heurística: quem defende um lado tem o jogador mais extremo daquele lado (goleiro/última linha).

    A média de x engana: durante um ataque, os dois times estão no mesmo meio-campo.
    Comparamos os quantis extremos de x de cada time; SEMPRE confira no vídeo e,
    se preciso, force com --ataque-time0.
    """
    if forcar_time0:
        outro = "esquerda" if forcar_time0 == "direita" else "direita"
        return {0: forcar_time0, 1: outro}
    g = jog[jog["time"] >= 0].groupby("time")["x"]
    alto, baixo = g.quantile(0.98), g.quantile(0.02)
    # > 0: o time 0 chega mais perto do gol da direita e menos do gol da esquerda → defende a direita
    escore = (alto.get(0, 0) - alto.get(1, 0)) + (baixo.get(0, 0) - baixo.get(1, 0))
    ataque0 = "esquerda" if escore > 0 else "direita"
    return {0: ataque0, 1: "direita" if ataque0 == "esquerda" else "esquerda"}


# ── 4. Figuras ─────────────────────────────────────────────────────────────

def figuras(pasta, jog, p, passes, fps, pixels):
    fig_dir = pasta / "figuras"
    fig_dir.mkdir(exist_ok=True)

    if not pixels:
        fig, axs = plt.subplots(1, 2, figsize=(16, 5.5))
        for t, ax in zip((0, 1), axs):
            desenhar_campo(ax)
            g = jog[jog["time"] == t]
            ax.hist2d(g["x"], g["y"], bins=[42, 27], range=[[0, COMPRIMENTO], [0, LARGURA]],
                      cmap="magma", alpha=0.75, cmin=1)
            ax.set_title(f"Mapa de calor — time {t}")
        fig.tight_layout()
        fig.savefig(fig_dir / "mapa_calor_times.png", dpi=130)
        plt.close(fig)

        if len(passes):
            ax = desenhar_campo()
            for r in passes.itertuples():
                ax.annotate("", xy=(r.x1, r.y1), xytext=(r.x0, r.y0),
                            arrowprops=dict(arrowstyle="->", lw=1.6,
                                            color="#22c55e" if r.sucesso else "#ef4444"))
            ax.set_title("Passes detectados (verde = certo, vermelho = errado)")
            ax.figure.savefig(fig_dir / "mapa_passes.png", dpi=130, bbox_inches="tight")
            plt.close(ax.figure)

    fig, ax = plt.subplots(figsize=(12, 2.2))
    t = p["frame"] / fps
    for time_, cor in CORES_TIME.items():
        ax.fill_between(t, 0, 1, where=p["time_posse"] == time_, color=cor, step="mid",
                        label=f"time {time_}")
    ax.set_yticks([])
    ax.set_xlabel("tempo (s)")
    ax.set_title("Linha do tempo de posse")
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "posse_linha_tempo.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Métricas de jogo: físicas, posse e passes")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--raio-posse", type=float, default=None,
                    help="Distância máx. jogador–bola para posse (padrão 2 m ou 35 px)")
    ap.add_argument("--min-quadros", type=int, default=3, help="Duração mínima de um toque")
    ap.add_argument("--dist-min-passe", type=float, default=None, help="Padrão 3 m ou 40 px")
    ap.add_argument("--dur-max-passe", type=float, default=4.0, help="Segundos")
    ap.add_argument("--ataque-time0", choices=["direita", "esquerda"], default=None)
    ap.add_argument("--pixels", action="store_true", help="Usa pixels (sem homografia)")
    a = ap.parse_args()

    pasta = Path(a.saida)
    meta = json.loads((pasta / "meta.json").read_text())
    fps = meta["fps"]
    raio = a.raio_posse or (35.0 if a.pixels else 2.0)
    dist_min = a.dist_min_passe or (40.0 if a.pixels else 3.0)
    un = "px" if a.pixels else "m"

    jog = carregar_jogadores(pasta, a.pixels)
    bola = carregar_bola(pasta, a.pixels)

    fis = metricas_fisicas(jog, fps, a.pixels)
    fis.to_csv(pasta / "metricas_jogadores.csv", index=False)

    p = posse_por_quadro(jog, bola, raio, a.min_quadros)
    p.to_csv(pasta / "posse_por_quadro.csv", index=False)

    passes = detectar_passes(p, jog, fps, dist_min, a.dur_max_passe)
    passes.to_csv(pasta / "passes.csv", index=False)

    posse = p["time_posse"].value_counts(normalize=True)
    resumo = {"unidade": un, "raio_posse": raio, "direcao_ataque": direcao_ataque(jog, a.ataque_time0),
              "quadros_com_bola": int(len(p)), "quadros_total": meta["n_frames"]}
    for t in (0, 1):
        pt = passes[passes["time"] == t] if len(passes) else passes
        resumo[f"time{t}"] = {
            "posse_pct": round(100 * float(posse.get(float(t), 0.0)), 1),
            "passes": int(len(pt)),
            "passes_certos": int(pt["sucesso"].sum()) if len(pt) else 0,
            "precisao_pct": round(100 * float(pt["sucesso"].mean()), 1) if len(pt) else None,
            "distancia_total": round(float(fis.loc[fis["time"] == t, "distancia"].sum()), 1),
        }
    (pasta / "metricas_times.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False))
    figuras(pasta, jog, p, passes, fps, a.pixels)

    print(json.dumps(resumo, indent=2, ensure_ascii=False))
    if not a.ataque_time0:
        print(f"\nATENÇÃO: direção de ataque estimada automaticamente ({resumo['direcao_ataque']}).\n"
              "  Confira no rastreado.mp4 (goleiros mal classificados enganam a heurística) e,\n"
              "  se estiver errada, rode de novo com --ataque-time0 direita|esquerda.")
    print(f"\nTop 5 distância ({un}):")
    print(fis.head().to_string(index=False))
    print(f"\n{len(passes)} passes → {pasta/'passes.csv'}  |  figuras em {pasta/'figuras'}/")


if __name__ == "__main__":
    main()
