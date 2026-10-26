#!/usr/bin/env python3
"""
eXP — Expected Pass: probabilidade de um passe ser completado, dado o contexto.

Relação com o artigo de Gomes et al. (2026): lá, a SoccerMap recebe 17 canais
espaciais (Tabela 1) e produz uma SUPERFÍCIE P(sucesso | destino, estado) para
todas as células do campo. Aqui fazemos uma versão "de bolso" com poucas
features geométricas extraídas do nosso rastreamento — as mesmas ideias de
"distância", "ocupação" e "jogadores entre a bola e a célula", porém resumidas
em números e combinadas por uma regressão logística:

    eXP = σ(β0 + β1·distância + β2·adversários_na_linha + β3·pressão_no_passador
              + β4·espaço_no_destino + β5·progressão_x + β6·distância_ao_gol)

Três modelos são comparados, como na Tabela 2 do artigo:
  - baseline   : regressão logística só com a distância do passe;
  - heuristico : coeficientes fixos (chute inicial, não usa os rótulos);
  - logistico  : regressão logística ajustada nos SEUS passes (validação cruzada).

Uso
---
python xpass.py --saida ../saidas/jogo1
python xpass.py --saida ../saidas/jogo1 --extra ../saidas/jogo2/passes_features.csv
python xpass.py --saida ../saidas/jogo1 --frame 812 --passador 7     # superfície específica
python xpass.py --saida ../saidas/jogo1 --what-if

Saídas: passes_features.csv, resultados_xp.csv, jogadores_xp.csv,
        figuras/superficie_xp_*.png, figuras/calibracao_xp.png, what_if.csv
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from campo import COMPRIMENTO, GOL_DIREITA, LARGURA, desenhar_campo
from metricas import carregar_jogadores

FEATURES = ["distancia", "adversarios_na_linha", "pressao_passador",
            "espaco_destino", "progressao_x", "dist_destino_gol"]

# Chute inicial: passes longos, com adversários na linha e passador pressionado
# erram mais; espaço livre no destino ajuda. Ajuste e discuta esses valores!
COEF_HEURISTICO = {"intercepto": 2.5, "distancia": -0.08, "adversarios_na_linha": -0.9,
                   "pressao_passador": -0.4, "espaco_destino": 0.15,
                   "progressao_x": -0.02, "dist_destino_gol": 0.0}


def sigmoide(z):
    return 1.0 / (1.0 + np.exp(-z))


# ── Features ───────────────────────────────────────────────────────────────

def features_destinos(origem, destinos, adversarios, raio_pressao=3.0, cone=0.15):
    """
    Features para um passe da `origem` (2,) para cada ponto de `destinos` (N, 2),
    dados os `adversarios` (M, 2). Coordenadas já normalizadas: ataque → direita.

    adversarios_na_linha: adversários dentro de um "cone de interceptação" ao
    longo do segmento origem→destino; a largura cresce com a distância
    percorrida pela bola (mais tempo para o defensor chegar).
    """
    origem = np.asarray(origem, float)
    destinos = np.atleast_2d(np.asarray(destinos, float))
    adv = np.atleast_2d(np.asarray(adversarios, float)).reshape(-1, 2)

    vet = destinos - origem                                  # (N, 2)
    dist = np.linalg.norm(vet, axis=1)                       # (N,)
    if len(adv):
        rel = adv - origem                                   # (M, 2)
        # projeção escalar de cada adversário em cada segmento, normalizada em [0, 1]
        s = (vet @ rel.T) / np.maximum(dist[:, None] ** 2, 1e-6)          # (N, M)
        proj = origem + s[..., None] * vet[:, None, :]                    # (N, M, 2)
        d_perp = np.linalg.norm(adv[None] - proj, axis=2)                 # (N, M)
        largura = 1.0 + cone * s * dist[:, None]
        na_linha = ((s > 0.05) & (s < 1.0) & (d_perp < largura)).sum(axis=1)
        pressao = (np.linalg.norm(rel, axis=1) < raio_pressao).sum()
        espaco = np.linalg.norm(destinos[:, None] - adv[None], axis=2).min(axis=1)
    else:
        na_linha = np.zeros(len(destinos))
        pressao, espaco = 0, np.full(len(destinos), 30.0)

    return pd.DataFrame({
        "distancia": dist,
        "adversarios_na_linha": na_linha,
        "pressao_passador": np.full(len(destinos), pressao),
        "espaco_destino": np.minimum(espaco, 10.0),          # satura: 10 m já é "livre"
        "progressao_x": destinos[:, 0] - origem[0],
        "dist_destino_gol": np.linalg.norm(destinos - np.array(GOL_DIREITA), axis=1),
    })


def normalizar(xy, ataque: str):
    """Espelha o campo (rotação de 180°) para que o time com a bola ataque à direita."""
    xy = np.asarray(xy, float).copy()
    if ataque == "esquerda":
        xy[..., 0] = COMPRIMENTO - xy[..., 0]
        xy[..., 1] = LARGURA - xy[..., 1]
    return xy


def estado_no_quadro(jog: pd.DataFrame, frame: int, tolerancia: int = 3) -> pd.DataFrame:
    """Posições dos jogadores no quadro (ou no mais próximo dentro da tolerância)."""
    janela = jog[(jog["frame"] - frame).abs() <= tolerancia].copy()
    janela["dif"] = (janela["frame"] - frame).abs()
    return janela.sort_values("dif").drop_duplicates("track_id")


def montar_features(passes, jog, direcao) -> pd.DataFrame:
    linhas = []
    for r in passes.itertuples():
        ataque = direcao[str(r.time)] if str(r.time) in direcao else direcao[r.time]
        est = estado_no_quadro(jog, r.frame_ini)
        adv = normalizar(est.loc[est["time"] == 1 - r.time, ["x", "y"]].to_numpy(), ataque)
        o = normalizar([r.x0, r.y0], ataque)
        d = normalizar([[r.x1, r.y1]], ataque)
        f = features_destinos(o, d, adv).iloc[0].to_dict()
        linhas.append({**r._asdict(), **f})
    return pd.DataFrame(linhas).drop(columns=["Index"], errors="ignore")


# ── Modelos ────────────────────────────────────────────────────────────────

def xp_heuristico(X: pd.DataFrame) -> np.ndarray:
    z = COEF_HEURISTICO["intercepto"] + sum(COEF_HEURISTICO[c] * X[c] for c in FEATURES)
    return sigmoide(np.asarray(z, float))


def avaliar(y, p) -> dict:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return {"AUC-ROC (↑)": roc_auc_score(y, p), "Brier (↓)": brier_score_loss(y, p),
            "Log-Loss (↓)": log_loss(y, p), "Avg Precision (↑)": average_precision_score(y, p),
            "média prevista": p.mean(), "taxa observada": np.mean(y)}


def comparar_modelos(df: pd.DataFrame, pasta: Path):
    y = df["sucesso"].to_numpy()
    n_min = int(min(y.sum(), len(y) - y.sum()))
    res = {"heuristico": xp_heuristico(df)}
    modelo_final = None

    if n_min >= 3:
        cv = StratifiedKFold(n_splits=min(5, n_min), shuffle=True, random_state=42)
        base = make_pipeline(StandardScaler(), LogisticRegression())
        res["baseline (distância)"] = cross_val_predict(base, df[["distancia"]], y, cv=cv,
                                                        method="predict_proba")[:, 1]
        log = make_pipeline(StandardScaler(), LogisticRegression(C=1.0))
        res["logistico"] = cross_val_predict(log, df[FEATURES], y, cv=cv, method="predict_proba")[:, 1]
        modelo_final = log.fit(df[FEATURES], y)
        coefs = dict(zip(FEATURES, modelo_final[-1].coef_[0]))
        print("Coeficientes (features padronizadas):",
              {k: round(v, 3) for k, v in coefs.items()})
    else:
        print(f"Poucos passes de uma das classes ({n_min}): só o modelo heurístico será avaliado.\n"
              "Junte passes de outros trechos/grupos com --extra.")

    tabela = pd.DataFrame({nome: avaliar(y, p) for nome, p in res.items()}).T if len(set(y)) > 1 else None
    if tabela is not None:
        print("\n" + tabela.round(3).to_string())
        tabela.to_csv(pasta / "resultados_xp.csv")
        curva_calibracao(y, res, pasta / "figuras" / "calibracao_xp.png")

    melhor = "logistico" if "logistico" in res else "heuristico"
    df["eXP"] = res[melhor]
    return df, modelo_final


def curva_calibracao(y, res, arq, n_bins=5):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="calibração perfeita")
    for nome, p in res.items():
        bins = np.quantile(p, np.linspace(0, 1, n_bins + 1))
        idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
        mp = [p[idx == b].mean() for b in range(n_bins) if (idx == b).any()]
        oy = [y[idx == b].mean() for b in range(n_bins) if (idx == b).any()]
        ax.plot(mp, oy, "o-", label=nome)
    ax.set_xlabel("eXP médio previsto")
    ax.set_ylabel("taxa de acerto observada")
    ax.set_title("Calibração do eXP")
    ax.legend()
    fig.tight_layout()
    fig.savefig(arq, dpi=130)
    plt.close(fig)


# ── Superfície de eXP (inspirada na Fig. 1 de Gomes et al., 2026) ──────────

def superficie(jog, frame, passador, direcao, modelo, ball_xy, arq, passe=None):
    est = estado_no_quadro(jog, frame)
    linha = est[est["track_id"] == passador]
    if linha.empty:
        print(f"Passador {passador} não encontrado no quadro {frame}.")
        return
    time_ = int(linha["time"].iloc[0])
    ataque = direcao[str(time_)]
    origem = normalizar(ball_xy if ball_xy is not None else linha[["x", "y"]].to_numpy()[0], ataque)
    adv = normalizar(est.loc[est["time"] == 1 - time_, ["x", "y"]].to_numpy(), ataque)

    gx, gy = np.meshgrid(np.arange(0.5, COMPRIMENTO, 1.0), np.arange(0.5, LARGURA, 1.0))
    destinos = np.c_[gx.ravel(), gy.ravel()]
    X = features_destinos(origem, destinos, adv)
    p = modelo.predict_proba(X[FEATURES])[:, 1] if modelo is not None else xp_heuristico(X)
    P = p.reshape(gx.shape)
    if ataque == "esquerda":           # volta para o referencial da câmera
        P = P[::-1, ::-1]

    ax = desenhar_campo(cor_grama="#1f2937")
    im = ax.imshow(P, extent=[0, COMPRIMENTO, LARGURA, 0], cmap="RdYlBu_r", vmin=0, vmax=1, alpha=0.85)
    for t, cor in ((0, "#ef4444"), (1, "#3b82f6")):
        g = est[est["time"] == t]
        ax.scatter(g["x"], g["y"], s=60, c=cor, edgecolors="white", zorder=3, label=f"time {t}")
    ax.scatter(linha["x"], linha["y"], s=140, c="black", edgecolors="white", zorder=4, label="passador")
    if passe is not None:
        ax.annotate("", xy=(passe.x1, passe.y1), xytext=(passe.x0, passe.y0), zorder=5,
                    arrowprops=dict(arrowstyle="->", lw=2.2, color="white"))
    plt.colorbar(im, ax=ax, fraction=0.03, label="eXP (prob. de sucesso)")
    ax.set_title(f"Superfície de eXP — quadro {frame}, passador #{passador} (ataca à {ataque})")
    ax.legend(loc="upper left", fontsize=8)
    ax.figure.savefig(arq, dpi=130, bbox_inches="tight")
    plt.close(ax.figure)
    print(f"Superfície salva em {arq}")


# ── Análise what-if (Seção 3.6 de Gomes et al., 2026) ──────────────────────

def what_if(df, jog, direcao, modelo, raios=(3, 4.5, 6), n_ang=16):
    """Para passes errados: se o destino estivesse 3–6 m ao lado, quanto subiria o eXP?"""
    linhas = []
    ang = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    desloc = np.array([[r * np.cos(a), r * np.sin(a)] for r in raios for a in ang])
    for r in df[df["sucesso"] == 0].itertuples():
        ataque = direcao[str(r.time)]
        est = estado_no_quadro(jog, r.frame_ini)
        adv = normalizar(est.loc[est["time"] == 1 - r.time, ["x", "y"]].to_numpy(), ataque)
        o = normalizar([r.x0, r.y0], ataque)
        d = normalizar([r.x1, r.y1], ataque)
        cand = np.clip(d + desloc, [0, 0], [COMPRIMENTO, LARGURA])
        X = features_destinos(o, np.vstack([d, cand]), adv)
        p = modelo.predict_proba(X[FEATURES])[:, 1] if modelo is not None else xp_heuristico(X)
        k = int(np.argmax(p[1:]))
        linhas.append({"frame_ini": r.frame_ini, "passador": r.passador, "eXP_original": p[0],
                       "eXP_melhor": p[1 + k], "ganho": p[1 + k] - p[0],
                       "dx_m": desloc[k, 0], "dy_m": desloc[k, 1]})
    return pd.DataFrame(linhas)


def main():
    ap = argparse.ArgumentParser(description="eXP — expected pass")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--extra", nargs="*", default=[], help="passes_features.csv de outros trechos")
    ap.add_argument("--frame", type=int, default=None, help="Quadro da superfície")
    ap.add_argument("--passador", type=int, default=None, help="track_id do passador")
    ap.add_argument("--what-if", action="store_true")
    a = ap.parse_args()

    pasta = Path(a.saida)
    (pasta / "figuras").mkdir(exist_ok=True)
    direcao = json.loads((pasta / "metricas_times.json").read_text())["direcao_ataque"]
    jog = carregar_jogadores(pasta)
    passes = pd.read_csv(pasta / "passes.csv")
    if passes.empty:
        raise SystemExit("Nenhum passe em passes.csv — revise metricas.py (raio de posse, bola).")

    feats = montar_features(passes, jog, direcao)
    feats.to_csv(pasta / "passes_features.csv", index=False)
    todos = pd.concat([feats] + [pd.read_csv(f) for f in a.extra], ignore_index=True)
    print(f"{len(todos)} passes ({todos['sucesso'].mean()*100:.1f}% certos) "
          f"— {len(feats)} deste trecho, {len(todos)-len(feats)} extras")

    todos, modelo = comparar_modelos(todos, pasta)
    feats["eXP"] = todos["eXP"].iloc[:len(feats)].to_numpy()

    # Passes acima do esperado (PAx) por jogador: acertos − soma dos eXP
    jx = feats.groupby(["passador", "time"]).agg(passes=("sucesso", "size"), certos=("sucesso", "sum"),
                                                eXP=("eXP", "sum")).reset_index()
    jx["PAx"] = jx["certos"] - jx["eXP"]
    jx.sort_values("PAx", ascending=False).to_csv(pasta / "jogadores_xp.csv", index=False)
    print("\nPasses acima do esperado (PAx):")
    print(jx.sort_values("PAx", ascending=False).round(2).head(8).to_string(index=False))

    # Superfície: quadro/passador pedidos ou o primeiro passe detectado
    if a.frame is not None and a.passador is not None:
        superficie(jog, a.frame, a.passador, direcao, modelo, None,
                   pasta / "figuras" / f"superficie_xp_f{a.frame}.png")
    else:
        r = feats.iloc[0]
        superficie(jog, int(r.frame_ini), int(r.passador), direcao, modelo, (r.x0, r.y0),
                   pasta / "figuras" / f"superficie_xp_f{int(r.frame_ini)}.png", passe=r)

    if a.what_if:
        wi = what_if(feats, jog, direcao, modelo)
        wi.to_csv(pasta / "what_if.csv", index=False)
        if len(wi):
            print(f"\nWhat-if: ganho médio de eXP {wi['ganho'].mean():.3f} "
                  f"({(wi['ganho'] > 0.1).mean()*100:.0f}% dos passes errados com ganho > 0,10)")


if __name__ == "__main__":
    main()
