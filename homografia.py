#!/usr/bin/env python3
"""
Do pixel ao campo: homografia + compensação do movimento da câmera.

Por que isto é necessário? (Ahmad Yousef et al., 2025, Fig. 8)
A câmera lateral introduz distorção de perspectiva: 1 pixel perto da câmera vale
poucos centímetros, 1 pixel do outro lado do campo vale quase meio metro.
Distâncias, velocidades e o eXP precisam ser calculados em METROS.

Ideia
-----
1. Calibração (uma vez, no frame0.jpg): clicamos ≥ 4 marcos conhecidos do campo
   (ver campo.MARCOS) e estimamos H0 tal que  [x_m, y_m, 1]ᵀ ~ H0 · [u, v, 1]ᵀ.
2. Câmera: a transmissão faz pan/zoom. Estimamos a homografia G_t que leva o
   quadro t ao quadro 0 (ORB + RANSAC só no gramado, sem jogadores e placar),
   casando cada quadro com um quadro-chave para reduzir o acúmulo de erro.
3. Projeção: ponto do quadro t vai ao campo por  H0 · G_t · [u, v, 1]ᵀ.

Subcomandos
-----------
python homografia.py calibrar --saida ../saidas/jogo1 --pontos meio_sup,centro,meio_inf,ga_dir_sup,ga_dir_inf
python homografia.py camera   --saida ../saidas/jogo1
python homografia.py projetar --saida ../saidas/jogo1

Sem interface gráfica (ex.: Colab)? Crie <saida>/calibracao.json à mão com
{"pixels": [[u, v], ...], "marcos": ["meio_sup", ...]} e rode "calibrar --sem-janela".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from campo import COMPRIMENTO, LARGURA, MARCOS


# ── Utilitário ─────────────────────────────────────────────────────────────

def aplicar_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Aplica a homografia H (3×3) a pontos (N×2) em coordenadas cartesianas."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


# ── 1. Calibração ──────────────────────────────────────────────────────────

def _clicar_pontos(img: np.ndarray, marcos: list[str]) -> list[list[float]]:
    """Janela OpenCV: clique nos marcos na ordem pedida. 'u' desfaz, 'q' termina."""
    pts: list[list[float]] = []
    janela = "Calibracao - clique nos marcos (u = desfazer, q = sair)"

    def ao_clicar(evento, x, y, *_):
        if evento == cv2.EVENT_LBUTTONDOWN and len(pts) < len(marcos):
            pts.append([float(x), float(y)])

    cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(janela, ao_clicar)
    while True:
        vis = img.copy()
        for (x, y), nome in zip(pts, marcos):
            cv2.circle(vis, (int(x), int(y)), 6, (0, 0, 255), -1)
            cv2.putText(vis, nome, (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        prox = marcos[len(pts)] if len(pts) < len(marcos) else "pronto! tecle q"
        cv2.putText(vis, f"Proximo: {prox}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.imshow(janela, vis)
        tecla = cv2.waitKey(20) & 0xFF
        if tecla == ord("u") and pts:
            pts.pop()
        elif tecla == ord("q"):
            break
    cv2.destroyAllWindows()
    return pts


def calibrar(pasta: Path, marcos: list[str] | None, sem_janela: bool):
    arq = pasta / "calibracao.json"
    if sem_janela:
        cal = json.loads(arq.read_text())
        pixels, marcos = cal["pixels"], cal["marcos"]
    else:
        desconhecidos = [m for m in marcos if m not in MARCOS]
        if desconhecidos:
            raise SystemExit(f"Marcos desconhecidos: {desconhecidos}. Opções: {list(MARCOS)}")
        pixels = _clicar_pontos(cv2.imread(str(pasta / "frame0.jpg")), marcos)
        marcos = marcos[:len(pixels)]

    if len(pixels) < 4:
        raise SystemExit("São necessários pelo menos 4 pontos (não colineares).")

    src = np.array(pixels, dtype=np.float64)
    dst = np.array([MARCOS[m] for m in marcos], dtype=np.float64)
    H0, _ = cv2.findHomography(src, dst, cv2.RANSAC if len(src) > 4 else 0, 1.0)

    # Erro de reprojeção: quão bem H0 leva cada clique ao seu marco (em metros)
    erro = np.linalg.norm(aplicar_h(H0, src) - dst, axis=1)
    json.dump({"pixels": pixels, "marcos": marcos, "H0": H0.tolist(),
               "erro_reprojecao_m": erro.tolist()}, open(arq, "w"), indent=2)
    print(f"H0 salva em {arq}")
    for m, e in zip(marcos, erro):
        print(f"  {m:18s} erro = {e:.2f} m")

    # Imagem de conferência: grade do campo reprojetada sobre o frame0
    img = cv2.imread(str(pasta / "frame0.jpg"))
    Hinv = np.linalg.inv(H0)
    linhas = [[[x, 0], [x, LARGURA]] for x in np.arange(0, COMPRIMENTO + 0.1, 5)]
    linhas += [[[0, y], [COMPRIMENTO, y]] for y in np.arange(0, LARGURA + 0.1, 5)]
    for seg in linhas:
        # clip evita overflow quando a linha projetada cai muito fora da imagem
        a, b = np.clip(aplicar_h(Hinv, seg), -1e4, 1e4).astype(int)
        cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), (0, 255, 255), 1)
    cv2.imwrite(str(pasta / "calibracao_grade.jpg"), img)
    print(f"Confira a grade de 5 m em {pasta/'calibracao_grade.jpg'}")


# ── 2. Movimento da câmera ─────────────────────────────────────────────────

def _mascara(frame: np.ndarray, caixas: np.ndarray, margem_placar: float) -> np.ndarray:
    """Máscara dos pixels úteis para estimar o movimento da câmera.

    Mantém só o GRAMADO (com as linhas): a torcida se mexe e os painéis de LED
    são animados, o que engana o RANSAC e faz a homografia "derivar".
    Remove também os jogadores (se movem por conta própria) e o placar.
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    verde = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))
    # Fechamento morfológico: inclui as linhas brancas que cortam o gramado
    verde = cv2.morphologyEx(verde, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    verde = cv2.erode(verde, np.ones((15, 15), np.uint8))   # afasta da borda com as placas
    verde[: int(h * margem_placar), :] = 0
    for x1, y1, x2, y2 in caixas.astype(int):
        verde[max(y1 - 5, 0):y2 + 5, max(x1 - 5, 0):x2 + 5] = 0
    return verde


def movimento_camera(pasta: Path, limiar_corte: float, margem_placar: float, min_inliers: int = 150):
    """Estima G_t (quadro t → quadro 0) casando cada quadro com um QUADRO-CHAVE.

    Encadear quadro a quadro (G_t = G_{t-1}·M_t) soma o erro de centenas de
    estimativas. Casando com um quadro-chave, o erro só se acumula quando o
    quadro-chave é trocado (quando a câmera andou tanto que sobram poucos inliers).
    """
    meta = json.loads((pasta / "meta.json").read_text())
    tracks = pd.read_csv(pasta / "tracks.csv")
    caixas_por_frame = {f: g[["x1", "y1", "x2", "y2"]].to_numpy()
                        for f, g in tracks.groupby("frame")}

    cap = cv2.VideoCapture(meta["video"])
    cap.set(cv2.CAP_PROP_POS_MSEC, meta["inicio_s"] * 1000)
    orb = cv2.ORB_create(nfeatures=3000, fastThreshold=10)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    n = meta["n_frames"]
    G = np.zeros((n, 3, 3))
    cortes, falhas, n_chaves = [], 0, 1
    chave = None          # (keypoints, descritores, G do quadro-chave)
    ant = None            # (keypoints, descritores, G) do quadro anterior
    ant_hist = None

    for f in tqdm(range(n), desc="Câmera"):
        ok, frame = cap.read()
        if not ok:
            G = G[:f]
            break
        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = _mascara(frame, caixas_por_frame.get(f, np.empty((0, 4))), margem_placar)
        kp, des = orb.detectAndCompute(cinza, mask)

        # Detecção de corte de câmera: queda brusca na correlação dos histogramas
        hist = cv2.calcHist([cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        if f == 0:
            G[0] = np.eye(3)
            chave = ant = (kp, des, G[0])
            ant_hist = hist
            continue
        if cv2.compareHist(ant_hist, hist, cv2.HISTCMP_CORREL) < limiar_corte:
            cortes.append(f)
        ant_hist = hist

        M, inliers = _casar(kp, des, chave[0], chave[1], bf)
        if M is not None and inliers >= min_inliers:
            G[f] = chave[2] @ M
        else:
            # Poucos inliers com o quadro-chave: casa com o quadro anterior
            # e transforma o anterior no novo quadro-chave
            M, inliers = _casar(kp, des, ant[0], ant[1], bf)
            if M is None:
                falhas += 1
                M = np.eye(3)
            G[f] = ant[2] @ M
            chave = ant
            n_chaves += 1
        ant = (kp, des, G[f])
    cap.release()

    np.save(pasta / "camera.npy", G)
    print(f"Homografias salvas em {pasta/'camera.npy'} ({len(G)} quadros, "
          f"{n_chaves} quadros-chave, {falhas} falhas)")
    if cortes:
        t = [f"{c / meta['fps']:.1f}s" for c in cortes]
        print(f"ATENÇÃO: {len(cortes)} possível(is) corte(s) de câmera em {t}.")
        print("  Após um corte a projeção deixa de valer: use um trecho sem cortes ou")
        print("  divida o vídeo em segmentos e calibre cada um separadamente.")
    json.dump({"cortes_frames": cortes}, open(pasta / "camera_cortes.json", "w"))


def _casar(kp, des, kp_ref, des_ref, bf):
    """Homografia (quadro atual → referência) por ORB + teste da razão de Lowe + RANSAC."""
    if des is None or des_ref is None or len(kp) < 10 or len(kp_ref) < 10:
        return None, 0
    pares = bf.knnMatch(des, des_ref, k=2)
    bons = [m for m, *o in pares if o and m.distance < 0.75 * o[0].distance]
    if len(bons) < 10:
        return None, 0
    p = np.float32([kp[m.queryIdx].pt for m in bons])
    q = np.float32([kp_ref[m.trainIdx].pt for m in bons])
    M, ok = cv2.findHomography(p, q, cv2.RANSAC, 2.0)
    return M, int(ok.sum()) if ok is not None else 0


# ── 3. Projeção ────────────────────────────────────────────────────────────

def projetar(pasta: Path, sem_camera: bool):
    H0 = np.array(json.loads((pasta / "calibracao.json").read_text())["H0"])
    G = None if sem_camera else np.load(pasta / "camera.npy")

    def para_campo(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        xy = np.full((len(df), 2), np.nan)
        for f, idx in df.groupby("frame").indices.items():
            if G is not None and f >= len(G):
                continue
            Ht = H0 if G is None else H0 @ G[f]
            xy[idx] = aplicar_h(Ht, df.iloc[idx][["px", "py"]].to_numpy())
        df["x_m"], df["y_m"] = xy[:, 0], xy[:, 1]
        # Descarta pontos muito fora do campo (erros de projeção/detecções na arquibancada)
        fora = (df["x_m"] < -5) | (df["x_m"] > COMPRIMENTO + 5) | (df["y_m"] < -5) | (df["y_m"] > LARGURA + 5)
        df.loc[fora, ["x_m", "y_m"]] = np.nan
        return df

    tracks = pd.read_csv(pasta / "tracks.csv")
    jog = para_campo(tracks[tracks["classe"] != "bola"])
    jog.to_csv(pasta / "tracks_campo.csv", index=False)
    print(f"Jogadores projetados: {jog['x_m'].notna().mean()*100:.1f}% válidos → tracks_campo.csv")

    if (pasta / "bola.csv").exists():
        bola = para_campo(pd.read_csv(pasta / "bola.csv"))
        bola.to_csv(pasta / "bola_campo.csv", index=False)
        print(f"Bola projetada: {bola['x_m'].notna().mean()*100:.1f}% válidos → bola_campo.csv")


def main():
    p = argparse.ArgumentParser(description="Homografia pixel → campo (metros)")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrar")
    c.add_argument("--saida", required=True)
    c.add_argument("--pontos", default="", help="Marcos separados por vírgula (ver campo.MARCOS)")
    c.add_argument("--sem-janela", action="store_true", help="Lê os pixels de calibracao.json")
    m = sub.add_parser("camera")
    m.add_argument("--saida", required=True)
    m.add_argument("--limiar-corte", type=float, default=0.5, help="Correlação mínima entre quadros")
    m.add_argument("--margem-placar", type=float, default=0.12, help="Fração superior ignorada")
    j = sub.add_parser("projetar")
    j.add_argument("--saida", required=True)
    j.add_argument("--sem-camera", action="store_true", help="Assume câmera fixa (usa só H0)")
    a = p.parse_args()

    pasta = Path(a.saida)
    if a.cmd == "calibrar":
        calibrar(pasta, [s for s in a.pontos.split(",") if s], a.sem_janela)
    elif a.cmd == "camera":
        movimento_camera(pasta, a.limiar_corte, a.margem_placar)
    else:
        projetar(pasta, a.sem_camera)


if __name__ == "__main__":
    main()
