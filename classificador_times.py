"""
Classificador de times por cor da camisa (K-Means no espaço HSV).

Versão comentada em PT-BR de implementacoes/football_track/team_classifier.py,
com duas extensões:
  1. k configurável: com k=3, o menor cluster vira "outros" (árbitro/goleiros);
  2. votação: cada track_id recebe o time mais votado nas primeiras observações,
     em vez de ficar preso à primeira predição (que pode vir de um recorte ruim).

Estratégia
----------
1. Calibração: recortes de jogadores nos primeiros N quadros.
2. Para cada recorte, cor média da região da camisa (ignorando pixels de grama)
   num espaço HSV "circular": o matiz H é um ângulo, e o vermelho fica nas duas
   pontas (H≈0 e H≈179). A média simples de H daria ≈90 (ciano)! Por isso usamos
   (S·cos H, S·sin H, V), que trata o matiz como ângulo.
3. K-Means sobre esses vetores de cor → centróides dos times.
4. Inferência: cada jogador rastreado vai para o centróide mais próximo.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import cv2
import numpy as np
from sklearn.cluster import KMeans

# Faixa vertical do bounding box usada como "camisa" (evita cabeça, calção e meião)
_CAMISA_TOPO = 0.10
_CAMISA_BASE = 0.55

# Faixa HSV do verde da grama (OpenCV: H ∈ [0, 179])
_GRAMA_H = (35, 85)
_GRAMA_S_MIN = 40
_GRAMA_V_MIN = 40

OUTROS = -2        # árbitro, goleiros ou cluster minoritário
DESCONHECIDO = -1  # ainda não calibrado / recorte inválido


def cor_camisa(recorte_bgr: np.ndarray) -> np.ndarray | None:
    """Cor média (S·cos H, S·sin H, V) da região da camisa, sem os pixels de grama."""
    h, w = recorte_bgr.shape[:2]
    if h < 20 or w < 10:
        return None

    camisa = recorte_bgr[int(h * _CAMISA_TOPO): int(h * _CAMISA_BASE), :]
    hsv = cv2.cvtColor(camisa, cv2.COLOR_BGR2HSV)

    grama = (
        (hsv[:, :, 0] >= _GRAMA_H[0]) & (hsv[:, :, 0] <= _GRAMA_H[1]) &
        (hsv[:, :, 1] >= _GRAMA_S_MIN) & (hsv[:, :, 2] >= _GRAMA_V_MIN)
    )
    manter = ~grama
    if manter.sum() < 30:
        return None
    px = hsv[manter].astype(np.float32)
    ang = px[:, 0] * (2 * np.pi / 180.0)          # H do OpenCV ∈ [0, 180) → radianos
    return np.array([(px[:, 1] * np.cos(ang)).mean(),
                     (px[:, 1] * np.sin(ang)).mean(),
                     px[:, 2].mean()], dtype=np.float32)


def para_bgr(cor: np.ndarray) -> tuple[int, int, int]:
    """Converte (S·cos H, S·sin H, V) de volta para BGR (para desenhar/conferir)."""
    h = (np.arctan2(cor[1], cor[0]) % (2 * np.pi)) * 180.0 / (2 * np.pi)
    s = np.clip(np.hypot(cor[0], cor[1]), 0, 255)
    hsv = np.uint8([[[h, s, np.clip(cor[2], 0, 255)]]])
    return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


class ClassificadorTimes:
    """K-Means de cor de camisa, calibrado nos quadros iniciais do clipe."""

    def __init__(self, k: int = 2, votos_necessarios: int = 10):
        self.k = k
        self.votos_necessarios = votos_necessarios
        self.calibrado = False
        self._kmeans: KMeans | None = None
        self._cluster_para_time: dict[int, int] = {}
        self._amostras: list[np.ndarray] = []
        self._votos: dict[int, Counter] = defaultdict(Counter)
        self._time_fixo: dict[int, int] = {}

    # ── Calibração ─────────────────────────────────────────────────────
    def adicionar_amostra(self, recorte_bgr: np.ndarray):
        cor = cor_camisa(recorte_bgr)
        if cor is not None:
            self._amostras.append(cor)

    def ajustar(self) -> bool:
        """Ajusta o K-Means; retorna False se ainda há poucas amostras."""
        if len(self._amostras) < 20 * self.k:
            return False
        X = np.stack(self._amostras)
        self._kmeans = KMeans(n_clusters=self.k, n_init=10, random_state=42).fit(X)

        # Os 2 maiores clusters são os times 0 e 1; os demais viram OUTROS
        tamanhos = np.bincount(self._kmeans.labels_, minlength=self.k)
        ordem = np.argsort(-tamanhos)
        self._cluster_para_time = {int(c): (i if i < 2 else OUTROS) for i, c in enumerate(ordem)}

        self.calibrado = True
        self._amostras.clear()
        return True

    def centroides_bgr(self) -> dict[int, tuple[int, int, int]]:
        """Cor média de cada time em BGR (útil para desenhar e conferir)."""
        return {time: para_bgr(self._kmeans.cluster_centers_[c])
                for c, time in self._cluster_para_time.items()}

    # ── Inferência ─────────────────────────────────────────────────────
    def prever(self, track_id: int, recorte_bgr: np.ndarray) -> int:
        """Time (0, 1), OUTROS ou DESCONHECIDO para este jogador."""
        if not self.calibrado:
            return DESCONHECIDO
        if track_id in self._time_fixo:
            return self._time_fixo[track_id]

        cor = cor_camisa(recorte_bgr)
        if cor is None:
            return self._mais_votado(track_id)

        cluster = int(self._kmeans.predict(cor.reshape(1, -1))[0])
        self._votos[track_id][self._cluster_para_time[cluster]] += 1

        # Depois de votos suficientes, a decisão fica fixa (estabilidade temporal)
        if sum(self._votos[track_id].values()) >= self.votos_necessarios:
            self._time_fixo[track_id] = self._mais_votado(track_id)
        return self._mais_votado(track_id)

    def _mais_votado(self, track_id: int) -> int:
        votos = self._votos.get(track_id)
        return votos.most_common(1)[0][0] if votos else DESCONHECIDO

    def esquecer(self, track_id: int):
        """Chamar quando um track some, para reclassificá-lo se voltar."""
        self._votos.pop(track_id, None)
        self._time_fixo.pop(track_id, None)
