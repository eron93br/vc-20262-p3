# `src/` — Códigos de apoio da Prática 3

Scripts para ir de um vídeo do YouTube até métricas de jogo e eXP. O roteiro com as questões está em [ROTEIRO.md](ROTEIRO.md) e a teoria em [CONCEITOS.md](CONCEITOS.md).

> Os scripts são um **ponto de partida**, não uma solução fechada: vários parâmetros (raio de posse, coeficientes do eXP, limiares) precisam ser estudados e justificados pelo grupo.

## Pipeline

```
YouTube ──► baixar_video.py ──► clipe.mp4
                                   │
                    detectar_rastrear.py   (YOLO + ByteTrack + K-Means de cor)
                                   │
              tracks.csv · frame0.jpg · meta.json · rastreado.mp4
                                   │
                ┌──────────────────┴──────────────────┐
         bola_kalman.py                     homografia.py calibrar / camera
          (bola.csv)                    (calibracao.json · camera.npy)
                └──────────────────┬──────────────────┘
                         homografia.py projetar
                   (tracks_campo.csv · bola_campo.csv)
                                   │
                             metricas.py
     (metricas_jogadores.csv · passes.csv · metricas_times.json · figuras/)
                                   │
                ┌──────────────────┴──────────────────┐
         validar_passes.py                        xpass.py
      (precisão/revocação)        (passes_features.csv · resultados_xp.csv ·
                                   jogadores_xp.csv · superfície de eXP)
```

| Arquivo | Etapa | O que faz |
|---|---|---|
| `baixar_video.py` | Parte 1 | Baixa **só o trecho** pedido (yt-dlp `download_ranges`), até 720p |
| `detectar_rastrear.py` | Partes 2–3 | YOLO (pessoa + bola) + ByteTrack + times; gera `tracks.csv` |
| `classificador_times.py` | Parte 3 | K-Means em HSV da camisa, com votação por `track_id` |
| `bola_kalman.py` | Parte 4 | Filtro de Kalman (velocidade constante) com *gating* para a bola |
| `campo.py` | Parte 4 | Dimensões do campo, marcos para calibração e desenho com matplotlib |
| `homografia.py` | Parte 4 | Calibração H0, movimento de câmera (ORB + RANSAC) e projeção em metros |
| `metricas.py` | Parte 5 | Distância/velocidade, posse por proximidade, detecção de passes, figuras |
| `validar_passes.py` | Parte 5 | Compara passes detectados com anotação manual |
| `xpass.py` | Parte 6 | Features do passe, eXP (baseline, heurístico, logístico), superfície, what-if, PAx |
| `soccermap_mini.py` | Extra | FCN em 3 escalas inspirada na SoccerMap (PyTorch) |

## Instalação

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# ffmpeg é necessário para baixar trechos:  brew install ffmpeg  |  sudo apt install ffmpeg
```

No Google Colab basta `!pip install -r requirements.txt` (ffmpeg já vem instalado). A calibração da homografia abre uma janela OpenCV; no Colab use o modo `--sem-janela` (veja abaixo).

## Passo a passo

```bash
cd src
# 1. Trecho de 4 minutos
python baixar_video.py "https://www.youtube.com/watch?v=VIDEO_ID" --inicio 12 --fim 16 --saida ../dados/videos

# 2–3. Detecção + rastreamento + times (teste antes com 30 s: --inicio 0 --fim 0.5)
python detectar_rastrear.py --video "../dados/videos/<titulo>_clip.mp4" --saida ../saidas/jogo1 --modelo yolov8m.pt

# 4. Bola e homografia
python bola_kalman.py --saida ../saidas/jogo1
python homografia.py calibrar --saida ../saidas/jogo1 --pontos meio_sup,centro,meio_inf,circulo_sup,circulo_inf
python homografia.py camera   --saida ../saidas/jogo1
python homografia.py projetar --saida ../saidas/jogo1

# 5. Métricas e validação
python metricas.py --saida ../saidas/jogo1
python validar_passes.py --saida ../saidas/jogo1 --anotado passes_anotados.csv

# 6. eXP
python xpass.py --saida ../saidas/jogo1 --what-if
```

### Dicas de desempenho

| Hardware | Sugestão |
|---|---|
| GPU NVIDIA / Colab T4 | `--modelo yolov8m.pt --imgsz 1280` (≈ 15–30 fps) |
| Apple Silicon (MPS) | `--modelo yolov8m.pt --imgsz 1280` (≈ 4–10 fps; a predição extra da bola pesa) |
| Só CPU | `--modelo yolov8s.pt --imgsz 640 --sem-video` e trechos curtos |

### Calibração sem janela (Colab)

Abra `frame0.jpg` em qualquer visualizador, anote os pixels (u, v) de ≥ 4 marcos e crie `saidas/jogo1/calibracao.json`:

```json
{"pixels": [[640, 112], [598, 361], [520, 702], [1180, 230]],
 "marcos": ["meio_sup", "centro", "meio_inf", "ga_dir_sup"]}
```

Depois rode `python homografia.py calibrar --saida ../saidas/jogo1 --sem-janela`. Os nomes dos marcos estão em `campo.MARCOS`.

### Sem homografia?

Se a calibração não for possível (câmera muito fechada, sem linhas visíveis), `metricas.py --pixels` calcula posse e passes em pixels. É uma aproximação: as distâncias ficam distorcidas pela perspectiva e o `xpass.py` não deve ser usado. Discutam essa limitação.

## Formato dos arquivos

**`tracks.csv`** — uma linha por detecção

| coluna | descrição |
|---|---|
| `frame`, `t_s` | índice do quadro e tempo (s) desde o início do trecho |
| `track_id` | ID do ByteTrack (−1 para a bola) |
| `classe` | nome da classe do modelo (`person`, `player`, `referee`, `bola`, …) |
| `time` | 0 ou 1; −2 = outros (árbitro/goleiro com `--k 3`); −1 = ainda não classificado |
| `x1,y1,x2,y2`, `conf` | bounding box em pixels e confiança |
| `px, py` | ponto de referência: pé do jogador ou centro da bola |

**`passes.csv`** — `frame_ini`, `frame_fim`, `passador`, `time`, `receptor` (pretendido, se errado), `tocou_em`, `sucesso`, `x0,y0` → `x1,y1` (m), `distancia`, `duracao_s`.

**`passes_features.csv`** — passes + `distancia`, `adversarios_na_linha`, `pressao_passador`, `espaco_destino`, `progressao_x`, `dist_destino_gol`.

## Problemas comuns

- **Quase nenhuma bola detectada:** aumente `--imgsz`, reduza `--conf-bola`, use um modelo maior ou ajustado para futebol.
- **Times trocados ou misturados:** confira `meta.json → times_bgr`; uniformes parecidos ou goleiros exigem `--k 3` ou outro descritor.
- **Projeção absurda (jogadores fora do campo):** refaça a calibração com marcos bem espalhados; verifique cortes de câmera em `camera_cortes.json`.
- **Direção de ataque errada** (superfície de eXP "espelhada"): a heurística usa os jogadores mais extremos de cada time e erra se o goleiro for classificado no time errado. Force com `metricas.py --ataque-time0 direita` (ou `esquerda`) e rode o `xpass.py` de novo.
- **Nenhum passe:** revise `--raio-posse` e a cobertura da bola; confira `posse_por_quadro.csv`.
