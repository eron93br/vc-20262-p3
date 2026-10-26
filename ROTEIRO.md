# Prática 3 — Futebol Analytics com YOLO

> **Visão Computacional · Roteiro da prática**
> Entrega: **link do repositório GitHub do grupo** no Classroom.

## Objetivo

Construir, a partir de um vídeo de transmissão do YouTube, um pipeline que **detecta**, **rastreia** e **projeta no campo** jogadores e bola, e então extrai **métricas de jogo**: posse de bola, distância percorrida, passes certos/errados e o **eXP (expected pass)** — a probabilidade de um passe ser completado dado o contexto.

O que vocês vão fazer:

1. **Obter** um trecho de 3–5 min de um jogo (não o jogo todo);
2. **Detectar e rastrear** jogadores e bola com YOLO + ByteTrack e separar os times;
3. **Projetar** as posições para coordenadas do campo (metros) via homografia;
4. **Calcular e validar** métricas de jogo e um modelo de eXP, com análise crítica.

## Artigos de referência

| Artigo | O que aproveitar |
|---|---|
| **Ahmad Yousef et al. (2025)** — *Real-Time Football Match Analysis: Leveraging YOLO for Enhanced Object Detection and Possession Tracking*. Procedia Computer Science 272. | Comparação YOLOv8x × YOLOv12x; dificuldade com a **bola** (F1 ≈ 0,31) resolvida com **filtro de Kalman** (F1 ≈ 0,96); **posse por proximidade** jogador–bola; discussão da **distorção de perspectiva**. |
| **Gomes et al. (2026)** — *Modelagem de Passes no Futebol via Componentes do Expected Possession Value com Redes Totalmente Convolucionais*. REIC 24:1. | Formulação do **EPV** e do componente de passe P(sucesso \| destino, estado); representação do estado em **17 canais espaciais**; **SoccerMap**; avaliação com **AUC, Brier, Log-Loss** contra baseline de distância; análise **what-if**. |

A implementação de partida é `implementacoes/football_track/` (YOLOv8 + ByteTrack + K-Means de cor). Os códigos de apoio estão nesta pasta `src/` — veja o [README](README.md).

## Regras

- **Repositório GitHub do grupo** é o único formato aceito; vale o estado no prazo.
- **Uso de IA:** permitido no código; a análise e o README devem ser autorais, com declaração de uso.
- **Vídeo:** não versionar vídeos no repositório. Registrar no README a **URL** e o **intervalo** (início–fim) usados.
- **Mesmo protocolo:** os números citados no README devem ser os gerados pelos scripts (`metricas_times.json`, `resultados_xp.csv`).

---

## Parte 1 — Aquisição do vídeo

Use `baixar_video.py` para baixar **apenas o trecho** (3–5 min) de um jogo. Prefira a câmera principal (plano aberto), com jogo corrido e **sem replays/cortes**.

```bash
python baixar_video.py "URL" --inicio 12 --fim 16 --saida ../dados/videos
```

- **Q1.1** Justifiquem a escolha do jogo e do trecho (câmera, iluminação, cores dos uniformes, presença de cortes).
- **Q1.2** Qual a resolução e a taxa de quadros do vídeo? Quantos quadros vocês vão processar? Qual o custo estimado em tempo no hardware de vocês?

## Parte 2 — Detecção com YOLO

Rode `detectar_rastrear.py` com pelo menos **dois tamanhos de modelo** (ex.: `yolov8n.pt` e `yolov8m.pt`, ou `yolo11s.pt` e `yolo11m.pt`) em um mesmo subtrecho de 30 s.

- **Q2.1** Compare os modelos em: fps de processamento, nº médio de jogadores detectados por quadro e **% de quadros com bola detectada**. Apresente uma tabela.
- **Q2.2** Por que a bola é tão mais difícil que os jogadores? Relacione com tamanho em pixels, borrão de movimento e com o resultado de Ahmad Yousef et al. (Tabela 2).
- **Q2.3** Qual o efeito de `--imgsz` (640 × 1280) e de `--conf-bola` na detecção da bola? Mostrem exemplos de falsos positivos.
- **Q2.4 (opcional)** Ajustem um modelo com um dataset de futebol do Roboflow Universe (classes *player, goalkeeper, referee, ball*) e comparem com o modelo COCO.

## Parte 3 — Rastreamento e times

- **Q3.1** Assistam ao `rastreado.mp4`. Estimem quantas **trocas de ID** ocorrem em 1 minuto e em que situações (oclusão, saída do quadro, aglomeração).
- **Q3.2** A classificação por K-Means de cor funcionou? Mostrem as cores dos centróides (`meta.json → times_bgr`) e casos de erro (goleiro, árbitro, sombra). Por que `classificador_times.py` usa (S·cos H, S·sin H, V) em vez da média simples de H? O que aconteceria com uma camisa vermelha?
- **Q3.3** Testem `--k 3`. O árbitro passou a ser separado? Discutam vantagens e riscos.

## Parte 4 — Bola e homografia (do pixel ao campo)

```bash
python bola_kalman.py --saida ../saidas/jogo1
python homografia.py calibrar --saida ../saidas/jogo1 --pontos meio_sup,centro,meio_inf,ga_dir_sup,ga_dir_inf,ga_dir_sup_linha
python homografia.py camera   --saida ../saidas/jogo1
python homografia.py projetar --saida ../saidas/jogo1
```

- **Q4.1** Qual a cobertura da bola antes e depois do filtro de Kalman? Mostrem um trecho em que o Kalman acerta e outro em que ele "inventa" a bola.
- **Q4.2** Apresentem a `calibracao_grade.jpg` e o **erro de reprojeção** de cada marco. Quais marcos geram mais erro e por quê?
- **Q4.3** Explique a composição `H0 · G_t`. O que acontece com o erro ao longo do clipe (deriva)? Proponham uma forma de re-ancorar a calibração.
- **Q4.4** Por que usar o **pé** do jogador (base do box) e não o centro do box como ponto de projeção? E para a bola, que está no ar em lançamentos?

## Parte 5 — Métricas de jogo

```bash
python metricas.py --saida ../saidas/jogo1
python validar_passes.py --saida ../saidas/jogo1 --anotado passes_anotados.csv
```

- **Q5.1** Reportem posse (%) por time, distância total e velocidade máxima. Os valores são fisicamente plausíveis (ex.: um jogador corre ~10–12 km em 90 min; ~120 m/min)? Se não, expliquem a causa.
- **Q5.2** Estudem a sensibilidade da posse e do nº de passes ao `--raio-posse` (1, 2, 3 m). Façam um gráfico.
- **Q5.3** **Validação:** anotem manualmente os passes do trecho (`passes_anotados.csv`) e reportem precisão, revocação e F1 da detecção. Quais erros dominam?
- **Q5.4** A direção de ataque estimada (`metricas_times.json`) está correta? Se não, por quê? Como trocas de ID e jogadores fora do quadro afetam distância e passes? Que correção vocês proporiam?

## Parte 6 — eXP (expected pass)

```bash
python xpass.py --saida ../saidas/jogo1 --what-if
```

- **Q6.1** Descrevam cada feature do eXP e relacionem com os canais da Tabela 1 de Gomes et al. (2026). Que informação dos 17 canais vocês **não** conseguem extrair do vídeo, e por quê?
- **Q6.2** Comparem **baseline (distância)**, **heurístico** e **logístico** em AUC, Brier e Log-Loss (`resultados_xp.csv`). Com poucos passes, o que muda na confiança dessas métricas? Juntem passes de outro trecho/grupo com `--extra`.
- **Q6.3** Analisem a **curva de calibração** e a média prevista × taxa observada.
- **Q6.4** Interpretem a **superfície de eXP** de um lance: as regiões quentes e frias fazem sentido tático?
- **Q6.5** Usem o **what-if** e o **PAx** (passes acima do esperado): que jogador se destaca? Quais as limitações de tirar conclusões de um trecho de 5 min?
- **Desafio extra:** treinem a `soccermap_mini.py` com os passes da turma e comparem com a regressão logística.

---

## Entrega

Link do repositório no Classroom contendo:

- `README.md` com URL e intervalo do vídeo, comandos executados, respostas às questões e declaração de uso de IA;
- código (scripts ou notebook) e a pasta `saidas/` **sem vídeos** (`tracks.csv`, `passes.csv`, `metricas_*.{csv,json}`, `resultados_xp.csv`, `passes_anotados.csv`);
- pasta `figuras/` (mapas de calor, passes, posse, superfície de eXP, calibração).

## Critérios de avaliação

| Peso | Item |
|---:|---|
| 10% | Aquisição e preparação (Parte 1) |
| 25% | Detecção, rastreamento e times (Partes 2–3) |
| 30% | Homografia, métricas e eXP (Partes 4–6) |
| 35% | Análise crítica e validação |

**Penalidades:** vídeo versionado no repositório −10% · números do README ≠ arquivos gerados −20% · sem validação manual dos passes (Q5.3) −15%.
