# Conceitos — Prática 3

Resumo da teoria por trás de cada etapa do pipeline.

## 1. Detecção com YOLO

O YOLO (*You Only Look Once*) é um detector de **estágio único**: uma única passada da CNN prevê, para cada região da imagem, caixas `(x, y, w, h)`, um escore de objeto e as probabilidades das classes. Em seguida, a **NMS** (*non-maximum suppression*) remove caixas duplicadas.

- Modelos pré-treinados no **COCO** já trazem `person` (0) e `sports ball` (32), mas não distinguem jogador, goleiro e árbitro.
- Os tamanhos `n/s/m/l/x` trocam velocidade por precisão. Ahmad Yousef et al. (2025) compararam YOLOv8x (68,2 M parâmetros, 15,3 ms) e YOLOv12x (59,1 M, 11,8 ms): o v8x teve F1 melhor para jogadores e árbitros.
- A **bola** é o objeto mais difícil: poucos pixels, borrão de movimento, oclusões. Sem o filtro de Kalman, o F1 da bola ficou em torno de 0,31 no Dataset 2 do artigo.

## 2. Rastreamento com ByteTrack

O detector trabalha quadro a quadro; o rastreador liga as detecções no tempo e dá a cada jogador um `track_id`.

1. Um **filtro de Kalman** prevê onde cada track estará no próximo quadro.
2. As detecções de **alta confiança** são associadas às predições por IoU (algoritmo húngaro).
3. As de **baixa confiança** são usadas numa segunda rodada, para recuperar tracks parcialmente ocultos.

Limitações: trocas de ID em aglomerações e IDs novos quando o jogador sai e volta ao quadro. Isso afeta diretamente distância percorrida e passes.

## 3. Times por cor da camisa (K-Means em HSV)

Da metade superior do box (camisa) remove-se o verde da grama e calcula-se a cor média em **HSV**. O HSV separa matiz (cor) de saturação e brilho, o que dá mais robustez a sombras e mudanças de exposição. Um **K-Means** com k = 2 separa os times; com k = 3, o cluster menor tende a juntar árbitro e goleiros.

## 4. Filtro de Kalman para a bola

Estado `x = [px, py, vx, vy]`, com modelo de velocidade constante:

```
predição:  x⁻ = F·x          P⁻ = F·P·Fᵀ + Q
correção:  K = P⁻Hᵀ(HP⁻Hᵀ + R)⁻¹    x = x⁻ + K(z − Hx⁻)    P = (I − KH)P⁻
```

- Nos quadros **sem detecção**, usa-se só a predição (preenche lacunas curtas).
- O **gating** por distância de Mahalanobis descarta detecções incoerentes com o movimento (falsos positivos).

## 5. Homografia: do pixel ao campo

Como o campo é um **plano**, pixels `(u, v)` e metros `(x, y)` se relacionam por uma homografia 3×3:

```
[x·w, y·w, w]ᵀ = H · [u, v, 1]ᵀ
```

- São necessários **≥ 4 pares de pontos** não colineares (marcos do campo: meio-campo, quinas das áreas, círculo central).
- A câmera de transmissão faz **pan e zoom**: estimamos a homografia entre quadros consecutivos com pontos ORB fora dos jogadores e encadeamos `G_t = G_{t-1} · M_t`. A posição no campo é `H0 · G_t · p`. Pequenos erros se acumulam (**deriva**), e um **corte de câmera** invalida a cadeia.
- Projetamos o **pé** do jogador, que está no plano do gramado. A bola no ar viola essa hipótese.

## 6. Posse de bola e passes

Seguindo Ahmad Yousef et al. (2025): em cada quadro, o dono da bola é o jogador mais próximo, se a distância for menor que um raio (≈ 2 m). A posse de um time é a fração do tempo em que ele detém a bola; nos quadros "em trânsito", a posse fica com o último time a tocar.

Um **passe** é uma troca de dono: certo se o novo dono é do mesmo time, errado caso contrário. Filtros mínimos: deslocamento da bola ≥ 3 m (evita troca de ID) e duração ≤ 4 s.

## 7. eXP e EPV

O **EPV** (*Expected Possession Value*) estima o valor esperado de uma posse, `EPV_t = E[G | T_t]`, com G ∈ {1, −1, 0} (gol marcado, sofrido, nenhum). Gomes et al. (2026) seguem Fernández et al. (2021) e decompõem esse valor por ação (passe, condução, chute). Para o passe:

```
E[G | passe, T_t] = ∫ P(D_t | passe, T_t) · Σ_o P(O = o | D_t, T_t) · E[G | O = o, D_t, T_t] dD_t
```

O termo **P(O = 1 | D_t, T_t)**, a probabilidade de sucesso de um passe para o destino D_t, é o nosso **eXP**. No artigo ele é uma superfície 104 × 68 produzida pela **SoccerMap**, uma FCN multiescala alimentada por 17 canais. Nesta prática usamos uma regressão logística sobre features geométricas:

| Feature | Intuição | Canal análogo no artigo |
|---|---|---|
| `distancia` | passes longos erram mais | distância à bola / ao início do evento |
| `adversarios_na_linha` | defensores no cone de interceptação | defensores entre bola e célula |
| `pressao_passador` | adversários a < 3 m do passador | ocupação do time defensor |
| `espaco_destino` | distância do adversário mais próximo ao destino | ocupação / espaço |
| `progressao_x` | passes para frente são mais arriscados | posição / ângulo ao gol |
| `dist_destino_gol` | zonas perto do gol são mais disputadas | distância ao gol |

### Avaliação (como na Tabela 2 de Gomes et al.)

- **AUC-ROC**: capacidade de ordenar passes certos acima dos errados (0,5 = aleatório). No artigo: 0,622 (SoccerMap) × 0,588 (baseline).
- **Brier**: erro quadrático médio da probabilidade (quanto menor, melhor).
- **Log-Loss**: penaliza previsões confiantes e erradas.
- **Calibração**: a média prevista deve bater com a taxa observada (artigo: 0,830 × 0,829).

Com a taxa de acerto alta (~80% dos passes são completados), a AUC fica naturalmente comprimida; por isso a comparação com um **baseline** é obrigatória.

### PAx e what-if

- **PAx** (*passes acima do esperado*) = passes certos − Σ eXP. Um valor positivo indica um passador que completa mais do que o contexto sugere.
- **What-if** (Seção 3.6 do artigo): perturbe o destino de um passe errado em 3–6 m e reavalie o eXP. Ganhos grandes sugerem que um reposicionamento do receptor tornaria o passe mais seguro.

## Referências

- Ahmad Yousef, K. M. et al. Real-Time Football Match Analysis: Leveraging YOLO for Enhanced Object Detection and Possession Tracking. *Procedia Computer Science*, 272, 38–45, 2025. doi:10.1016/j.procs.2025.10.176
- Gomes, R. et al. Modelagem de Passes no Futebol via Componentes do Expected Possession Value com Redes Totalmente Convolucionais. *REIC*, 24:1, 2026. doi:10.5753/reic.2026.8499
- Fernández, J.; Bornn, L.; Cervone, D. A framework for the fine-grained evaluation of the instantaneous expected value of soccer possessions. *Machine Learning*, 2021.
- Fernández, J.; Bornn, L. SoccerMap: A Deep Learning Architecture for Visually-Interpretable Analysis in Soccer. *ECML PKDD*, 2020.
- Zhang, Y. et al. ByteTrack: Multi-Object Tracking by Associating Every Detection Box. *ECCV*, 2022.
