#!/usr/bin/env python3
"""
Baixa um TRECHO de um vídeo do YouTube (até 720p) para a prática.

Adaptado de implementacoes/download_youtube_videos.py. A diferença principal é
que aqui o yt-dlp baixa apenas o intervalo pedido (download_ranges), em vez de
baixar o jogo inteiro e cortar depois — um jogo completo em 720p passa de 1 GB.

Uso
---
# Trecho de 3 minutos (do minuto 12 ao 15)
python baixar_video.py "https://www.youtube.com/watch?v=VIDEO_ID" --inicio 12 --fim 15

# Escolhendo a pasta de saída
python baixar_video.py "URL" --inicio 12 --fim 15 --saida ../dados/videos

Dicas para escolher o trecho
----------------------------
- Prefira a câmera principal (plano aberto, lateral), SEM replays e cortes de câmera.
- 3 a 5 minutos são suficientes; jogo corrido rende mais passes que bola parada.
- Não versione o vídeo no GitHub: registre apenas a URL e o intervalo no README.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yt_dlp
from yt_dlp.utils import download_range_func


def _parse_args():
    p = argparse.ArgumentParser(description="Baixa um trecho de vídeo do YouTube (≤720p).")
    p.add_argument("url", help="URL do vídeo no YouTube")
    p.add_argument("--inicio", type=float, default=None, metavar="MIN",
                   help="Início do trecho em minutos (ex.: 12.5 = 12min30s)")
    p.add_argument("--fim", type=float, default=None, metavar="MIN",
                   help="Fim do trecho em minutos")
    p.add_argument("--saida", default="videos", help="Pasta de saída (padrão: videos/)")
    return p.parse_args()


def _nome_seguro(nome: str) -> str:
    """Remove caracteres problemáticos para nomes de arquivo."""
    return "".join(c if c.isalnum() or c in " ._-" else "_" for c in nome).strip()


def baixar(url: str, inicio_min: float | None, fim_min: float | None, saida: str) -> Path:
    pasta = Path(saida)
    pasta.mkdir(parents=True, exist_ok=True)

    trecho = inicio_min is not None or fim_min is not None
    if trecho and shutil.which("ffmpeg") is None:
        sys.exit("Erro: o ffmpeg é necessário para baixar trechos e não está no PATH.")

    # Formato: melhor vídeo MP4 até 720p + melhor áudio M4A
    opcoes = {
        "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]",
        "merge_output_format": "mp4",
        "outtmpl": str(pasta / "%(title)s.%(ext)s"),
    }

    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    titulo = _nome_seguro(info.get("title", "video"))
    duracao = float(info.get("duration") or 0)

    if trecho:
        ini_s = (inicio_min or 0.0) * 60
        fim_s = fim_min * 60 if fim_min is not None else duracao
        if fim_s <= ini_s:
            sys.exit("Erro: --fim deve ser maior que --inicio.")
        # Baixa somente o intervalo [ini_s, fim_s]; cortar nos keyframes evita
        # quadros pretos no início do clipe.
        opcoes["download_ranges"] = download_range_func(None, [(ini_s, fim_s)])
        opcoes["force_keyframes_at_cuts"] = True
        opcoes["outtmpl"] = str(pasta / f"{titulo}_clip.%(ext)s")
        print(f"Baixando trecho {ini_s/60:.1f}–{fim_s/60:.1f} min de: {info.get('title')}")
    else:
        opcoes["outtmpl"] = str(pasta / f"{titulo}.%(ext)s")
        print(f"Baixando vídeo completo ({duracao/60:.0f} min): {info.get('title')}")

    with yt_dlp.YoutubeDL(opcoes) as ydl:
        ydl.download([url])

    arquivo = pasta / (f"{titulo}_clip.mp4" if trecho else f"{titulo}.mp4")
    print(f"Pronto. Arquivo: {arquivo}")
    return arquivo


if __name__ == "__main__":
    a = _parse_args()
    baixar(a.url, a.inicio, a.fim, a.saida)
