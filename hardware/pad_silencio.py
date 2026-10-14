#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quanto tempo o controle 8BitDo fica MUDO com alguem segurando um comando?

Por que existe. O adaptador so manda relatorio quando algo muda, e quando o
controle desliga ele nao avisa: o ultimo estado fica valendo (medido em
2026-10-13, ver docs/CONTROLE_8BITDO.md). Se um dia o visitante dirigir pelo
controle, a protecao e parar o carro quando faltar relatorio novo por um tempo
curto. Isso so serve se, com a pessoa segurando o comando de "andar", os
relatorios chegarem sem parar. Este script mede isso.

O QUE ELE FAZ
  - a cada 2 s imprime quantos relatorios chegaram e o maior silencio do
    periodo. Se o silencio continua, o numero cresce de uma linha para a outra.
  - Nao toca em camera, LiDAR nem PCA9685. O carro nao tem como andar.

COMO USAR
  sudo python3 hardware/pad_silencio.py
  Segure cada situacao por ~6 s e anote as linhas:
    1. RT apertado ate o fim, analogicos soltos;
    2. analogico direito empurrado para a frente e segurado;
    3. analogico esquerdo segurado para um lado;
    4. tudo solto.
  Ctrl+C encerra.

AINDA NAO RODADO no Jetson (escrito em 2026-10-13).
"""
import sys
import time

import usb.core

d = usb.core.find(idVendor=0x2dc8, idProduct=0x310a)
if d is None:
    sys.exit("adaptador nao encontrado")
if d.is_kernel_driver_active(0):
    d.detach_kernel_driver(0)
d.ctrl_transfer(0xC1, 0x01, 0x0100, 0x00, 20)
d.write(0x05, bytes([0x01, 0x03, 0x02]), 1000)

inicio = ultimo = time.time()
n = 0
maior = 0.0
while True:
    try:
        d.read(0x84, 64, 100)
        agora = time.time()
        maior = max(maior, agora - ultimo)
        ultimo = agora
        n += 1
    except usb.core.USBError:
        agora = time.time()
        maior = max(maior, agora - ultimo)
    if agora - inicio >= 2.0:
        print("relatorios em 2 s: %4d   maior silencio: %.2f s" % (n, maior))
        inicio = agora
        n = 0
        maior = 0.0
