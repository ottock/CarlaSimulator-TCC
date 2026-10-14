#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""O controle 8BitDo chega no Jetson? Mostra botoes, analogicos e gatilhos.

Por que existe. O controle e um 8BitDo Ultimate 2C com o adaptador USB de 2.4G
(lsusb: 2dc8:310a). O kernel 4.9 do Jetson nao o reconhece, e mesmo forcando o
driver xpad (new_id) o /dev/input/js0 aparece mas NAO entrega eventos. O que
funciona e conversar direto com o adaptador pela USB, que e o que este script
faz. A historia inteira esta em docs/CONTROLE_8BITDO.md.

O QUE ELE FAZ
  - acha o adaptador, manda a mensagem de inicializacao e o comando de LED do
    Xbox 360, e le os relatorios de 20 bytes;
  - imprime uma linha SO quando algo muda: botoes apertados, os dois
    analogicos (lado, vertical) de -1.00 a +1.00 e os gatilhos de 0.00 a 1.00;
  - "SEM DADOS do adaptador" e normal com a mao parada: o adaptador so manda
    relatorio quando algo muda.
  - Nao toca em camera, LiDAR nem PCA9685. O carro nao tem como andar.

COMO USAR
  sudo apt install python3-usb        (uma vez)
  sudo python3 hardware/pad_teste.py
  Ligue o controle (chave traseira em 2.4G, luz do Home fixa) e mexa.
  Ctrl+C encerra.

Rodado no Jetson em 2026-10-13 (com sudo).
"""
import struct
import sys

import usb.core

d = usb.core.find(idVendor=0x2dc8, idProduct=0x310a)
if d is None:
    sys.exit("adaptador nao encontrado")
if d.is_kernel_driver_active(0):
    d.detach_kernel_driver(0)
d.ctrl_transfer(0xC1, 0x01, 0x0100, 0x00, 20)
d.write(0x05, bytes([0x01, 0x03, 0x02]), 1000)

NOMES = ["cima", "baixo", "esq", "dir", "start", "back", "L3", "R3",
         "LB", "RB", "home", "?", "A", "B", "X", "Y"]
anterior = None
mudo = False
while True:
    try:
        r = d.read(0x84, 64, 500)
    except usb.core.USBError:
        if not mudo:
            print("SEM DADOS do adaptador")
            mudo = True
        continue
    if mudo:
        print("dados voltaram")
        mudo = False
    if len(r) < 14 or r[0] != 0:
        continue
    botoes = r[2] | (r[3] << 8)
    lt, rt = r[4], r[5]
    ex, ey, dx, dy = struct.unpack_from("<4h", bytes(r), 6)
    estado = (botoes, lt // 16, rt // 16,
              ex // 2048, ey // 2048, dx // 2048, dy // 2048)
    if estado != anterior:
        anterior = estado
        apertados = [n for i, n in enumerate(NOMES) if botoes >> i & 1]
        print(apertados,
              "esq %+.2f %+.2f" % (ex / 32767.0, ey / 32767.0),
              "dir %+.2f %+.2f" % (dx / 32767.0, dy / 32767.0),
              "LT %.2f RT %.2f" % (lt / 255.0, rt / 255.0))
