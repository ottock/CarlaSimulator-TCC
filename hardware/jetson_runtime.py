#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Roda o modelo treinado no carro (Fase 6b, primeiro teste com rodas no ar).

Camera CSI + LiDAR COIN-D6 -> engine TensorRT -> servo. O ESC fica TRAVADO EM
NEUTRO: nesta fase o carro nao anda.

Ambiente: Jetson Nano / JetPack 4.6.x / Python 3.6.

Antes de rodar, gere a engine NO PROPRIO JETSON (engines sao especificas da
maquina e da versao):

    /usr/src/tensorrt/bin/trtexec --onnx=driving_track_180.onnx \\
        --saveEngine=driving_track_180.engine --fp16

Uso:
    python3 hardware/jetson_runtime.py \\
        --engine driving_track_180.engine \\
        --config driving_track_180.json \\
        --out runs/car_teste1 --seconds 60
"""
import argparse
import glob
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

import cv2
import numpy as np
import serial

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401  (inicializa o contexto CUDA)
import Adafruit_PCA9685

from ai.car.coin_d6 import CoinD6Parser
from ai.car.config import car_max_range, load_model_config
from ai.car.control_map import ESC_MAX_US, ESC_MIN_MOVE_US, ESC_NEUTRAL_US, STEER_CENTER_US
from ai.car.loop import DriveLoop
from ai.car.run_log import RunLogger

# ============================ SEGURANCA ============================
# Para o carro ANDAR sao precisas DUAS coisas independentes:
#   1. ESC_ARMADO = True aqui (exige editar o arquivo -- nao e flag de CLI de
#      proposito, para ninguem armar sem querer num comando copiado);
#   2. --cruise-us acima de 1600 na linha de comando.
# Faltando qualquer uma, o ESC fica em neutro e o carro nao sai do lugar.
#
# A velocidade e CONSTANTE (sem aceleracao) e vai a ZERO em qualquer condicao
# degradada: sem volta completa do LiDAR, sem imagem, laco travado, ou obstaculo
# dentro do cone frontal (parada de emergencia).
ESC_ARMADO = False
# ===================================================================

I2C_ADDRESS = 0x40
I2C_BUSNUM = 1
PWM_FREQ = 50
SERVO_CHANNEL = 15
ESC_CHANNEL = 12

BAUD_RATE = 230400
CMD_START = bytes([0xAA, 0x55, 0xF0, 0x0F])
CMD_STOP = bytes([0xAA, 0x55, 0xF5, 0x0A])

CAP_WIDTH, CAP_HEIGHT, CAP_FPS = 1280, 720, 60
FLIP_METHOD = 2          # o modulo IMX219 vem de cabeca pra baixo (ver camera_teste.py)


class CsiCamera:
    """Camera CSI IMX219 via GStreamer (nao abre como webcam USB comum)."""

    def __init__(self, sensor_id=0, flip_method=FLIP_METHOD):
        pipeline = (
            "nvarguscamerasrc sensor-id={sid} ! "
            "video/x-raw(memory:NVMM), width=(int){cw}, height=(int){ch}, "
            "framerate=(fraction){fps}/1 ! "
            "nvvidconv flip-method={flip} ! "
            "video/x-raw, format=(string)BGRx ! videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1"
        ).format(sid=sensor_id, cw=CAP_WIDTH, ch=CAP_HEIGHT, fps=CAP_FPS, flip=flip_method)
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError("nao consegui abrir a camera CSI")

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        self.cap.release()


class SerialLidar:
    """COIN-D6 na serial, devolvendo pontos (angulo, distancia)."""

    def __init__(self, port=None, max_range_m=12.0):
        port = port or self._find_port()
        if not port:
            raise RuntimeError("nenhuma porta de LiDAR encontrada (/dev/ttyUSB*)")
        self.ser = serial.Serial(port=port, baudrate=BAUD_RATE, timeout=0)
        self.ser.reset_input_buffer()
        time.sleep(0.1)
        self.ser.write(CMD_START)
        time.sleep(0.3)
        # max_range aqui e o FISICO do sensor (12 m), nao o do modelo: filtrar
        # antes na escala do carro descartaria leituras validas.
        self.parser = CoinD6Parser(max_range_m=max_range_m)

    @staticmethod
    def _find_port():
        portas = sorted(glob.glob("/dev/ttyUSB*")) + sorted(glob.glob("/dev/ttyACM*"))
        return portas[0] if portas else None

    def read_points(self):
        n = self.ser.in_waiting
        if not n:
            return []
        return self.parser.feed(self.ser.read(n))

    def close(self):
        try:
            self.ser.write(CMD_STOP)
        except Exception:
            pass
        self.ser.close()


class TrtEngine:
    """Engine TensorRT com dois inputs (image, lidar) e um output (control)."""

    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as fh, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(fh.read())
        if self.engine is None:
            raise RuntimeError("falha ao deserializar a engine: {0}".format(engine_path))
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.host = {}
        self.device = {}
        self.bindings = []
        for idx in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(idx)
            shape = tuple(self.context.get_binding_shape(idx))
            shape = tuple(1 if d < 0 else d for d in shape)   # batch dinamico -> 1
            dtype = trt.nptype(self.engine.get_binding_dtype(idx))
            host = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
            dev = cuda.mem_alloc(host.nbytes)
            self.host[name] = host
            self.device[name] = dev
            self.bindings.append(int(dev))
        self._out_name = [self.engine.get_binding_name(i)
                          for i in range(self.engine.num_bindings)
                          if not self.engine.binding_is_input(i)][0]

    def infer(self, img_chw, lidar_vec):
        """(3,66,200) float32 + (72,) float32 -> (steer, throttle, brake)."""
        np.copyto(self.host["image"], np.ascontiguousarray(img_chw, dtype=np.float32).ravel())
        np.copyto(self.host["lidar"], np.ascontiguousarray(lidar_vec, dtype=np.float32).ravel())
        cuda.memcpy_htod_async(self.device["image"], self.host["image"], self.stream)
        cuda.memcpy_htod_async(self.device["lidar"], self.host["lidar"], self.stream)
        self.context.execute_async_v2(bindings=self.bindings,
                                      stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.host[self._out_name], self.device[self._out_name],
                               self.stream)
        self.stream.synchronize()
        out = self.host[self._out_name]
        return float(out[0]), float(out[1]), float(out[2])


class Pca9685Actuator:
    """Servo e ESC no PCA9685. O ESC so sai de neutro se ESC_ARMADO for True."""

    def __init__(self):
        self.pwm = Adafruit_PCA9685.PCA9685(address=I2C_ADDRESS, busnum=I2C_BUSNUM)
        self.pwm.set_pwm_freq(PWM_FREQ)
        self.period_us = 1000000.0 / PWM_FREQ

    def _set_us(self, channel, microseconds):
        steps = int(microseconds / (self.period_us / 4096))
        self.pwm.set_pwm(channel, 0, max(0, min(4095, steps)))

    def set_servo_us(self, us):
        self._set_us(SERVO_CHANNEL, us)

    def set_esc_us(self, us):
        if not ESC_ARMADO:
            us = ESC_NEUTRAL_US
        self._set_us(ESC_CHANNEL, us)

    def safe_state(self):
        """Servo ao centro, ESC em neutro. Chamado na saida, aconteca o que acontecer."""
        self.set_servo_us(STEER_CENTER_US)
        self._set_us(ESC_CHANNEL, ESC_NEUTRAL_US)


def main():
    p = argparse.ArgumentParser(description="Runtime do modelo no carro (Fase 6b)")
    p.add_argument("--engine", required=True, help="Arquivo .engine do TensorRT")
    p.add_argument("--config", required=True, help="Sidecar .json do modelo")
    p.add_argument("--out", required=True, help="Diretorio do log desta corrida")
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--crop-frac", type=float, default=0.48,
                   help="Fracao central da imagem (lente de 130 vs 62.2 do treino). "
                        "0.48 e um CHUTE a calibrar; 1.0 desliga o recorte.")
    p.add_argument("--scale", type=float, default=12.0, help="Escala do modelo (1:12)")
    p.add_argument("--jpeg-every", type=int, default=10)
    p.add_argument("--cruise-us", type=int, default=ESC_NEUTRAL_US,
                   help="PWM CONSTANTE do ESC. %d = parado (padrao); o ESC so move a "
                        "partir de %d; maximo %d. Exige ESC_ARMADO=True no arquivo."
                        % (ESC_NEUTRAL_US, ESC_MIN_MOVE_US, ESC_MAX_US))
    p.add_argument("--stop-dist", type=float, default=0.25,
                   help="Parada de emergencia: metros no cone frontal (padrao 0.25)")
    a = p.parse_args()

    cfg = load_model_config(a.config)
    max_range = car_max_range(cfg, scale=a.scale)
    print("modelo: fov={0} deg  n_sectors={1}  max_range no carro={2:.3f} m"
          .format(cfg["fov_deg"], cfg["n_sectors"], max_range))
    anda = ESC_ARMADO and a.cruise_us >= ESC_MIN_MOVE_US
    print("ESC ARMADO: {0}   cruise={1}us   parada de emergencia: {2:.2f} m"
          .format(ESC_ARMADO, a.cruise_us, a.stop_dist))
    if not ESC_ARMADO:
        print("O carro NAO anda (ESC_ARMADO=False). Rodas no ar mesmo assim.")
    elif a.cruise_us < ESC_MIN_MOVE_US:
        print("O carro NAO anda: --cruise-us {0} esta abaixo de {1} (zona morta do ESC)."
              .format(a.cruise_us, ESC_MIN_MOVE_US))
    if anda:
        print("!!! O CARRO VAI ANDAR a {0}us constantes. Kill switch na mao. !!!"
              .format(a.cruise_us))

    camera = CsiCamera()
    lidar = SerialLidar()
    engine = TrtEngine(a.engine)
    actuator = Pca9685Actuator()
    actuator.safe_state()
    logger = RunLogger(a.out, meta={
        "fov_deg": cfg["fov_deg"], "n_sectors": cfg["n_sectors"],
        "max_range_m_car": max_range, "scale": a.scale,
        "crop_frac": a.crop_frac, "esc_armado": ESC_ARMADO,
        "cruise_us": a.cruise_us, "stop_dist_m": a.stop_dist,
        "engine": os.path.basename(a.engine),
    }, jpeg_every=a.jpeg_every)

    loop = DriveLoop(camera=camera, lidar=lidar, engine=engine, actuator=actuator,
                     logger=logger, fov_deg=cfg["fov_deg"], max_range=max_range,
                     crop_frac=a.crop_frac, n_sectors=cfg["n_sectors"],
                     cruise_us=a.cruise_us, stop_dist_m=a.stop_dist)

    t_end = time.monotonic() + a.seconds
    n, t_report = 0, time.monotonic()
    try:
        while time.monotonic() < t_end:
            tele = loop.step()
            n += 1
            if time.monotonic() - t_report >= 1.0:
                fps = n / (time.monotonic() - t_report)
                print("fps={0:5.1f}  steer={1:+.3f}  servo={2}us  esc={3}us{4}  scan={5}"
                      .format(fps, tele["steer"], tele["servo_us"], tele["esc_us"],
                              "  PARADA" if tele["blocked"] else "",
                              "ok" if tele["has_scan"] else "AGUARDANDO"))
                n, t_report = 0, time.monotonic()
    except KeyboardInterrupt:
        print("\ninterrompido")
    finally:
        actuator.safe_state()
        logger.close()
        camera.close()
        lidar.close()
        print("estado seguro. log em {0}".format(a.out))


if __name__ == "__main__":
    main()
