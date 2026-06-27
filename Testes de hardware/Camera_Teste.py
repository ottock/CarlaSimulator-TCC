#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Teste_Camera_CSI.py
# Validacao da camera CSI IMX219 - Etapa 1.3
#
# Mostra o video ao vivo numa janela. Requer monitor HDMI no Jetson
# (usa janela grafica; nao funciona headless via SSH puro sem X forwarding).
#
# Ambiente: Jetson Nano (Yahboom) / JetPack 4.6.x / Python 3.6
#   - A camera CSI NAO abre como webcam USB comum. Entra pela pipeline
#     GStreamer via nvarguscamerasrc. O OpenCV do JetPack ja vem com
#     suporte a GStreamer compilado.
#
# Sair: tecla 'q' com a janela em foco, ou Ctrl+C no terminal.

import cv2

# ---------------------------------------------------------------------------
# Configuracao da camera
# ---------------------------------------------------------------------------
SENSOR_ID = 0          # porta CSI (0 = primeira porta)

# Resolucao de CAPTURA do sensor (o que a IMX219 entrega).
# 1280x720 @ 60fps e um modo nativo comum e leve para teste.
CAP_WIDTH  = 1280
CAP_HEIGHT = 720
CAP_FPS    = 60

# Resolucao de EXIBICAO (redimensiona pra janela). Menor = mais leve.
DISP_WIDTH  = 960
DISP_HEIGHT = 540

# A IMX219 costuma vir de cabeca pra baixo em muitos modulos.
# flip-method=2 gira 180 graus. Se a imagem aparecer invertida,
# troque para 0 (sem giro) ou outro valor (0-7).
FLIP_METHOD = 2


def build_gstreamer_pipeline():
    """Monta a string de pipeline GStreamer para a camera CSI."""
    # .format() em vez de f-string por compatibilidade com Python 3.6.
    return (
        "nvarguscamerasrc sensor-id={sensor_id} ! "
        "video/x-raw(memory:NVMM), width=(int){cap_w}, height=(int){cap_h}, "
        "framerate=(fraction){fps}/1 ! "
        "nvvidconv flip-method={flip} ! "
        "video/x-raw, width=(int){disp_w}, height=(int){disp_h}, "
        "format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
    ).format(
        sensor_id=SENSOR_ID,
        cap_w=CAP_WIDTH, cap_h=CAP_HEIGHT, fps=CAP_FPS,
        flip=FLIP_METHOD,
        disp_w=DISP_WIDTH, disp_h=DISP_HEIGHT,
    )


def main():
    pipeline = build_gstreamer_pipeline()
    print("Abrindo camera CSI com a pipeline:")
    print(pipeline)
    print("")

    # CAP_GSTREAMER e essencial: diz ao OpenCV pra usar GStreamer,
    # nao o backend de webcam USB (que NAO funciona com a CSI).
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("ERRO: nao consegui abrir a camera.")
        print("Cheque: 1) cabo flat da CSI bem encaixado;")
        print("        2) 'ls /dev/video0' existe?;")
        print("        3) o teste 'gst-launch-1.0 nvarguscamerasrc ...' funciona?")
        return

    print("Camera aberta. Janela ao vivo. Tecla 'q' para sair.")
    janela = "Teste Camera CSI - IMX219"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Falha ao ler frame. Encerrando.")
                break
            cv2.imshow(janela, frame)
            # 1ms de espera; 'q' encerra.
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera liberada. Encerrado.")


if __name__ == "__main__":
    main()