#!/usr/bin/env python3
"""
PWM Signal Reader para Jetson Nano
===================================
Le sinais PWM do receptor RC e ESC do WLToys 124017 V2
para entender os comandos de steering e throttle.

COMO CONECTAR:
--------------
Cada cabo de 3 pinos do RC segue o padrao:
  - Marrom/Preto = GND
  - Vermelho     = VCC (5-6V) — NAO CONECTAR NO JETSON!
  - Laranja/Branco = SINAL (PWM)

                ┌─────────────────────────────────────┐
                │          JETSON NANO (GPIO)          │
                │                                      │
                │   Pin 6 (GND) ─── GND receptor       │
                │   Pin 6 (GND) ─── GND ESC/servo      │
                │                                      │
                │   Pin 32 (GPIO12) ─── Sinal CH1      │
                │                       (steering)     │
                │                                      │
                │   Pin 18 (GPIO24) ─── Sinal CH2      │
                │                       (throttle)     │
                │                                      │
                └─────────────────────────────────────┘

IMPORTANTE - PROTECAO DE VOLTAGEM:
Se o sinal for 5V (maioria dos receptores RC), use um
divisor de tensao ANTES de conectar no GPIO do Jetson:

  Sinal 5V ──[1kΩ]──┬──> GPIO Jetson (3.3V safe)
                     │
                   [2kΩ]
                     │
                    GND

Sem isso, voce pode QUEIMAR o GPIO do Jetson Nano.

COMO TESTAR SEM DIVISOR (mais seguro):
Meca a voltagem do sinal com um multimetro primeiro.
Se for ~3.3V, pode conectar direto. Se for ~5V, use o divisor.

USO:
  python3 pwm_reader.py
  python3 pwm_reader.py --pin1 32 --pin2 18
  python3 pwm_reader.py --single 32        # ler so 1 canal
"""

import time
import sys
import argparse
import threading
from collections import deque

# ============================================================
# Tenta importar Jetson.GPIO — se nao tiver, roda em modo mock
# pra voce poder testar a logica no PC tambem
# ============================================================
try:
    import Jetson.GPIO as GPIO
    MOCK_MODE = False
    print("[OK] Jetson.GPIO encontrado — modo real")
except ImportError:
    MOCK_MODE = True
    print("[!!] Jetson.GPIO nao encontrado — rodando em modo SIMULADO")
    print("     (instale com: sudo pip3 install Jetson.GPIO)")
    print("     Ou rode direto no Jetson Nano.\n")


# ============================================================
# Configuracao dos pinos (BOARD numbering)
# ============================================================
# Pinagem do Jetson Nano (header J41):
#
#   Pin 1  (3.3V)    Pin 2  (5V)
#   Pin 3  (SDA)     Pin 4  (5V)
#   Pin 5  (SCL)     Pin 6  (GND)  <-- usar este GND
#   Pin 7  (GPIO216) Pin 8  (TXD)
#   Pin 9  (GND)     Pin 10 (RXD)
#   Pin 11 (GPIO50)  Pin 12 (GPIO79)
#   Pin 13 (GPIO14)  Pin 14 (GND)
#   Pin 15 (GPIO194) Pin 16 (GPIO232)
#   Pin 17 (3.3V)    Pin 18 (GPIO15)  <-- CH2 default
#   Pin 19 (SPI_MO)  Pin 20 (GND)
#   Pin 21 (SPI_MI)  Pin 22 (GPIO13)
#   Pin 23 (SPI_CK)  Pin 24 (SPI_CS0)
#   Pin 25 (GND)     Pin 26 (SPI_CS1)
#   Pin 27 (SDA_1)   Pin 28 (SCL_1)
#   Pin 29 (GPIO149) Pin 30 (GND)
#   Pin 31 (GPIO200) Pin 32 (GPIO168) <-- CH1 default
#   Pin 33 (GPIO38)  Pin 34 (GND)
#   Pin 35 (GPIO76)  Pin 36 (GPIO51)
#   Pin 37 (GPIO12)  Pin 38 (GPIO77)
#   Pin 39 (GND)     Pin 40 (GPIO78)

DEFAULT_PIN_CH1 = 32   # steering (receptor -> servo)
DEFAULT_PIN_CH2 = 18   # throttle (receptor -> ESC)


class PWMChannel:
    """Le e analisa o sinal PWM de um pino GPIO."""

    def __init__(self, pin, name="CH"):
        self.pin = pin
        self.name = name
        self.rise_time = 0
        self.pulse_width_us = 0      # largura do pulso em microsegundos
        self.period_us = 0           # periodo total
        self.last_rise = 0
        self.lock = threading.Lock()

        # Historico para media movel (suaviza ruido)
        self.history = deque(maxlen=20)

        # Estatisticas
        self.min_pw = 9999
        self.max_pw = 0
        self.sample_count = 0

    def _callback(self, channel):
        """Callback chamado em cada borda (rising/falling) do sinal."""
        now = time.monotonic()

        # Ler o estado atual do pino
        state = GPIO.input(self.pin)

        with self.lock:
            if state == GPIO.HIGH:
                # Borda de subida — inicio do pulso
                if self.last_rise > 0:
                    self.period_us = (now - self.last_rise) * 1_000_000
                self.rise_time = now
                self.last_rise = now
            else:
                # Borda de descida — fim do pulso
                if self.rise_time > 0:
                    pw = (now - self.rise_time) * 1_000_000
                    # Filtro basico: so aceita pulsos entre 500us e 2500us
                    if 500 < pw < 2500:
                        self.pulse_width_us = pw
                        self.history.append(pw)
                        self.sample_count += 1
                        self.min_pw = min(self.min_pw, pw)
                        self.max_pw = max(self.max_pw, pw)

    def get_data(self):
        """Retorna os dados atuais do canal."""
        with self.lock:
            pw = self.pulse_width_us
            period = self.period_us
            count = self.sample_count

        freq = 1_000_000 / period if period > 0 else 0
        avg_pw = sum(self.history) / len(self.history) if self.history else 0

        return {
            "pulse_width_us": pw,
            "avg_pulse_width_us": avg_pw,
            "frequency_hz": freq,
            "period_us": period,
            "samples": count,
            "min_us": self.min_pw if self.min_pw < 9999 else 0,
            "max_us": self.max_pw,
        }


def pw_to_normalized(pw_us, center=1500, range_us=500):
    """
    Converte largura de pulso para valor normalizado.

    Para steering: -1.0 (esquerda) a +1.0 (direita)
    Para throttle: -1.0 (re/freio) a +1.0 (frente)
    Centro (1500us) = 0.0
    """
    if pw_us == 0:
        return 0.0
    val = (pw_us - center) / range_us
    return max(-1.0, min(1.0, val))


def pw_to_bar(pw_us, center=1500, width=30):
    """Cria uma barra visual do valor do PWM."""
    if pw_us == 0:
        return " " * width + "│" + " " * width
    
    normalized = pw_to_normalized(pw_us, center)
    half = width
    pos = int(normalized * half)
    
    bar = list(" " * half + "│" + " " * half)
    if pos > 0:
        for i in range(half, half + pos):
            if i < len(bar):
                bar[i] = "█"
    elif pos < 0:
        for i in range(half + pos, half):
            if 0 <= i < len(bar):
                bar[i] = "█"
    
    return "".join(bar)


def interpret_channel(name, pw_us):
    """Interpreta o que o sinal significa no contexto RC."""
    norm = pw_to_normalized(pw_us)
    
    if "steer" in name.lower() or "ch1" in name.lower():
        if norm < -0.1:
            return f"ESQUERDA ({norm:+.2f})"
        elif norm > 0.1:
            return f"DIREITA  ({norm:+.2f})"
        else:
            return f"CENTRO   ({norm:+.2f})"
    else:
        if norm < -0.1:
            return f"RE/FREIO ({norm:+.2f})"
        elif norm > 0.1:
            return f"FRENTE   ({norm:+.2f})"
        else:
            return f"NEUTRO   ({norm:+.2f})"


def clear_screen():
    """Limpa o terminal."""
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def print_header():
    """Cabecalho do monitor."""
    print("=" * 72)
    print("  PWM SIGNAL READER — WLToys 124017 V2 + Jetson Nano")
    print("  TCC: Digital Twins para Veiculos Autonomos")
    print("=" * 72)


def run_monitor(channels, refresh_rate=20):
    """Loop principal que mostra os dados em tempo real."""
    interval = 1.0 / refresh_rate
    
    print("\n  Monitorando sinais PWM... (Ctrl+C para sair)\n")
    print("  Mexa nos controles do radio para ver os valores mudarem.\n")
    time.sleep(1)

    try:
        while True:
            clear_screen()
            print_header()
            print()

            for ch in channels:
                data = ch.get_data()
                pw = data["avg_pulse_width_us"]
                freq = data["frequency_hz"]
                raw_pw = data["pulse_width_us"]
                samples = data["samples"]
                min_pw = data["min_us"]
                max_pw = data["max_us"]

                norm = pw_to_normalized(pw)
                bar = pw_to_bar(pw)
                interp = interpret_channel(ch.name, pw)

                print(f"  ┌─ {ch.name} (Pin {ch.pin}) ──────────────────────────────────")
                print(f"  │")

                if samples == 0:
                    print(f"  │  ⚠  SEM SINAL — verifique a conexao!")
                    print(f"  │     - Fio de sinal no pino {ch.pin}?")
                    print(f"  │     - GND conectado?")
                    print(f"  │     - Radio/receptor ligado?")
                else:
                    print(f"  │  Pulso:    {pw:7.1f} μs  (raw: {raw_pw:.0f} μs)")
                    print(f"  │  Freq:     {freq:7.1f} Hz  (esperado: ~50 Hz)")
                    print(f"  │  Range:    {min_pw:.0f} — {max_pw:.0f} μs")
                    print(f"  │  Valor:    {norm:+.3f}     {interp}")
                    print(f"  │  Amostras: {samples}")
                    print(f"  │")
                    print(f"  │  -1.0          0.0          +1.0")
                    print(f"  │  [{bar}]")

                print(f"  │")
                print(f"  └────────────────────────────────────────────────────")
                print()

            # Secao de mapeamento para o carro autonomo
            print("  ┌─ MAPEAMENTO PARA O CARRO AUTONOMO ──────────────────")
            print("  │")
            
            for ch in channels:
                data = ch.get_data()
                pw = data["avg_pulse_width_us"]
                norm = pw_to_normalized(pw)
                
                if "steer" in ch.name.lower() or "ch1" in ch.name.lower():
                    print(f"  │  Steering:  {norm:+.3f}  →  set_steering({norm:+.3f})")
                else:
                    if norm >= 0:
                        print(f"  │  Throttle:  {norm:.3f}   →  set_throttle({norm:.3f}), set_brake(0.0)")
                    else:
                        print(f"  │  Brake:     {abs(norm):.3f}   →  set_throttle(0.0), set_brake({abs(norm):.3f})")

            print("  │")
            print("  └────────────────────────────────────────────────────")
            print()
            print("  [Ctrl+C para sair]  [Mexa no controle para ver mudancas]")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n  Parando...")


def run_logger(channels, duration=10, rate=20):
    """Grava os dados em CSV para analise posterior."""
    import csv
    
    filename = f"pwm_log_{int(time.time())}.csv"
    interval = 1.0 / rate
    
    headers = ["timestamp_s"]
    for ch in channels:
        headers += [f"{ch.name}_pw_us", f"{ch.name}_normalized", f"{ch.name}_freq_hz"]
    
    print(f"\n  Gravando {duration}s de dados em {filename}...")
    print(f"  Mexa nos controles durante a gravacao!\n")
    
    start = time.monotonic()
    rows = []
    
    try:
        while (time.monotonic() - start) < duration:
            elapsed = time.monotonic() - start
            row = [f"{elapsed:.4f}"]
            
            for ch in channels:
                data = ch.get_data()
                pw = data["avg_pulse_width_us"]
                row.append(f"{pw:.1f}")
                row.append(f"{pw_to_normalized(pw):.4f}")
                row.append(f"{data['frequency_hz']:.1f}")
            
            rows.append(row)
            
            # Progresso
            pct = elapsed / duration * 100
            bar_len = 40
            filled = int(bar_len * elapsed / duration)
            bar = "█" * filled + "░" * (bar_len - filled)
            sys.stdout.write(f"\r  [{bar}] {pct:.0f}%  ({elapsed:.1f}s / {duration}s)")
            sys.stdout.flush()
            
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\n  Gravacao interrompida.")
    
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    
    print(f"\n\n  Salvo: {filename} ({len(rows)} amostras)")
    print(f"  Abra no Excel/Google Sheets ou plote com matplotlib.")
    return filename


def run_mock_demo():
    """Demo simulado quando nao tem Jetson.GPIO."""
    import math
    
    print("\n" + "=" * 72)
    print("  MODO DEMONSTRACAO (sem Jetson.GPIO)")
    print("  Simulando sinais PWM para voce ver como o script funciona")
    print("=" * 72)
    print()

    t = 0
    try:
        while True:
            clear_screen()
            print_header()
            print("  [MODO SIMULADO — conecte no Jetson Nano para dados reais]\n")

            # Simula steering oscilando e throttle constante
            steer_pw = 1500 + 400 * math.sin(t * 0.5)
            throttle_pw = 1500 + 200 * math.sin(t * 0.3) + 100

            for name, pw in [("CH1-Steering", steer_pw), ("CH2-Throttle", throttle_pw)]:
                norm = pw_to_normalized(pw)
                bar = pw_to_bar(pw)
                interp = interpret_channel(name, pw)

                print(f"  ┌─ {name} ──────────────────────────────────────")
                print(f"  │  Pulso:  {pw:7.1f} μs")
                print(f"  │  Freq:     50.0 Hz")
                print(f"  │  Valor:  {norm:+.3f}     {interp}")
                print(f"  │")
                print(f"  │  -1.0          0.0          +1.0")
                print(f"  │  [{bar}]")
                print(f"  └────────────────────────────────────────────────")
                print()

            print("  [Ctrl+C para sair]")
            time.sleep(0.05)
            t += 0.05

    except KeyboardInterrupt:
        print("\n\n  Demo encerrada.")


def main():
    parser = argparse.ArgumentParser(
        description="Le sinais PWM do receptor RC no Jetson Nano",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python3 pwm_reader.py                    # Ler 2 canais (pins 32 e 18)
  python3 pwm_reader.py --single 32        # Ler so 1 canal
  python3 pwm_reader.py --log 30           # Gravar 30s de dados em CSV
  python3 pwm_reader.py --pin1 32 --pin2 18
        """
    )
    parser.add_argument("--pin1", type=int, default=DEFAULT_PIN_CH1,
                        help=f"Pino BOARD do CH1/steering (default: {DEFAULT_PIN_CH1})")
    parser.add_argument("--pin2", type=int, default=DEFAULT_PIN_CH2,
                        help=f"Pino BOARD do CH2/throttle (default: {DEFAULT_PIN_CH2})")
    parser.add_argument("--single", type=int, default=None,
                        help="Ler apenas 1 canal (numero do pino)")
    parser.add_argument("--log", type=int, default=None,
                        help="Gravar N segundos de dados em CSV")
    parser.add_argument("--rate", type=int, default=20,
                        help="Taxa de atualizacao do display (Hz)")

    args = parser.parse_args()

    # Se nao tem GPIO, roda demo
    if MOCK_MODE:
        run_mock_demo()
        return

    # Configurar GPIO
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)

    channels = []

    try:
        if args.single:
            # Modo canal unico
            pin = args.single
            GPIO.setup(pin, GPIO.IN)
            ch = PWMChannel(pin, "CH1-Signal")
            GPIO.add_event_detect(pin, GPIO.BOTH, callback=ch._callback)
            channels.append(ch)
            print(f"  Lendo 1 canal: Pin {pin}")
        else:
            # Modo dois canais
            GPIO.setup(args.pin1, GPIO.IN)
            GPIO.setup(args.pin2, GPIO.IN)

            ch1 = PWMChannel(args.pin1, "CH1-Steering")
            ch2 = PWMChannel(args.pin2, "CH2-Throttle")

            GPIO.add_event_detect(args.pin1, GPIO.BOTH, callback=ch1._callback)
            GPIO.add_event_detect(args.pin2, GPIO.BOTH, callback=ch2._callback)

            channels = [ch1, ch2]
            print(f"  Lendo 2 canais: Steering=Pin {args.pin1}, Throttle=Pin {args.pin2}")

        # Esperar um pouco para os primeiros callbacks
        print("  Aguardando sinais...\n")
        time.sleep(0.5)

        if args.log:
            run_logger(channels, duration=args.log, rate=args.rate)
        else:
            run_monitor(channels, refresh_rate=args.rate)

    finally:
        GPIO.cleanup()
        print("  GPIO limpo. Ate a proxima!\n")


if __name__ == "__main__":
    main()