import Jetson.GPIO as GPIO
import time

ESC_PIN = 32
GPIO.setmode(GPIO.BOARD)
GPIO.setup(ESC_PIN, GPIO.OUT)

pwm = GPIO.PWM(ESC_PIN, 50)

# === ARMING ===
print("Armando ESC... (neutro por 3s)")
pwm.start(7.5)
time.sleep(3)
print("ESC armado!")

# === ACELERACAO LEVE ===
print("Acelerando leve (8.0)...")
pwm.ChangeDutyCycle(8.0)
time.sleep(2)

print("Neutro")
pwm.ChangeDutyCycle(7.5)
time.sleep(1)

# === FREIO ===
# Na maioria dos ESCs RC: abaixo do neutro = freio
print("Freio (7.0)...")
pwm.ChangeDutyCycle(7.0)
time.sleep(1)

print("Neutro")
pwm.ChangeDutyCycle(7.5)
time.sleep(1)

# === FREIO MAIS FORTE ===
print("Freio forte (6.0)...")
pwm.ChangeDutyCycle(6.0)
time.sleep(1)

print("Neutro")
pwm.ChangeDutyCycle(7.5)
time.sleep(1)

pwm.stop()
GPIO.cleanup()
print("Teste concluido!")