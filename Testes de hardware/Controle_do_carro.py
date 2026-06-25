import Jetson.GPIO as GPIO
import time

# Tenta pelo nome do pino no chip em vez do board
GPIO.setmode(GPIO.TEGRA_SOC)
GPIO.setup("SPI1_MOSI", GPIO.OUT)  # equivale ao PWM0

pwm = GPIO.PWM("SPI1_MOSI", 50)
pwm.start(7.5)

print("Neutro")
time.sleep(2)

print("Esquerda")
pwm.ChangeDutyCycle(5.0)
time.sleep(1)

print("Neutro")
pwm.ChangeDutyCycle(7.5)
time.sleep(1)

pwm.stop()
GPIO.cleanup()