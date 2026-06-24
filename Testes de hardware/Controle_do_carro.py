
import Jetson.GPIO as GPIO
import time

SERVO_PIN = 33
GPIO.setmode(GPIO.BOARD)
GPIO.setup(SERVO_PIN, GPIO.OUT)

pwm = GPIO.PWM(SERVO_PIN, 50)  # 50Hz = padrao servo
pwm.start(7.5)                  # neutro (reto)

print("Neutro (reto)")
time.sleep(2)

print("Esquerda")
pwm.ChangeDutyCycle(5.0)
time.sleep(1)

print("Neutro")
pwm.ChangeDutyCycle(7.5)
time.sleep(1)

print("Direita")
pwm.ChangeDutyCycle(10.0)
time.sleep(1)

print("Neutro")
pwm.ChangeDutyCycle(7.5)
time.sleep(1)

pwm.stop()
GPIO.cleanup()
print("Teste concluido!")