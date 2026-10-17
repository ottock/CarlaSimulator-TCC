# Controle 8BitDo no Jetson — o que funciona e o próximo passo

> Começado em **2026-10-13**, atualizado em **2026-10-17**. O que está marcado como "visto" foi
> medido no Jetson do carro. O que não foi testado está dito como tal.

## 1. Resumo

- **Controle:** 8BitDo Ultimate 2C Wireless, com o adaptador USB de 2.4G. No `lsusb` ele aparece
  como `2dc8:310a`.
- **Funciona no Jetson**, desde que o programa converse direto com o adaptador pela USB
  (biblioteca `pyusb`). O caminho normal do Linux (`/dev/input/js0`) **não entrega eventos**
  neste kernel; a seção 5 conta por quê.
- **O programa do controle (seções 6 e 8):** `hardware/controle_8bitdo.py` dirige o carro à mão
  e, no Start, larga a corrida autônoma; Back para. Os dois modos funcionaram no carro em
  programas separados em 2026-10-14. Juntos desde 2026-10-17, **ainda não rodado no Jetson**.

## 2. Como ler o controle

Uma vez, no Jetson:

```bash
sudo apt install python3-usb
```

A sequência que faz os dados chegarem:

```python
import usb.core

d = usb.core.find(idVendor=0x2dc8, idProduct=0x310a)
if d.is_kernel_driver_active(0):                 # só se alguém usou o new_id da seção 5
    d.detach_kernel_driver(0)
d.ctrl_transfer(0xC1, 0x01, 0x0100, 0x00, 20)    # mensagem de inicialização
d.write(0x05, bytes([0x01, 0x03, 0x02]), 1000)   # comando de LED do Xbox 360
relatorio = d.read(0x84, 64, 500)                # 20 bytes; dá USBError se nada mudou em 500 ms
```

Para conferir que está funcionando:

```bash
sudo python3 hardware/pad_teste.py
```

Com a luz do Home fixa, mexer o analógico esquerdo imprime linhas assim:

```
[] esq +0.84 +0.18 dir +0.00 +0.00 LT 0.00 RT 0.00
[] esq +1.00 +0.19 dir +0.00 +0.00 LT 0.00 RT 0.00
['home'] esq +1.00 +0.09 dir +0.00 +0.00 LT 0.00 RT 0.00
```

**Precisa de `sudo`.** Sem ele o erro é `Access denied (insufficient permissions)`. A regra de
permissão que dispensaria o `sudo` está na seção 7 e ainda não foi confirmada.

### Como o adaptador se apresenta

| Interface | Classe / sub / protocolo | Endpoints | O que é |
|---|---|---|---|
| 0 | `ff` / `5d` / `01` | `0x84` (entrada), `0x05` (saída) | controle no formato Xbox 360 — **é esta que usamos** |
| 1 | `03` / `01` / `01` | `0x82` | teclado HID, sem uso |
| 2 | `03` / `00` / `00` | `0x83`, `0x06` | HID genérico, sem uso |

## 3. Formato do relatório

O adaptador manda o relatório padrão de controle de Xbox 360, de 20 bytes. "A conferir" quer
dizer que a posição vem do padrão, mas ainda não apertamos esse comando olhando a saída.

| Bytes | Conteúdo | Situação |
|---|---|---|
| 0–1 | `00 14` (tipo e tamanho) | visto |
| 2 | bits 0–3: direcional (cima, baixo, esquerda, direita); bit 4: Start; bit 5: Back; bits 6–7: L3, R3 | Start e Back vistos (2026-10-14: o script larga e para a corrida no carro); o resto a conferir |
| 3 | bits 0–1: LB, RB; bit 2: Home; bits 4–7: A, B, X, Y | Home visto; o resto a conferir |
| 4, 5 | gatilhos LT e RT, de 0 a 255 | a conferir |
| 6–7, 8–9 | analógico esquerdo, lado e vertical (inteiro de 16 bits, byte baixo primeiro) | visto: vai de −1,00 a +1,00 e volta a 0 exato quando solto |
| 10–11, 12–13 | analógico direito, lado e vertical | a conferir |
| 14–19 | `10 1c 30 64 00 10`, sempre iguais | visto, sem uso |

O sentido dos eixos (direita e para cima positivos, pelo padrão) também está a conferir.

## 4. Comportamentos que o programa precisa conhecer

1. **O adaptador só manda relatório quando algo muda.** Com a mão parada não chega nada, e isso
   é normal. Ao mexer de novo, os dados voltam sozinhos, sem reenviar comando nenhum.
2. **Quando o controle desliga, o adaptador não avisa.** Ele não manda um relatório zerado: o
   último estado fica valendo. No teste, com o analógico segurado em `+1.00`, a última linha foi
   `['home'] esq +1.00` e depois silêncio.
3. **Por causa de 1 e 2, silêncio não distingue "mão parada" de "controle desligado".**
4. **O aperto do Home aparece antes de o controle desligar** (desligar exige segurar o Home).
5. **O adaptador aparece na USB mesmo com o controle desligado.** Achar o adaptador não quer
   dizer que o controle está ligado.

Luzes e chave do controle:

- **Luz do Home fixa:** ligado ao adaptador (visto). A chave traseira precisa estar em 2.4G.
- **Luz vermelha:** bateria. Fixa é carregando (visto). Piscando seria bateria fraca e apagada
  seria carga completa, segundo uma cópia do manual; não vimos esses dois estados.
- **Desligamento automático:** depois de 1 minuto sem conseguir ligação e de 15 minutos sem uso,
  segundo a mesma cópia do manual. Não testado.
- **Segurar o Home para desligar** não funcionou na primeira tentativa, com o controle no cabo de
  carga, e funcionou depois. A cópia do manual diz que ele não desliga enquanto está no cabo; não
  confirmamos que foi essa a causa.

## 5. O que não funcionou, para ninguém repetir

O kernel do Jetson é o 4.9 do JetPack 4.6. Na configuração dele: `CONFIG_JOYSTICK_XPAD=y`,
`CONFIG_INPUT_JOYDEV=y` e `CONFIG_JOYSTICK_XPAD_LEDS` desligado.

1. **Ligar o adaptador e esperar o `/dev/input/js0`:** não aparece. O driver `xpad` desse kernel
   não tem a 8BitDo na lista de fabricantes; ela só entrou em kernels bem mais novos.
2. **Forçar o driver** com `echo 2dc8 310a | sudo tee /sys/bus/usb/drivers/xpad/new_id`: o `js0`
   aparece (`Generic X-Box pad`, com 11 botões), mas **nenhum evento chega** ao mexer no
   controle, mesmo com a luz do Home fixa.
3. **Mandar à mão a mensagem de inicialização** que os kernels a partir de 2023 mandam: o
   adaptador responde com 20 bytes, e o `js0` continua mudo.

Os dados só chegaram lendo direto pela USB, depois da mensagem de inicialização **e** do comando
de LED (seção 2). **Não isolamos qual dos dois é o necessário.** O palpite é o comando de LED,
porque este kernel nunca o envia; é palpite, não medição.

O `new_id` escrito à mão vale até o próximo reinício. **Não crie regra de udev para ele:** faria
o driver que não funciona agarrar o adaptador a cada boot.

Fontes: [xpad do kernel 4.9](https://github.com/torvalds/linux/blob/v4.9/drivers/input/joystick/xpad.c),
[xpad atual](https://github.com/torvalds/linux/blob/master/drivers/input/joystick/xpad.c),
[inclusão do `310a` no kernel](https://lkml.iu.edu/hypermail/linux/kernel/2410.1/10849.html),
[mensagem de inicialização de 2023](https://lab.nexedi.cn/kirr/linux/-/commit/05e6329443734a2a0aefb845d74375081d6626b6),
[manual oficial, ligação por 2.4G](https://manual.8bitdo.com/ultimate-2c-wireless-controller/ultimate-2c-2.4g-windows-2.4g.html),
[cópia do manual com as luzes](https://teamplush.by/en/product/8bitdo-ultimate-2c/).

## 6. Largar e parar a corrida pelo controle

**Funcionou no carro em 2026-10-14**, como programa separado (`hardware/largada_controle.py`),
rodando sem `sudo`. **Desde 2026-10-17 faz parte do `hardware/controle_8bitdo.py`** (seção 8),
a pedido do Rafael; o programa separado foi removido.

Definido pelo Rafael: **Start inicia a corrida** e **Back, o botão ao lado do Start, para**.
Não há segundo botão de largada.

"Corrida" é o `hardware/jetson_runtime.py` rodando o modelo, com o comando do topo da seção 2
de `docs/ESTADO_IA.md`. O programa lança esse mesmo comando; não mexe no modelo, no esterço nem
no ESC do runtime. Cada corrida grava numa pasta nova em `runs/`, e qualquer flag que o
programa não conhece vai direto para o runtime (por exemplo `--seconds 120`).

Três regras que estão no código de propósito (`src/ai/car/largada.py`):

1. **Conta o instante em que o botão é apertado, não o botão estar apertado.** Como o adaptador
   não zera o estado quando o controle desliga (seção 4), um Start "preso" largaria uma corrida
   atrás da outra.
2. **Depois de um sinal de parada, outro só sai passados 3 segundos.** Um segundo Ctrl+C pode
   cair dentro da limpeza do runtime e interrompê-la antes de o ESC ir a neutro.
3. **O runtime nunca é morto à força.** Matar pula a limpeza, e o PCA9685 continua mandando o
   último pulso: o carro seguiria andando. Se o runtime não responder, o programa avisa na tela
   e o que resta é a chave do carro.

O que não resolve:

- **O modelo é carregado de novo a cada corrida.** Entre apertar Start e o carro andar passa o
  mesmo tempo de quando o comando é digitado (estimado em 1 a 2 minutos, nunca cronometrado).
  Resolver isso é manter o modelo carregado entre uma corrida e outra, e é outro passo.
- **No fim da pista o carro para, mas o runtime continua** até acabar o tempo (`--seconds`,
  padrão 60). É preciso apertar Back antes de largar de novo ou de voltar a dirigir.
- **Se o controle desligar no meio da corrida, não há Back.** O carro para sozinho no fim da
  pista, na parada frontal ou no fim do tempo.

**Não rode com `sudo`.** Em 2026-10-14, com `sudo`, o runtime parou no `import pycuda` com
`libcurand.so.10: cannot open shared object file`: o `sudo` limpa as variáveis de ambiente e,
muito provavelmente, o `LD_LIBRARY_PATH` que aponta para as bibliotecas do CUDA. O programa
recusa rodar como root. Ler o adaptador sem `sudo` exige a regra de permissão da seção 7.

## 7. Detalhes a mais (pendências, sem prioridade agora)

- **Proteção se o controle desligar.** No modo autônomo, perder o controle só tira o botão de
  parar à distância: o carro continua parando sozinho no fim da pista, no limite de tempo e na
  parada frontal. Na direção manual (seção 8) o Home já para o carro e há uma parada por
  silêncio; o tempo dela (0,5 s) ainda é um palpite.
- **Teste de silêncio** (`hardware/pad_silencio.py`, ainda não rodado). Mede por quanto tempo o
  adaptador fica mudo com alguém segurando um comando. É o dado para escolher o `--silencio-s`
  da direção manual.
- **Parada frontal na direção manual.** O `controle_8bitdo.py` não lê o LiDAR, então não tem a
  parada de 6 cm do runtime. Para visitante dirigir, ela deveria entrar antes.
- **Conferir o que está "a conferir" na seção 3:** A, B, Start, Back, gatilhos, analógico
  direito e o sentido dos eixos.
- **Religar o controle depois de desligado:** os dados voltam sem reenviar os dois comandos? Só
  testamos a volta depois do silêncio normal.
- **Regra de permissão para rodar sem `sudo`** (obrigatória: o programa recusa `sudo`, seção 6).
  Em uso no Jetson desde 2026-10-14. Depois de criá-la, é preciso tirar e recolocar o adaptador.
  ```bash
  echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="2dc8", ATTR{idProduct}=="310a", MODE="0666"' | sudo tee /etc/udev/rules.d/99-8bitdo.rules
  sudo udevadm control --reload
  ```
- **Reiniciar o Jetson com o adaptador plugado** e repetir o teste. É assim que ele vai ligar
  depois de cada troca de bateria.
- **Descobrir qual dos dois comandos destrava o envio** (seção 5).
- **Bateria do controle:** o fabricante informa 19 horas em 2.4G e 2 horas de carga. Não medido.

## 8. O programa do controle: dirigir à mão e largar a corrida

**Estado em 2026-10-17:** os dois modos funcionaram no carro em programas separados
(2026-10-14). **Juntos, no `hardware/controle_8bitdo.py`, ainda não rodaram no Jetson**: o
primeiro uso é com as rodas no ar. Roda só no terminal, sem monitor no carro, e **sem `sudo`**.

```bash
python3 hardware/controle_8bitdo.py
python3 hardware/controle_8bitdo.py --max-us 1650     # teto mais alto à mão
python3 hardware/controle_8bitdo.py --seconds 120     # flag que vai para o runtime
```

**A regra que manda: o PCA9685 tem um dono por vez.** Sem corrida, o dono é o controle manual.
Com corrida, é o runtime, e o programa não escreve nada na placa: dois processos mandando pulsos
ao mesmo servo e ESC fariam o carro obedecer a quem escreveu por último. A troca está em
`src/ai/car/controle_unico.py`.

| Comando | Sem corrida (à mão) | Com corrida |
|---|---|---|
| Analógico esquerdo | volante | nada: quem dirige é o modelo |
| RT | acelerador | nada |
| LT | ré | nada |
| Back | parada; para andar de novo, soltar os dois gatilhos | para a corrida |
| Home | parada, como o Back | nada |
| Start | estaciona o carro (centro e neutro) e lança a corrida | nada |
| Ctrl+C | estaciona e sai | para a corrida, espera ela sair, estaciona e sai |

Quando a corrida acaba, o modo manual volta **parado até os dois gatilhos serem soltos**: um RT
que ficou apertado durante a corrida não arranca o carro.

Detalhes do modo manual (lógica em `src/ai/car/manual.py`):

- **Volante:** proporcional. Centro 1500 µs; batentes 1800 µs (esquerda) e 1200 µs (direita),
  span 300. O modelo usa 350 (1850/1150 µs), e com 300 a curva mais fechada pode não sair à mão;
  o Rafael escolheu começar em 300 para testar (2026-10-14). `--span 350` iguala ao modelo. O
  teclado usa ±260, tudo ou nada.
- **Acelerador (RT):** proporcional ao curso, de 1600 µs, o mínimo que move o ESC, até 1630 µs.
  Era até 1700 µs, o máximo calibrado; no primeiro teste o Rafael achou rápido demais e baixou
  para 1630 (2026-10-14). A IA anda a 1610 fixos. Abaixo de 10% do curso conta como solto.
  `--max-us` muda o teto, até 1700.
- **Ré (LT):** 1400 µs fixos, como a seta para baixo do teclado.

Proteções:

- `ESC_ARMADO` no arquivo, como no `controle_teste.py` (`True`: o carro anda à mão). A corrida
  segue o `ESC_ARMADO` do `jetson_runtime.py`.
- Laço travado por mais de 0,25 s: ESC em neutro naquele quadro.
- **Silêncio:** andando à mão, 0,5 s sem relatório do controle param o carro, e ele volta no
  relatório seguinte. Existe porque o adaptador não avisa quando o controle desliga (seção 4).
  O preço: segurar o RT no fundo com o analógico parado também é silêncio e pode parar o carro
  numa reta. O tempo muda com `--silencio-s`.
- Adaptador sumiu da USB: para a corrida, se houver, estaciona e sai.
- **Fio do I2C que solta por um instante.** Visto em 2026-10-14: no meio da direção um jumper
  soltou, uma escrita no PCA9685 falhou com `Errno 121, Remote I/O error` e a primeira versão
  caiu; a escrita seguinte, a do estado seguro, já passou. Agora uma falha isolada não derruba
  o programa: o comando vai de novo no quadro seguinte, e a tela mostra `I2C FALHOU` e conta as
  falhas. Se elas duram 0,1 s seguidos, o programa sai, tenta o neutro 10 vezes e, se nenhuma
  passar, manda **desligar a chave do carro**: o PCA9685 continua mandando o último pulso
  sozinho. A lógica está em `src/ai/car/escrita_tolerante.py`. **O conserto de verdade é o fio.**
  O runtime autônomo não tem essa tolerância: lá uma falha encerra a corrida e tenta o neutro.
- Na corrida, as três regras da seção 6.

O que ele **não** tem: a parada frontal do LiDAR no modo manual. O runtime corta o motor quando
algo fica a menos de 6 cm à frente; à mão, quem dirige pode bater na parede.

### Primeiro teste, com as rodas no ar

1. `python3 hardware/controle_8bitdo.py`. Ele põe o ESC em neutro por 2 s antes de liberar.
2. À mão: analógico para a direita vira as rodas para a direita. **Se virar ao contrário, pare
   aqui.** A troca é `SINAL_DO_ANALOGICO` em `src/ai/car/manual.py`.
3. RT devagar, LT para a ré, e Back com o RT apertado: as rodas param e só voltam depois de
   soltar e apertar o RT de novo.
4. Com o RT apertado, **Start**: as rodas param na hora (o carro é estacionado) e aparece
   `CORRIDA LANCADA`. Mexer o analógico e o RT durante o carregamento do modelo não deve fazer
   nada.
5. Quando o modelo carregar, as rodas giram pela corrida. **Back**: o runtime imprime
   `interrompido` e `estado seguro`, e o programa imprime `corrida encerrada`.
6. Se o RT estiver apertado quando a corrida acabar, as rodas ficam paradas. Soltar e apertar o
   RT: volta o modo manual.
7. Start, e **Ctrl+C** no terminal durante a corrida: a corrida para, o carro é estacionado e o
   programa sai.

Testado no PC: `tests/test_car_manual.py`, `tests/test_car_controle_unico.py`,
`tests/test_car_largada.py`, `tests/test_car_escrita_tolerante.py` e
`tests/test_controle_8bitdo.py`. Este último roda o programa inteiro com placa, controle e runtime
de mentira e confere, entre outras coisas, que **nada chega à placa durante a corrida**. **Não
testado:** os dois modos juntos no carro.
