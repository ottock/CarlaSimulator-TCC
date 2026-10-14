# Controle 8BitDo no Jetson — o que funciona e o próximo passo

> Estado em **2026-10-13**. Tudo o que está marcado como "visto" foi medido no Jetson do carro
> nesse dia. O que não foi testado está dito como tal.

## 1. Resumo

- **Controle:** 8BitDo Ultimate 2C Wireless, com o adaptador USB de 2.4G. No `lsusb` ele aparece
  como `2dc8:310a`.
- **Funciona no Jetson**, desde que o programa converse direto com o adaptador pela USB
  (biblioteca `pyusb`). O caminho normal do Linux (`/dev/input/js0`) **não entrega eventos**
  neste kernel; a seção 5 conta por quê.
- **Largar e parar pelo controle (seção 6):** `hardware/largada_controle.py` inicia a corrida no
  Start e para no Back. Está escrito e testado no PC, e ainda não foi rodado no Jetson.

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
| 2 | bits 0–3: direcional (cima, baixo, esquerda, direita); bit 4: Start; bit 5: Back; bits 6–7: L3, R3 | a conferir |
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

## 6. Largar e parar pelo controle

**Estado em 2026-10-13: escrito e testado no PC; ainda NÃO rodado no Jetson.** O primeiro uso é
com as rodas no ar.

Definido pelo Rafael no mesmo dia: **Start inicia a corrida** e **Back, o botão ao lado do
Start, para**. Só esses dois botões; não há segundo botão de largada.

```bash
cd ~/CarlaSimulator-TCC
python3 hardware/largada_controle.py
```

"Corrida" é o `hardware/jetson_runtime.py` rodando o modelo, com o comando do topo da seção 2
de `docs/ESTADO_IA.md`. O script lança esse mesmo comando; ele não mexe no modelo, no esterço
nem no ESC. Qualquer flag que ele não conhece vai direto para o runtime (por exemplo
`--seconds 120`).

O que ele faz:

- **Start** lança o runtime, gravando numa pasta nova em `runs/`. Com uma corrida em andamento,
  Start não faz nada.
- **Back** manda ao runtime o mesmo sinal do Ctrl+C. O runtime já centraliza o servo e põe o
  ESC em neutro ao sair.
- Quando a corrida termina, por Back ou pelo fim do tempo, ele volta a esperar o Start.
- **Ctrl+C no script, ou o adaptador sumir da USB,** para a corrida e sai.

Três regras que estão no código de propósito (`src/ai/car/largada.py`):

1. **Conta o instante em que o botão é apertado, não o botão estar apertado.** Como o adaptador
   não zera o estado quando o controle desliga (seção 4), um Start "preso" largaria uma corrida
   atrás da outra.
2. **Depois de um sinal de parada, outro só sai passados 3 segundos.** Um segundo Ctrl+C pode
   cair dentro da limpeza do runtime e interrompê-la antes de o ESC ir a neutro.
3. **O runtime nunca é morto à força.** Matar pula a limpeza, e o PCA9685 continua mandando o
   último pulso: o carro seguiria andando. Se o runtime não responder, o script avisa na tela e
   o que resta é a chave do carro.

O que ele não resolve:

- **O modelo é carregado de novo a cada corrida.** Entre apertar Start e o carro andar passa o
  mesmo tempo de quando o comando é digitado (estimado em 1 a 2 minutos, nunca cronometrado).
  Resolver isso é manter o modelo carregado entre uma corrida e outra, e é outro passo.
- **No fim da pista o carro para, mas o runtime continua** até acabar o tempo (`--seconds`,
  padrão 60). É preciso apertar Back antes da próxima largada.
- **Se o controle desligar no meio da corrida, não há Back.** O carro para sozinho no fim da
  pista, na parada frontal ou no fim do tempo.

### Primeiro teste, com as rodas no ar

Antes:

- com `hardware/pad_teste.py`, conferir que Start e o botão ao lado aparecem como `start` e
  `back`. Se os nomes vierem trocados ou diferentes, não rode o script: avise.
- decidir entre `sudo` e a regra de permissão da seção 7. Com `sudo` o script avisa na tela que
  a corrida vai rodar como root, o que nunca foi testado.

Depois, nesta ordem:

1. Apertar Start. Deve aparecer a linha do comando e `CORRIDA LANCADA`, e em seguida a saída
   normal do runtime. As rodas giram quando o modelo terminar de carregar.
2. Apertar Start de novo com a corrida em andamento. Não deve acontecer nada.
3. Apertar Back. O runtime deve imprimir `interrompido` e `estado seguro`, as rodas param e o
   script imprime `corrida encerrada`.
4. Start outra vez, e Back **durante o carregamento** do modelo. A corrida deve encerrar sem as
   rodas terem girado. Nesse caso o runtime mostra um erro `KeyboardInterrupt` na tela em vez de
   `estado seguro`; é esperado, porque ele foi interrompido antes de chegar ao laço.
5. Start, esperar as rodas girarem, e apertar Back duas vezes seguidas. As rodas devem parar
   igual.
6. Start, e Ctrl+C no terminal do script. As rodas param e o script sai.

O que foi testado sem o carro: a decodificação dos botões contra relatórios reais do adaptador,
as três regras acima e o caminho completo Start, Back, fim (`tests/test_car_gamepad.py` e
`tests/test_car_largada.py`), com um adaptador e um runtime de mentira. **Não foi testado:** o
sinal de verdade chegando ao runtime, e as posições de Start e Back no relatório.

## 7. Detalhes a mais (pendências, sem prioridade agora)

- **Proteção se o controle desligar.** No modo autônomo, perder o controle só tira o botão de
  parar à distância: o carro continua parando sozinho no fim da pista, no limite de tempo e na
  parada frontal. Uma proteção barata é tratar **Home apertado como parar**, já que ele aparece
  antes de o controle desligar. Se um dia o visitante dirigir pelo controle, falta uma proteção
  maior: parar o carro quando não chegar relatório novo por um tempo curto. Bateria acabando e
  interferência não avisam.
- **Teste de silêncio** (`hardware/pad_silencio.py`, ainda não rodado). Mede por quanto tempo o
  adaptador fica mudo com alguém segurando um comando. Decide qual comando pode ser o "segurar
  para andar" e qual o tempo de silêncio tolerado.
- **Conferir o que está "a conferir" na seção 3:** A, B, Start, Back, gatilhos, analógico
  direito e o sentido dos eixos.
- **Religar o controle depois de desligado:** os dados voltam sem reenviar os dois comandos? Só
  testamos a volta depois do silêncio normal.
- **Rodar sem `sudo`** (é o que evita a corrida rodar como root no script da seção 6). A regra
  abaixo foi sugerida e não confirmada. Depois de criá-la, é preciso tirar e recolocar o adaptador.
  ```bash
  echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="2dc8", ATTR{idProduct}=="310a", MODE="0666"' | sudo tee /etc/udev/rules.d/99-8bitdo.rules
  sudo udevadm control --reload
  ```
- **Reiniciar o Jetson com o adaptador plugado** e repetir o teste. É assim que ele vai ligar
  depois de cada troca de bateria.
- **Descobrir qual dos dois comandos destrava o envio** (seção 5).
- **Bateria do controle:** o fabricante informa 19 horas em 2.4G e 2 horas de carga. Não medido.
