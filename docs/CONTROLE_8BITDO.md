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
  e, no Start, roda a corrida autônoma; Back para. Juntos, os dois modos funcionaram no carro em
  2026-10-17. No mesmo dia a corrida passou a rodar **dentro** do programa, com o modelo carregado
  uma vez, para largar e parar na hora; essa versão **ainda não rodou no Jetson**.

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
| 3 | bits 0–1: LB, RB; bit 2: Home; bits 4–7: A, B, X, Y | Home visto; o resto a conferir (LB + RB desligam o Jetson desde 2026-10-20: o primeiro uso confere os dois) |
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

Definido pelo Rafael: **Start inicia a corrida** e **Back, o botão ao lado do Start, para**.
Não há segundo botão de largada.

Como isso evoluiu:

1. **2026-10-14, programa separado** (`hardware/largada_controle.py`, removido): o Start lançava
   o `jetson_runtime.py` como outro programa e o Back mandava a ele o sinal do Ctrl+C. Funcionou
   no carro. Mas cada corrida carregava CUDA, câmera, LiDAR e modelo do zero (1 a 2 minutos,
   nunca cronometrado), e depois do Back o manual só voltava quando aquele programa terminava de
   fechar a câmera.
2. **2026-10-17, juntado ao `hardware/controle_8bitdo.py`** (seção 8), ainda lançando o runtime
   como outro programa. Funcionou no carro.
3. **2026-10-17, corrida dentro do programa**, a pedido do Rafael ("o objetivo é ser algo mais
   rápido"). O modelo, a câmera e o LiDAR são abertos **uma vez**, quando o programa abre. O
   Start só começa a chamar o laço de direção do runtime: o mesmo `DriveLoop`, montado pelas
   mesmas funções e com as mesmas opções que o `jetson_runtime.py` usa (`cria_parser`,
   `abre_componentes`, `cria_laco`, separadas do `main()` dele sem mudar o que ele faz). O carro
   anda depois da trava de arranque do runtime (cerca de 1 s com os sensores em dia), e o Back
   para na hora. **Esta versão ainda não rodou no Jetson.**

Regras que estão no código de propósito (`src/ai/car/controle_unico.py`):

- **Conta o instante em que o botão é apertado, não o botão estar apertado.** Como o adaptador
  não zera o estado quando o controle desliga (seção 4), um Start "preso" largaria uma corrida
  atrás da outra.
- **Cada corrida começa com estado novo:** trava de arranque, parada de fim de pista e filtro do
  esterço zerados, e a fila do LiDAR (que seguiu chegando enquanto o carro andava à mão) é
  jogada fora antes do primeiro quadro.
- **Quem escreve na placa é um só.** Durante a corrida, só o laço do modelo; à mão, só o
  controle.
- **A corrida acaba sozinha** no limite de tempo (`--seconds`, padrão 60) ou se o laço do modelo
  der erro (câmera, I2C...). Nos dois casos o carro é estacionado e o manual volta.

Gravações: **desligadas por padrão**, para não gastar tempo de quadro nem o disco pequeno do
carro. Com `--grava`, cada corrida volta a ter a sua pasta em `runs/`, com o mesmo conteúdo que
o runtime gravava.

O que não resolve:

- **No fim da pista o carro para, mas a corrida continua** até o Back ou o fim do tempo.
- **Se o controle desligar no meio da corrida, não há Back.** O carro para sozinho no fim da
  pista, na parada frontal ou no fim do tempo.
- **A espera passou para a abertura do programa:** carregar tudo continua levando o mesmo tempo,
  uma vez. Depois disso a câmera e o LiDAR ficam ligados o tempo todo, inclusive dirigindo à mão.

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

**Estado em 2026-10-17:** juntos, o modo manual e a corrida funcionaram no carro. A versão com
a corrida **dentro** do programa (seção 6, item 3) ainda não rodou no Jetson: o primeiro uso é
com as rodas no ar. Roda só no terminal, sem monitor no carro, e **sem `sudo`**.

```bash
python3 hardware/controle_8bitdo.py
python3 hardware/controle_8bitdo.py --grava           # cada corrida numa pasta em runs/
python3 hardware/controle_8bitdo.py --max-us 1650     # teto mais alto à mão
python3 hardware/controle_8bitdo.py --seconds 120     # flag que vai para o runtime
```

Ao abrir, o programa carrega o modelo, a câmera e o LiDAR (uma vez só) antes de mostrar os
comandos.

**A regra que manda: quem escreve na placa é um só.** Sem corrida, o controle manual. Com
corrida, o laço do modelo. Os dois nunca no mesmo quadro. A troca está em
`src/ai/car/controle_unico.py`.

| Comando | Sem corrida (à mão) | Com corrida |
|---|---|---|
| Analógico esquerdo | volante | nada: quem dirige é o modelo |
| RT | acelerador | nada |
| LT | ré | nada |
| Back | parada; para andar de novo, soltar os dois gatilhos | para a corrida |
| Home | parada, como o Back | nada |
| Start | estaciona o carro (centro e neutro) e começa a corrida | nada |
| Ctrl+C | estaciona e sai | para a corrida, estaciona e sai |

Quando a corrida acaba, o modo manual volta **parado até os dois gatilhos serem soltos**: um RT
que ficou apertado durante a corrida não arranca o carro.

Detalhes do modo manual (lógica em `src/ai/car/manual.py`):

- **Volante:** proporcional. Centro 1500 µs; batentes 1800 µs (esquerda) e 1200 µs (direita),
  span 300. O modelo usa 350 (1850/1150 µs), e com 300 a curva mais fechada pode não sair à mão;
  o Rafael escolheu começar em 300 para testar (2026-10-14). `--span 350` iguala ao modelo. O
  teclado usa ±260, tudo ou nada.
- **Acelerador (RT):** proporcional ao curso, de 1600 µs, o mínimo que move o ESC, até 1620 µs.
  Era até 1700 µs, o máximo calibrado; no primeiro teste o Rafael achou rápido demais e baixou
  para 1630 (2026-10-14) e depois para 1620. A corrida autônoma segue a 1610 fixos. Abaixo de
  10% do curso conta como solto.
  `--max-us` muda o teto, até 1700.
- **Ré (LT):** 1380 µs fixos. Quanto mais abaixo do neutro (1500), mais forte a ré. Era 1400,
  como a seta para baixo do teclado. Ajustada pelo Rafael no carro: 1420 em 2026-10-17 (dava
  menos ré), depois 1380, 1390, e de volta a 1380, que ficou.

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
  Na corrida, uma falha encerra a corrida, estaciona o carro e volta ao manual.
- Na corrida, as regras da seção 6.

O que ele **não** tem: a parada frontal do LiDAR no modo manual. O runtime corta o motor quando
algo fica a menos de 6 cm à frente; à mão, quem dirige pode bater na parede.

### Primeiro teste, com as rodas no ar

1. `python3 hardware/controle_8bitdo.py`. Ele põe o ESC em neutro por 2 s antes de liberar.
2. À mão: analógico para a direita vira as rodas para a direita. **Se virar ao contrário, pare
   aqui.** A troca é `SINAL_DO_ANALOGICO` em `src/ai/car/manual.py`.
3. RT devagar, LT para a ré, e Back com o RT apertado: as rodas param e só voltam depois de
   soltar e apertar o RT de novo.
4. Com o RT apertado, **Start**: aparece `CORRIDA COMECOU` e, uma vez por segundo, uma linha
   `corrida: fps=...`. As rodas giram **em cerca de 1 s** (a trava de arranque). Mexer o
   analógico e o RT durante a corrida não deve fazer nada.
5. **Back**: as rodas param na hora e aparece `corrida parada (Back)`.
6. Com o RT ainda apertado depois do Back, as rodas ficam paradas. Soltar e apertar o RT: volta
   o modo manual.
7. Start de novo: deve largar tão rápido quanto da primeira vez, sem carregar nada.
8. Start, e **Ctrl+C** no terminal durante a corrida: a corrida para, o carro é estacionado e o
   programa sai.

Testado no PC: `tests/test_car_manual.py`, `tests/test_car_controle_unico.py`,
`tests/test_car_largada.py`, `tests/test_car_escrita_tolerante.py`, `tests/test_car_run_log.py`
e `tests/test_controle_8bitdo.py`. Este último roda o programa inteiro com placa, controle e
`jetson_runtime` de mentira e confere, entre outras coisas, que o modelo, a câmera e o LiDAR são
abertos **uma vez** para várias corridas e que o controle é lido sem esperar durante a corrida.
Também foi conferido que o `main()` do `jetson_runtime.py` antes e depois da separação das
funções monta exatamente a mesma corrida. **Não testado:** a corrida dentro do programa no
carro.

## 9. Abrir sozinho quando o Jetson liga (dia da feira)

**Pedido do Rafael em 2026-10-20:** na feira só o `controle_8bitdo.py` roda, e o carro não tem
monitor nem teclado. Ligou o Jetson, o programa tem que estar aberto. Quem faz isso é um serviço
do systemd, instalado por `hardware/servico_carro.py`. **Escrito e testado no PC; ainda não rodou
no Jetson.**

O serviço (`carro-controle`):

- roda como o **seu usuário**, não root (o programa recusa root, seção 6);
- leva o `LD_LIBRARY_PATH` (as bibliotecas do CUDA) **do terminal onde você instalou**: um serviço
  não lê o `.bashrc`, e foi a falta disso que derrubou a corrida com sudo em 2026-10-14. Instale
  do mesmo terminal em que o programa funciona à mão;
- **para com SIGINT, igual ao Ctrl+C**: o programa estaciona o carro (centro e neutro) antes de
  sair. Com o SIGTERM, o padrão do systemd, ele morreria sem estacionar e o PCA9685 continuaria
  mandando o último pulso. Vale também para `sudo shutdown`;
- espera o daemon da câmera (`nvargus-daemon`) e, se o programa cair (adaptador ou câmera ainda
  não apareceram no boot, controle perdido, erro), **abre de novo 5 s depois**, sem desistir.
  Parado com `systemctl stop`, fica parado.

```bash
python3 hardware/servico_carro.py mostra            # só imprime o serviço, não instala
python3 hardware/servico_carro.py instala           # abre em todo boot (pede a senha do sudo)
python3 hardware/servico_carro.py instala --grava   # flags extras vão para o controle_8bitdo.py
python3 hardware/servico_carro.py remove            # para e desfaz

sudo systemctl start carro-controle                 # abrir agora, sem reiniciar
journalctl -u carro-controle -f                     # ver o que o programa imprime
sudo systemctl stop carro-controle                  # parar (estaciona o carro)
```

Rode o `servico_carro.py` **sem sudo**: ele pede a senha só para gravar o serviço. O `instala`
apenas habilita, e o programa só abre no próximo boot ou com o `start`.

**Com o serviço rodando, nada mais pode usar o carro.** Antes de abrir qualquer programa à mão
(`controle_8bitdo.py`, `jetson_runtime.py`, calibrações), rode `sudo systemctl stop
carro-controle`. Uma segunda cópia do programa do controle acha o adaptador ocupado e sai com uma
mensagem que diz isso. Os outros programas não avisam: o `jetson_runtime.py` falha na câmera, e
os de calibração escreveriam no PCA9685 junto com o serviço.

### Como saber que está pronto, sem monitor

Mexa o analógico esquerdo: **quando as rodas virarem, o programa está pronto**. Antes disso ele
ainda está carregando o modelo, a câmera e o LiDAR (não cronometrado; o mesmo tempo de abrir à
mão). Não aperte Start antes de as rodas responderem.

### Testes antes da feira, com as rodas no ar

1. `python3 hardware/servico_carro.py mostra`: confira que aparece a linha
   `Environment="LD_LIBRARY_PATH=..."` com o caminho do CUDA.
2. `instala`, depois `sudo systemctl start carro-controle` e `journalctl -u carro-controle -f`:
   deve aparecer `=== Controle 8BitDo ===`. O volante responde, Start larga e Back para.
3. `sudo systemctl stop carro-controle`: o journal deve terminar em
   `estado seguro. saindo: interrompido`. É a prova de que parar o serviço estaciona o carro.
4. **Controle desligado, reinicie o Jetson**, espere uns 2 minutos e só então ligue o controle.
   O volante responde? **Nunca testado:** até hoje o programa sempre abriu com o controle já
   ligado, e não sabemos se os dois comandos de abertura (seção 2) precisam ser reenviados
   quando o controle se liga depois (seção 7).
5. Com tudo funcionando, desligue o controle (segure o Home) e ligue de novo. O volante volta a
   responder? Na feira isso vai acontecer sozinho: o controle desliga depois de 15 minutos sem
   uso, segundo o manual.

Se o 4 ou o 5 falharem, o controle precisa dos comandos de abertura de novo. O contorno, até
mudar o código, é reiniciar o programa com o controle já ligado: `sudo systemctl restart
carro-controle`, ou desligar e religar o Jetson (pelo LB + RB, abaixo).

### Desligar pelo controle: LB + RB por 3 s

**Não tire a bateria do Jetson com ele ligado.** Já aconteceu uma vez (a bateria acabou com o
carro ligado): o Jetson não ligou mais e foi preciso consertar o sistema de arquivos à mão
(`docs/BRIEFING_APRESENTACAO.md`). Na feira isso aconteceria a cada troca de bateria.

**Decisão do Rafael em 2026-10-20:** desligar pelo controle. Com o carro fora de corrida,
segure **LB + RB juntos por 3 s**. O programa estaciona o carro, **vira as rodas para os dois
lados, duas vezes** (o aviso), e manda o Jetson desligar (`sudo /sbin/shutdown -h now`).

A ordem para desligar o carro:

1. Back (carro parado) e a **chave do motor desligada**.
2. LB + RB por 3 s. As rodas mexem.
3. Espere o Jetson terminar de desligar. **Quanto tempo leva ainda não foi medido** (abaixo).
4. Só então tire a bateria do Jetson.

Regras do programa (`src/ai/car/desligamento.py` e `hardware/controle_8bitdo.py`):

- **Durante a corrida, não faz nada.** O aperto tem que começar fora de corrida: segurar os dois
  na corrida e continuar depois dela não desliga até soltar e apertar de novo.
- **Rodas mexendo quer dizer "está desligando".** O programa confere antes se o sudo deixa
  desligar sem senha (`sudo -n -l`); se não deixa, as rodas não mexem e nada acontece. A falta da
  regra aparece na abertura do programa: `desligar INDISPONIVEL`.
- Depois do comando, o programa ignora o Ctrl+C (que é como o sistema para o serviço) para a
  saída segura ir até o fim, e sai sem ser reaberto.
- Se o comando falhar mesmo assim, aparece `O COMANDO DE DESLIGAR FALHOU` e o programa segue
  funcionando. Sem monitor ninguém vê essa mensagem; por isso a conferência antes de mexer as
  rodas.

A permissão: o `servico_carro.py instala` grava `/etc/sudoers.d/carro-desligar`, que libera
**só** o `/sbin/shutdown -h now`, sem senha, para o seu usuário. Uma regra quebrada pode
inutilizar o sudo, então ela é gravada primeiro com um nome com ponto (o sudo ignora esses
arquivos), conferida pelo `visudo` e só então renomeada. O `remove` apaga a regra junto com o
serviço.

**Testes, com monitor, uma vez antes da feira:**

6. Na abertura do programa **não** pode aparecer `desligar INDISPONIVEL`.
7. Fora de corrida, segure LB + RB: as rodas viram para os dois lados e o Jetson desliga. Se
   nada acontecer, LB e RB podem não estar nos bits 8 e 9 (ainda não vistos no carro,
   seção 3): rode o `hardware/pad_teste.py` apertando os dois.
8. **Cronometre** do aviso das rodas até a tela apagar. Esse tempo, com folga, é a espera do
   passo 3 na feira.
9. Durante uma corrida, segure LB + RB: não deve acontecer nada além da corrida.

Testado no PC: `tests/test_servico_carro.py` (o serviço gerado, a regra do sudo e os comandos
de instalar e remover, com sudo e systemctl de mentira), `tests/test_car_desligamento.py` (o
segurar LB + RB) e, em `tests/test_controle_8bitdo.py`, o programa inteiro desligando, sem a
regra, durante a corrida e com o comando falhando, além da mensagem do adaptador ocupado. **Não
testado no Jetson.**
