# Briefing: preparar a apresentação do TCC na feira

Você vai me ajudar a planejar **tudo o que precisa estar pronto no dia da apresentação** do meu
TCC. Este documento tem o contexto completo. Você não conhece o projeto, então leia até o fim
antes de responder.

**Importante:** o carro já funciona. Não quero ajuda para treinar o modelo nem para mexer no
código de direção. Quero ajuda com o **evento**: roteiro, materiais, logística, ensaio, riscos e
plano de contingência.

Cada informação abaixo está marcada como:
- **[FATO]** — medido ou testado por nós;
- **[PLANO]** — o que pretendemos fazer, ainda pode mudar;
- **[A CONFIRMAR]** — eu ainda não sei ou não conferi. Não trate como verdade.

---

## 1. O que eu preciso de você

Primeiro, **me faça as perguntas da seção 9** (e outras que julgar necessárias). Não invente
datas, números nem medidas que não estão aqui.

Depois, entregue:

1. **Recomendação sobre a decisão em aberto** (seção 6): o carro anda sozinho, ou o visitante
   tenta dirigir antes? Com prós, contras e, se fizer sentido, uma opção intermediária.
2. **Roteiro da apresentação em três tamanhos**: 30 segundos (quem só passa), 3 minutos (quem
   para) e 10 minutos (quem quer entender a fundo ou é da banca). Com falas sugeridas.
3. **Lista de materiais** a produzir ou comprar (adereços, cartazes, sinalização, peças
   reserva), com prioridade: indispensável / desejável / se sobrar tempo.
4. **Cronograma de preparação**, de trás para frente a partir da data do evento, incluindo
   ensaios.
5. **Plano do dia**: montagem, escala de quem fica no estande ao longo das 12 horas, pausas,
   troca e recarga de baterias, desmontagem.
6. **Planta do estande** de 4 × 2 m, considerando o tamanho da pista (seção 5.2).
7. **Plano de contingência**: o que fazer e o que dizer ao público se cada coisa da seção 7
   falhar.
8. **Texto dos cartazes** e das perguntas dos jogos com o público.

---

## 2. O projeto em um minuto

**[FATO]** É um carrinho de controle remoto em escala 1:12 (WLtoys 124017) que **dirige
sozinho** numa pista de paredes brancas e piso preto. Ele usa uma **câmera** e um **LiDAR**
(sensor a laser que mede distâncias ao redor). Uma rede neural, rodando num computador embarcado
(Jetson Nano), olha a imagem e o LiDAR e decide para onde virar o volante. A velocidade é
constante.

**[FATO]** A rede aprendeu a dirigir **inteiramente num simulador** (CARLA), numa cópia digital
da pista — um "gêmeo digital". Um piloto programado dirige no simulador, a rede aprende imitando
(*Behavioral Cloning*), e depois o modelo é copiado para o carro real. O carro real nunca foi
"ensinado" na pista física.

**[FATO]** O carro não precisa conhecer a pista de antemão: o objetivo do TCC é que **alguém
monte uma pista na hora e o carro a percorra**.

Equipe: **[A CONFIRMAR]** quantas pessoas e quem.

---

## 3. O evento

- **[FATO]** Feira acadêmica.
- **[FATO]** Estande de **4 × 2 m**.
- **[FATO]** Das **8h às 20h** (12 horas seguidas).
- **[FATO]** O público passa o tempo todo, assiste e quer entender o projeto. Não é uma
  apresentação única para uma plateia sentada: é um fluxo contínuo de pessoas, que chegam no
  meio e saem quando querem.
- **[A CONFIRMAR]** data, local, se há tomada e internet no estande, se podemos levar TV ou
  monitor, e se há avaliação por banca durante a feira.

---

## 4. Roteiro que temos em mente

### 4.1 Abertura: o problema **[PLANO]**

Queremos começar pelo problema, com adereços físicos:

- **Jogo do ranking:** a pessoa ordena as 5 maiores causas de morte entre jovens. Depois
  revelamos que a maior são os acidentes de trânsito.
- **Engarrafamento fantasma, com carrinhos Hot Wheels:** mostramos como ele se forma (alguém
  freia sem motivo, a onda se propaga para trás). Perguntamos: "quantos destes carros precisam
  ser autônomos para isso não acontecer mais?". Revelamos: de 5 a 10%.

**Afirmações que queremos usar — todas [A CONFIRMAR], preciso da fonte conferida antes de pôr
num cartaz:**

| Afirmação | De onde achamos que vem |
|---|---|
| Acidentes de trânsito são a maior causa de morte entre jovens | **Sem fonte ainda.** Parece ser um dado da OMS para a faixa de 5 a 29 anos **no mundo**. Conferir a faixa etária e o recorte: no Brasil a principal causa entre jovens pode ser outra, e aí o jogo do ranking precisa dizer "no mundo". |
| Com 5% a 10% dos carros autônomos dá para evitar engarrafamentos fantasma | Estudos 1 e 2 abaixo. O estudo 1 é um experimento controlado numa pista circular, não uma previsão para o trânsito de uma cidade: a frase precisa respeitar isso. |
| Engarrafamentos fantasma são 20% do trânsito | **Sem fonte.** Nenhum dos estudos abaixo diz isso, até onde sei. |
| O fim deles reduz o consumo de combustível em 40% | Estudo 1. Conferir o que exatamente foi medido (consumo de quais carros, em que condição). |
| A velocidade média do trânsito subiria de 6% a 12% | Estudo 2, segundo o resumo que temos. Conferir. |

**Estudos que encontramos.** Atenção: a lista abaixo veio de um **resumo que eu não conferi nos
artigos originais**. Trate títulos, autores, periódicos e números como [A CONFIRMAR].

1. **Stern e outros (2018), *Dissipation of stop-and-go waves via control of autonomous
   vehicles: Field experiments*, Transportation Research Part C.** Experimento de campo: um
   carro autônomo no meio de cerca de 20 carros com motoristas humanos, numa pista circular,
   dissolve as ondas de anda-e-para sem comunicação entre veículos. O resumo diz: até 40% menos
   combustível e até 15% mais vazão.
2. ***Smoothing Traffic Flow via Control of Autonomous Vehicles*.** O resumo atribui a "J. Wang,
   S. Zheng, C. Wu e colaboradores (MIT e UC Berkeley)", em IEEE Transactions on Control Systems
   Technology / arXiv, 2018–2020, e diz que 5% a 10% de autônomos bastam para estabilizar o
   tráfego em rodovias de várias faixas, com 6% a 12% a mais de velocidade média.
   **Desconfio da atribuição:** pode haver mistura de dois trabalhos diferentes (um artigo com
   esse título e outro grupo que estuda tráfego misto com aprendizado por reforço). Conferir
   autores, periódico e se os números estão mesmo nesse artigo.
3. **Sugiyama e outros (2008), *Traffic jams without bottlenecks — experimental evidence for
   the physical mechanism of the formation of a jam*, New Journal of Physics.** O experimento
   que mostrou o engarrafamento fantasma: 22 carros com motoristas humanos numa pista circular
   formam ondas de parada sozinhos, sem gargalo. É a base do nosso adereço com Hot Wheels.
4. **"Pesquisas do Instituto Fraunhofer / Universidade Temple (2023–2026)"**, com o simulador
   RODOS e Benjamin Seibold. O resumo diz que nem todo piloto automático ajuda: um controle de
   cruzeiro adaptativo "agressivo", que anda colado no carro da frente, pode **piorar** o
   engarrafamento; só ajuda o que suaviza a velocidade. **Não consegui confirmar que esse
   estudo existe como descrito.** A ideia em si aparece em trabalhos do mesmo grupo sobre
   controles de cruzeiro comerciais; achar a referência certa.

O ponto do item 4 é importante para a nossa fala ser honesta: "carro autônomo" não resolve o
trânsito por si só — depende de como ele é programado.

Me ajude a **conferir cada afirmação na fonte original, corrigir o que estiver errado no resumo
e propor uma frase honesta** para cada uma. Se não conseguir conferir, diga isso em vez de
confirmar de memória. Se alguma não se sustentar, diga e proponha outra. As três últimas da
tabela talvez nem caibam no tempo.

### 4.2 Teoria rápida **[PLANO]**

Explicar em poucas frases o que é um carro autônomo e como funciona o gêmeo digital, mostrando
um **vídeo do simulador (CARLA) rodando**.

- **[FATO]** O simulador exige um computador com placa de vídeo dedicada e é instável: o
  servidor cai entre uma execução e outra. **Recomendação nossa: vídeo gravado antes, não
  simulação ao vivo.**

### 4.3 Demonstração do carro **[PLANO]** — a parte mais importante

O visitante **monta um traçado de pista** e o carro percorre. Regras e detalhes na seção 5.

### 4.4 Encerramento

**[A CONFIRMAR]** Ainda não temos. Sugira.

---

## 5. A demonstração: o que é verdade hoje

### 5.1 Regras da pista

- **[FATO]** Dois tipos de peça: **reta** (50 cm de caminho) e **curva** (um quarto de círculo).
  A faixa onde o carro anda tem **56 cm** de largura, entre paredes brancas, com piso preto.
- **[FATO]** As peças são colocadas numa grade de **4 × 3 peças**.
- **[FATO]** Regra: **depois de toda curva vem uma reta** (nunca duas curvas seguidas).
- **[FATO]** A pista pode ser um circuito fechado ou ter começo e fim. Se tem fim, ele é
  **fechado com uma parede branca**, da largura toda, e o carro **para sozinho** ao chegar: o
  LiDAR percebe que não há mais saída à frente.
- **[FATO]** O carro larga parado, inteiro dentro da primeira reta, centralizado.
- **[FATO]** Com essas regras existem **2 circuitos fechados** (um oval e um quadrado) e
  **33 montagens abertas diferentes** (104 se contar o sentido de percurso e os espelhos). É um
  número bom para contar ao público.
- **[A CONFIRMAR]** Pistas que **terminam numa curva** são permitidas pela regra, mas nunca
  foram treinadas nem testadas. Por segurança, a regra para o visitante deveria ser "comece e
  termine com uma reta".

### 5.2 Espaço

- **[FATO]** A grade de 4 × 3 peças ocupa aproximadamente **2,5 × 2 m**.
- **Atenção:** o estande tem 4 × 2 m. A pista ocupa **toda a profundidade** do estande e mais da
  metade da largura. Sobra uma faixa de cerca de 1,5 × 2 m para mesa, adereços, tela, pessoas da
  equipe e visitantes. Isso precisa ser resolvido na planta (seção 1, item 6).

### 5.3 O que já foi testado

- **[FATO]** No carro real, em 12/10/2026, o modelo atual percorreu uma pista de 9 peças e
  4 curvas — a montagem mais apertada de todas — **sem nenhuma batida nem raspagem**, e parou
  corretamente no fim.
- **[FATO]** Essa pista fazia parte das usadas no treino. **No carro real, ainda não testamos
  uma montagem que o modelo nunca viu.** No simulador ele completou 21 de 21 pistas, 18 delas
  inéditas — mas o simulador já errou previsões antes, então isso não é garantia.
- **[FATO]** Limitação conhecida: às vezes o carro passa bem perto da parede **interna** da
  curva. Não bate, mas é visível.
- **[FATO]** Existe um modelo anterior de reserva, que só foi treinado num circuito oval. No
  teste de 12/10 ele raspou e bateu; não contamos com ele.

### 5.4 Como uma corrida acontece hoje

- **[FATO]** Alguém da equipe posiciona o carro na largada e **digita um comando num terminal**
  ligado ao computador do carro. Não existe botão de "iniciar".
- **[FATO]** O carro leva alguns segundos para começar a andar (carrega o modelo e espera os
  sensores estabilizarem). **[A CONFIRMAR]** quantos segundos; nunca cronometramos.
- **[FATO]** Uma pista aberta de tamanho médio leva da ordem de **10 a 20 segundos**; uma volta
  no oval, cerca de 30 segundos. **[A CONFIRMAR]** medir na pista do dia.
- **[FATO]** Cada execução grava um registro (imagens e dados) no disco do carro, que é pequeno.
  Em 12 horas de corridas ele enche se ninguém cuidar.
- **[FATO]** O carrinho tem a própria chave liga/desliga, que corta o motor, e uma parada de
  emergência automática quando há algo a 6 cm da frente.
- **[A CONFIRMAR]** quanto tempo um visitante leva para montar uma pista com as peças.

---

## 6. Decisão em aberto — preciso da sua recomendação

Na demonstração, temos duas opções:

- **Opção A — o carro anda sozinho.** O visitante monta a pista, o carro percorre. Mostra a
  autonomia de forma direta.
- **Opção B — o visitante tenta primeiro.** Ele monta a pista, tenta dirigir o carrinho por ela,
  percebe que é difícil, e só então o carro faz sozinho. O contraste valoriza o resultado.

O que o lado técnico sabe e que pesa na decisão:

- **[FATO]** Dirigir esse carro nessa pista é mesmo difícil: o carro tem 21 cm de largura numa
  faixa de 56 cm e precisa de um raio de 47 cm para virar. Em algumas pistas ele usa mais de
  80% do esterço.
- **[FATO]** Existe um programa de controle manual por **teclado**, usado nos testes de
  bancada. Ele roda no computador do carro e abre uma janela, então precisa de tela e teclado
  ligados ao carro, ou de acesso remoto. **[A CONFIRMAR]** se serve para um visitante usar: nunca
  foi pensado para isso, e não sei se o controle remoto original do carrinho ainda funciona.
- **[FATO]** As paredes da pista são de papel sobre suportes de madeira. Visitante dirigindo vai
  bater, e bater amassa parede, desalinha peça e gasta bateria.
- **[FATO]** A opção B pelo menos dobra o tempo por visitante, num evento de fluxo contínuo.
- Opinião do lado técnico, para você avaliar: **A como padrão** e **B como desafio opcional**
  quando o estande estiver calmo — desde que o controle manual seja testado antes e a velocidade
  limitada. Outra saída é o visitante "dirigir" no simulador ou num vídeo, sem risco para a
  pista. Discorde se achar que há algo melhor.

---

## 7. Riscos que já conhecemos

Use estes como ponto de partida do plano de contingência, e acrescente os que faltarem.

1. **Bateria.** **[FATO]** Uma vez a bateria acabou com o carro ligado, o computador do carro
   não ligou mais e foi preciso consertar o sistema de arquivos à mão. Em 12 horas isso não pode
   acontecer. **[A CONFIRMAR]** quanto dura cada bateria (do carro e do computador), quantas
   temos e quanto tempo levam para recarregar.
2. **Sem cópia de segurança do sistema do carro.** **[A CONFIRMAR]** se existe uma imagem do
   disco pronta para restaurar.
3. **Iluminação.** **[FATO]** O modelo depende da câmera e nunca foi testado na luz do local da
   feira. Foi treinado com variações de brilho e contraste, mas isso não é garantia.
4. **Pista inédita.** Ver 5.3: no carro real só rodamos uma montagem que estava no treino.
   Visitantes vão montar qualquer coisa.
5. **Gente perto da pista.** **[FATO]** O LiDAR enxerga a até 1 m, na metade da frente do carro.
   Mão, pé ou objeto dentro da faixa pode fazer o carro parar ou errar.
6. **Peças e paredes.** **[FATO]** A parede do fim precisa ser branca, da altura das outras e
   sem fresta; com fresta o carro não reconhece o fim. Paredes amassam com o uso.
7. **Hardware congelado.** **[FATO]** Não podemos mais mudar nada da parte física do carro. Se
   algo quebrar, não há tempo de refazer. **[A CONFIRMAR]** peças reserva.
8. **Operação por terminal.** Ver 5.4: cada corrida depende de alguém digitar um comando. Com
   público esperando, isso precisa virar um procedimento simples e ensaiado.
9. **Rede.** **[A CONFIRMAR]** como o notebook fala com o carro no estande. Wi-Fi de feira
   costuma ser ruim.
10. **Disco do carro cheio**, ver 5.4.
11. **Cansaço.** 12 horas, com fala repetida dezenas de vezes.

---

## 8. Restrições

- O hardware do carro não muda.
- Mudanças no **software** do carro ainda são possíveis (por exemplo, um jeito mais simples de
  iniciar uma corrida), mas cada uma precisa ser testada na pista antes do evento. Se o seu
  plano depender de algo assim, liste como **pedido para o lado técnico**, separado do resto.
- Orçamento: **[A CONFIRMAR]**.

---

## 9. Perguntas para você me fazer antes de planejar

Estas eu já sei que faltam. Acrescente as suas.

1. Qual é a data do evento e quantos dias faltam?
2. Quantas pessoas na equipe e quem pode ficar no estande em cada horário?
3. O estande tem tomada? Mesa? Paredes onde colar cartaz? Podemos levar TV ou monitor?
4. Há avaliação por banca durante a feira, com hora marcada? O que ela avalia?
5. Quem é o público: alunos, professores, empresas, famílias, crianças?
6. Quantas baterias temos e quanto dura cada uma?
7. Quanto tempo leva para montar uma pista? E para trocar uma parede amassada?
8. Qual o orçamento para adereços e impressão?
9. O carro pode andar o dia inteiro, ou é melhor fazer sessões com hora marcada?
10. O controle manual (opção B) funciona hoje de um jeito que um visitante consiga usar?

---

## 10. Como responder

- Em português.
- Comece pelas perguntas. Depois que eu responder, entregue os itens da seção 1.
- Separe sempre o que é **indispensável** do que é **desejável**.
- Onde faltar informação, diga qual suposição você fez.
- Para as estatísticas da seção 4.1, cite a fonte e diga o quanto confia nela.
