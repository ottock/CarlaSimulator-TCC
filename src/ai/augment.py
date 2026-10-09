"""Aumento fotometrico: o ponte entre a luz do simulador e a da pista real.

Medido nas corridas de 2026-09-12, na entrada do modelo (pixels 0-255):

    treino (simulador) : media 120   desvio 75
    real   (carro)     : media  79   desvio 36

Metade do contraste e bem mais escuro. A causa e fisica, nao de montagem: no
simulador o chao e PRETO e a parede BRANCA sob luz uniforme, enquanto na pista
real o chao e cinza medio, o isopor e esbranquicado e a janela joga contraluz.
O recorte da imagem NAO explica a diferenca -- medimos `crop_frac` de 0.30 a
1.00 e a distancia do perfil vertical ficou praticamente igual (0.35-0.37).

O modelo aprendeu a achar a borda parede/chao num contraste que nao existe la.
Treinar com estas variacoes o obriga a achar a GEOMETRIA em vez do brilho.

As faixas cobrem de proposito bem alem do medido: um aumento que so alcanca a
media real cobre o caso tipico, mas nao a variacao de luz ao longo do dia.

REMEDIDO em 2026-10-05, com o oval montado e o laco a 50 Hz, na entrada exata
do modelo (runs/Diag/diag hz contra dataset_oval_v2):

    treino (simulador) : brilho 151   contraste por quadro 58.2   gradiente h 0.6
    real   (carro)     : brilho 106   contraste por quadro 17.3   gradiente h 2.6

A imagem real e TRES VEZES E MEIA mais chapada, nao duas. E o aumento anterior
nao cobria isso: com contraste (0.45, 1.15) o p5 das amostras aumentadas ficava
em 24.0 e o real e 17.3 -- so 3% dos quadros de treino chegavam a ser tao
chapados quanto o que a camera entrega. O modelo praticamente nunca viu a
imagem que ia receber.

Dai as faixas novas. Com contraste (0.18, 1.10) o valor real passa a ficar
DENTRO da faixa e 13% das amostras ficam tao chapadas quanto a camera entrega,
contra 3% antes.

O que este modulo NAO resolve: a imagem real tem 4x mais gradiente de alta
frequencia (2.6 contra 0.6). Tentei somar ruido gaussiano aqui e medi que nao
adianta -- o jitter roda na imagem GRANDE e o `preprocess` reduz para 200x66,
e a media do redimensionamento apaga o granulado (gradiente aumentado ficou em
0.4-1.4, longe dos 2.6). Alem disso o excesso real provavelmente nao e ruido de
sensor e sim TEXTURA de cena: o granulado do isopor e a sujeira do chao, que
sobrevivem ao redimensionamento porque sao estrutura, nao ruido. Ruido branco
sintetico nao reproduziria isso. Fica como gap conhecido.
"""
import numpy as np

# (min, max) das tres distorcoes. Aplicadas juntas em cada amostra.
CONTRAST = (0.18, 1.10)      # 0.30 reproduz o contraste real; 0.18 da folga
BRIGHTNESS = (-60.0, 25.0)   # -45 reproduz o brilho real medido
GAMMA = (0.7, 1.6)           # nao-linearidade, para nao virar so um ajuste afim
# Granulado do sensor. A imagem real tem 4x o gradiente de alta frequencia da do
# simulador (2.6 contra 0.6 na entrada do modelo). A faixa e larga porque isto e
# somado na imagem GRANDE e depois o `preprocess` reduz de 800 para 200 de
# largura: a media de 4x4 pixels corta o desvio do ruido branco em 4.
NOISE_STD = (0.0, 14.0)

# As faixas com que cada modelo foi treinado, POR NOME. Um retreino que queira
# mudar uma coisa so (a margem do traçado, por exemplo) precisa repetir o resto
# da receita de um modelo anterior -- sem reverter codigo e sem depender de
# alguem lembrar os numeros. O nome usado vai gravado no checkpoint.
#   v2: driving_oval_v2, o primeiro a dar voltas no carro (antes do a9fbbc6)
#   v3: as faixas atuais, alargadas para alcancar o contraste real
FAIXAS = {
    "v2": {"contrast": (0.45, 1.15), "brightness": (-50.0, 25.0),
           "gamma": (0.7, 1.6), "noise_std": (0.0, 0.0)},
    "v3": {"contrast": CONTRAST, "brightness": BRIGHTNESS,
           "gamma": GAMMA, "noise_std": NOISE_STD},
}


def photometric_jitter(img_bgr, rng, contrast=CONTRAST, brightness=BRIGHTNESS,
                       gamma=GAMMA, noise_std=NOISE_STD):
    """Randomly re-light a BGR uint8 image, keeping its geometry untouched.

    Only brightness/contrast/gamma change -- never the shape or the layout. The
    label (steering) depends on WHERE things are, so any geometric distortion
    here would silently teach the wrong target.

    ``rng`` is passed in rather than drawn from a global, so training runs stay
    reproducible and the tests are deterministic.
    """
    a = float(rng.uniform(*contrast))
    b = float(rng.uniform(*brightness))
    g = float(rng.uniform(*gamma))
    sigma = float(rng.uniform(*noise_std)) if noise_std else 0.0

    x = img_bgr.astype(np.float32)
    x = (x - 128.0) * a + 128.0 + b
    if sigma > 0.0:
        # DEPOIS do contraste, nao antes: o granulado e somado pela camera,
        # entao ele nao encolhe junto quando a cena fica chapada. Aplicado
        # antes, um contraste de 0.2 o reduziria a um quinto e a imagem
        # chapada sairia limpa demais -- que e o que o carro NAO entrega.
        x += rng.normal(0.0, sigma, x.shape)
    np.clip(x, 0.0, 255.0, out=x)
    if g != 1.0:
        x = 255.0 * np.power(x / 255.0, g)
        np.clip(x, 0.0, 255.0, out=x)
    return x.astype(np.uint8)
