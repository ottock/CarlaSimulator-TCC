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
"""
import numpy as np

# (min, max) das tres distorcoes. Aplicadas juntas em cada amostra.
CONTRAST = (0.45, 1.15)      # 0.48 reproduz o contraste real medido
BRIGHTNESS = (-50.0, 25.0)   # -45 reproduz o brilho real medido
GAMMA = (0.7, 1.6)           # nao-linearidade, para nao virar so um ajuste afim


def photometric_jitter(img_bgr, rng, contrast=CONTRAST, brightness=BRIGHTNESS,
                       gamma=GAMMA):
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

    x = img_bgr.astype(np.float32)
    x = (x - 128.0) * a + 128.0 + b
    np.clip(x, 0.0, 255.0, out=x)
    if g != 1.0:
        x = 255.0 * np.power(x / 255.0, g)
        np.clip(x, 0.0, 255.0, out=x)
    return x.astype(np.uint8)
