"""LiDAR scan -> car frame: auto-oclusao e alinhamento. Puro, Python 3.6-safe.

Duas correcoes que o primeiro log no carro tornou obrigatorias.

**Auto-oclusao.** A carroceria bloqueia um arco fixo do sensor. Medido em
`runs/` (2026-09-12, carro parado em area aberta): ~0 a 120 graus lendo
0.20-0.32 m com spread <= 0.04 -- perto E constante, a assinatura de algo preso
ao proprio carro -- mais um ponto encostado em ~212 graus. Deixar isso passar
entrega a rede uma parede permanente a 25 cm em um terco do circulo, que e' uma
entrada que ela nunca viu no treino. Descartamos esses retornos, e os setores
correspondentes passam a ler ``max_range`` ("livre"), que e' exatamente como o
treino tratava os setores mascarados.

**Alinhamento.** O zero do sensor nao aponta necessariamente para a frente do
carro, e o sentido de rotacao pode ser o oposto do simulador. Como a mascara de
FOV e o cone da parada de emergencia sao definidos em torno da FRENTE, um sensor
girado faz a rede olhar para o lado errado -- sem erro nenhum no terminal.
"""


def _in_arc(angle, start, end):
    """True if ``angle`` lies in [start, end], handling an arc crossing zero."""
    a = angle % 360.0
    s = start % 360.0
    e = end % 360.0
    if s <= e:
        return s <= a <= e
    return a >= s or a <= e      # o arco passa pelo zero


def drop_self_occlusion(angles_deg, dist_m, arcs):
    """Drop returns that come from the car's own body.

    Args:
        angles_deg / dist_m: the raw scan, aligned.
        arcs: iterable of ``(start_deg, end_deg)`` in the SENSOR frame -- the
            body does not move relative to the sensor, so these are measured
            once and stay fixed until the LiDAR is remounted.

    Returns:
        ``(angles, dists)`` without the occluded returns.
    """
    if not arcs:
        return angles_deg, dist_m
    out_a, out_d = [], []
    for a, d in zip(angles_deg, dist_m):
        if any(_in_arc(a, s, e) for s, e in arcs):
            continue
        out_a.append(a)
        out_d.append(d)
    return out_a, out_d


def rotate_angles(angles_deg, offset_deg, invert=False):
    """Turn sensor angles into car angles, where 0 deg is the car's front.

    ``offset_deg`` is the sensor angle that points at the car's FRONT: after the
    rotation that angle reads 0. ``invert`` mirrors the direction of rotation,
    for a sensor that spins opposite to the simulator's convention.

    Order is fixed on purpose: mirror first (in the sensor frame), then rotate
    into the car frame. The opposite order gives a different -- wrong -- result.
    """
    out = []
    for a in angles_deg:
        x = (-a) if invert else a
        out.append((x - offset_deg) % 360.0)
    return out


def parse_arcs(text):
    """Parse ``"0:120,210:215"`` into ``[(0.0, 120.0), (210.0, 215.0)]``.

    Raises ``ValueError`` on anything malformed instead of quietly returning an
    empty list: silently masking nothing would put the car's own body back into
    the model's input with no sign that it happened.
    """
    if not text:
        return []
    arcs = []
    for piece in str(text).split(","):
        piece = piece.strip()
        if not piece:
            continue
        if piece.count(":") != 1:
            raise ValueError(
                "arco invalido %r: use inicio:fim, por exemplo 0:120" % piece)
        a, b = piece.split(":")
        try:
            arcs.append((float(a), float(b)))
        except ValueError:
            raise ValueError(
                "arco invalido %r: inicio e fim tem de ser numeros em graus" % piece)
    return arcs
