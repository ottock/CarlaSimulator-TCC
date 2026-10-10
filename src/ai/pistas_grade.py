"""Catalogo das pistas montaveis no espaco do TCC (2026-10-09).

O espaco tem ~2,5 x 2 m: cabem 4 pecas no comprimento e 3 na largura. Ha dois
tipos de peca -- reta (50 cm de caminho, 56 cm entre paredes) e curva (1/4 de
disco de 56 cm) -- e uma regra do Rafael: **toda curva tem uma reta depois, a nao
ser que seja a ultima peca**. Na pratica, nunca duas curvas seguidas.

Cada casa da grade recebe uma peca. O percurso entra numa casa e sai por outro
lado; se sai pelo lado oposto a peca e reta, senao e curva. Uma pista e entao uma
SEQUENCIA de pecas, escrita como codigo: ``S`` reta, ``E`` curva a esquerda,
``D`` curva a direita. Duas pistas em lugares diferentes da grade, ou giradas,
com o mesmo codigo sao a mesma pista para o carro.

Numeros (conferidos nos testes): 3 lacos e 227 percursos abertos na grade, que
viram **2 formatos fechados** (o oval de 10 pecas e o quadrado de 8) e **104
formatos abertos** contando o sentido de percurso. As pontas de um percurso
aberto sao retas; ``variantes_curva_final`` acrescenta os que terminam numa curva,
que a regra tambem permite.

Matematica pura: nao importa ``carla``.
"""
import random

LINHAS, COLUNAS = 4, 3
_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))

# Pecas do gemeo digital. Nao ha prop de 50 cm: uma reta real vira branca (31,5)
# + cinza (21,5) = 53 cm, a mesma faixa de 53 das curvas -- a casa da grade do
# simulador e quadrada. A troca de textura no meio da reta e de proposito: da
# sinal a camera onde senao e tudo igual (como no oval_tcc).
_RETA = ("tcc_reta_branca", "tcc_reta_cinza")
_ESQ, _DIR = "tcc_curva90", "tcc_curva90_r"


def _vizinhos(casa):
    for dl, dc in _DIRS:
        n = (casa[0] + dl, casa[1] + dc)
        if 0 <= n[0] < LINHAS and 0 <= n[1] < COLUNAS:
            yield n


def _direcao(a, b):
    return (b[0] - a[0], b[1] - a[1])


def _giro(d1, d2):
    """``S`` se segue reto, ``E``/``D`` se vira. Linha cresce para baixo."""
    if d1 == d2:
        return "S"
    # Em (x = coluna, y = -linha) a regra da mao direita vale: cruz > 0 e esquerda.
    x1, y1, x2, y2 = d1[1], -d1[0], d2[1], -d2[0]
    return "E" if x1 * y2 - y1 * x2 > 0 else "D"


def tipos(casas, fechada):
    """Sequencia de casas -> codigo das pecas. Pontas de percurso aberto sao retas."""
    n = len(casas)
    out = []
    for i in range(n):
        if not fechada and (i == 0 or i == n - 1):
            out.append("S")
            continue
        out.append(_giro(_direcao(casas[i - 1], casas[i]),
                         _direcao(casas[i], casas[(i + 1) % n])))
    return "".join(out)


def respeita_a_regra(codigo, fechada):
    """Nenhuma curva seguida de outra curva (no laco, contando a emenda)."""
    n = len(codigo)
    pares = range(n) if fechada else range(n - 1)
    return not any(codigo[i] != "S" and codigo[(i + 1) % n] != "S" for i in pares)


def enumera():
    """``(lacos, abertos)``: listas de sequencias de casas, cada percurso uma vez.

    Aberto e contado uma vez so (ida = volta); laco, uma vez por conjunto de
    emendas. Percursos abertos tem pelo menos 3 pecas.
    """
    abertos, lacos = {}, {}

    def busca(seq, vistas):
        atual = seq[-1]
        if len(seq) >= 4 and seq[0] in _vizinhos(atual):
            if respeita_a_regra(tipos(seq, True), True):
                chave = frozenset(frozenset((seq[i], seq[(i + 1) % len(seq)]))
                                  for i in range(len(seq)))
                lacos.setdefault(chave, list(seq))
        if len(seq) >= 3 and respeita_a_regra(tipos(seq, False), False):
            chave = frozenset(frozenset((seq[i], seq[i + 1])) for i in range(len(seq) - 1))
            abertos.setdefault(chave, list(seq))
        for n in _vizinhos(atual):
            if n in vistas:
                continue
            # Poda: duas curvas seguidas no meio nunca mais ficam validas.
            if len(seq) >= 3:
                t = tipos(seq + [n], False)
                if t[-2] != "S" and t[-3] != "S":
                    continue
            vistas.add(n)
            seq.append(n)
            busca(seq, vistas)
            seq.pop()
            vistas.discard(n)

    for lin in range(LINHAS):
        for col in range(COLUNAS):
            busca([(lin, col)], {(lin, col)})
    return list(lacos.values()), list(abertos.values())


def espelho(codigo):
    """A mesma pista vista no espelho: esquerda e direita trocadas."""
    return codigo.translate(str.maketrans("ED", "DE"))


def reverso(codigo):
    """A mesma pista aberta percorrida ao contrario."""
    return espelho(codigo[::-1])


def variantes_curva_final(codigo):
    """Variantes que terminam numa curva contra a parede: ``...SS`` -> ``...SE``/``...SD``.

    A regra deixa a ultima peca ser curva. So vale se a penultima for reta --
    senao seriam duas curvas seguidas.
    """
    if len(codigo) < 3 or codigo[-1] != "S" or codigo[-2] != "S":
        return []
    return [codigo[:-1] + "E", codigo[:-1] + "D"]


def formatos(curva_final=False):
    """``(fechados, abertos)``: codigos unicos, ordenados (deterministico).

    Abertos entram nos dois sentidos de percurso -- o carro largando de cada
    ponta ve uma pista diferente. ``curva_final`` acrescenta as variantes que
    terminam em curva.
    """
    lacos, abertos = enumera()
    fechados = sorted({tipos(c, True) for c in lacos})
    cods = set()
    for c in abertos:
        cods.add(tipos(c, False))
        cods.add(tipos(c[::-1], False))
    if curva_final:
        for c in list(cods):
            cods.update(variantes_curva_final(c))
    return fechados, sorted(cods, key=lambda c: (len(c), c))


def pecas(codigo):
    """Codigo -> nomes das pecas do gemeo, na ordem de montagem."""
    out = []
    for x in codigo:
        if x == "S":
            out.extend(_RETA)
        elif x == "E":
            out.append(_ESQ)
        elif x == "D":
            out.append(_DIR)
        else:
            raise ValueError("peca desconhecida %r no codigo %r (use S, E, D)" % (x, codigo))
    return out


def _classe(codigo):
    """A montagem fisica: a pista, o espelho, e os dois percorridos ao contrario."""
    r = reverso(codigo)
    return min(codigo, espelho(codigo), r, espelho(r))


def separa(codigos, frac_teste=0.15, semente=0):
    """``(treino, teste)``, separando juntas as variantes da mesma montagem.

    O treino espelha as amostras, e a mesma montagem percorrida a partir da outra
    ponta e o que alguem faria no dia: deixar o espelho ou o reverso de uma pista
    de teste no treino seria mostrar a pista de teste ao modelo.
    """
    classes = sorted({_classe(c) for c in codigos})
    rng = random.Random(semente)
    rng.shuffle(classes)
    n_teste = int(round(frac_teste * len(classes)))
    teste_cls = set(classes[:n_teste])
    teste = [c for c in codigos if _classe(c) in teste_cls]
    treino = [c for c in codigos if _classe(c) not in teste_cls]
    return treino, teste


GRUPOS = ("treino", "teste", "fechadas", "todas")


def expande_pistas(texto, frac_teste=0.15, semente=0, curva_final=False):
    """Lista de presets da linha de comando -> ``(pistas, teste)``.

    Aceita presets comuns (``oval_tcc``, ``pista1``), codigos (``grade:SDSES``) e
    grupos: ``grade:treino`` (os dois lacos + as abertas de treino),
    ``grade:teste``, ``grade:fechadas`` e ``grade:todas``. ``teste`` e sempre a
    lista das abertas separadas, para ir ao meta do dataset -- quem avalia
    precisa saber quais pistas o modelo nunca viu.
    """
    fechados, abertos = formatos(curva_final=curva_final)
    treino_ab, teste_ab = separa(abertos, frac_teste=frac_teste, semente=semente)
    grupos = {"treino": fechados + treino_ab, "teste": teste_ab,
              "fechadas": fechados, "todas": fechados + abertos}
    out = []
    for item in (x.strip() for x in str(texto).split(",")):
        if not item:
            continue
        nome = item[len("grade:"):] if item.startswith("grade:") else None
        if nome in grupos:
            out.extend("grade:" + c for c in grupos[nome])
        else:
            out.append(item)
    vistos = set()
    unicos = [x for x in out if not (x in vistos or vistos.add(x))]
    return unicos, ["grade:" + c for c in teste_ab]
