"""Decodificacao de teclas para operar a sonda estatica as cegas (Fase 6c).

A sonda e usada com o operador agachado na pista, posicionando o carro, SEM ver
o terminal. Entao ela e dirigida por tecla e sem Enter.

A parte que erra facil: as setas nao chegam como um caractere, e sim como uma
sequencia de escape de tres bytes (``\x1b[C`` para a direita). Tratar o
primeiro byte isoladamente faria a seta virar "ESC" e abortar a medicao no
meio -- justamente quando ninguem esta olhando para perceber.

Puro: so traduz texto, testavel sem Jetson.
"""

_SEQUENCIAS = {
    "\x1b[C": "direita",
    "\x1b[D": "esquerda",
    "\x1b[A": "cima",
    "\x1b[B": "baixo",
}

# Alternativas em letra, porque nem todo terminal por SSH manda a sequencia
# esperada e ficar sem saida no meio da pista seria pior.
_LETRAS = {
    "d": "direita", "a": "esquerda", "w": "cima", "s": "baixo",
    "q": "sair",
    "\r": "direita", "\n": "direita", " ": "direita",
    "\x1b": "sair",      # ESC sozinho: desistiu
    "\x03": "sair",      # Ctrl+C
}


def decodifica(texto):
    """Traduz o que veio do terminal para uma acao, ou ``None`` se nao reconhecer.

    Devolver ``None`` em vez de chutar e deliberado: uma tecla adivinhada
    dispararia uma medicao que ninguem pediu, e o operador nao veria acontecer.
    """
    if not texto:
        return None
    if texto in _SEQUENCIAS:
        return _SEQUENCIAS[texto]
    return _LETRAS.get(texto.lower())
