"""Decodificacao de teclas para operar a sonda estatica as cegas (Fase 6c).

A sonda e usada com o operador agachado na pista, posicionando o carro, SEM ver
o terminal. Entao ela e dirigida por tecla e sem Enter.

A parte que erra facil: as setas nao chegam como um caractere, e sim como uma
sequencia de escape de tres bytes (``\x1b[C`` para a direita). Tratar o
primeiro byte isoladamente faria a seta virar "ESC" e abortar a medicao no
meio -- justamente quando ninguem esta olhando para perceber.

Foi exatamente o que aconteceu na pista: QUALQUER seta encerrava a sonda. A
causa nao era a tabela abaixo, e sim COMO os bytes eram lidos. ``sys.stdin`` do
Python 3 e um ``TextIOWrapper`` com buffer: a seta manda 3 bytes de uma vez, o
``read(1)`` puxa os 3 do descritor para o buffer interno e devolve so o
``\x1b``. Depois ``select()`` pergunta ao SISTEMA se sobrou algo no descritor --
e o descritor esta vazio, porque o resto esta dentro do Python. Entao a leitura
terminava em ``\x1b`` sozinho, que esta mapeado para "sair".

Dai ``LeitorDeTeclas`` usar ``os.read``, que ignora o buffer do Python e
entrega a sequencia inteira numa leitura.

Sem Jetson: ``decodifica`` so traduz texto e ``LeitorDeTeclas`` aceita um
descritor injetado, entao os dois sao testaveis com um pipe comum.
"""
import os
import sys

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


class LeitorDeTeclas(object):
    """Le UMA tecla do terminal, sem Enter, e devolve o terminal ao normal.

    Use como contexto::

        with LeitorDeTeclas() as t:
            acao = decodifica(t.ler())

    ``fd`` existe para os testes: com um descritor injetado o terminal nao e
    tocado, e da para alimentar bytes por um pipe.
    """

    # Quanto esperar o resto de uma sequencia de escape partida. Um terminal
    # lento (SSH ruim) pode entregar o \x1b antes do "[C". Tambem e o atraso
    # que o ESC solitario paga para ser reconhecido como ESC.
    ESPERA_ESCAPE_S = 0.05

    def __init__(self, fd=None):
        self._fd = fd
        self._injetado = fd is not None
        self._antes = None
        self._termios = None

    def __enter__(self):
        if self._injetado:
            return self
        try:
            import termios
            import tty
            self._termios = termios
            self._fd = sys.stdin.fileno()
            self._antes = termios.tcgetattr(self._fd)
            tty.setraw(self._fd)
        except Exception:
            # Sem terminal de verdade (pipe, IDE, Windows): cai para Enter.
            self._fd = None
        return self

    def __exit__(self, *a):
        if self._antes is not None:
            self._termios.tcsetattr(self._fd, self._termios.TCSADRAIN, self._antes)
        return False

    def _tem_mais(self):
        """Sobrou byte no descritor? Agora a resposta e confiavel, porque nada
        fica preso num buffer do Python."""
        try:
            import select
            return bool(select.select([self._fd], [], [], self.ESPERA_ESCAPE_S)[0])
        except Exception:
            return False

    def limpa(self):
        """Joga fora tecla que ficou na fila enquanto o programa estava ocupado.

        Operando as cegas, o reflexo de quem nao ve a tela e apertar de novo
        achando que nao pegou. Sem isto a tecla repetida fica enfileirada no
        terminal e dispara a medicao SEGUINTE no instante em que a atual
        termina -- com o carro ainda na posicao antiga e o operador ainda
        agachado. Vira uma medicao errada que ninguem viu acontecer.
        """
        if self._fd is None:
            return
        try:
            import termios
            termios.tcflush(self._fd, termios.TCIFLUSH)
        except Exception:
            # Sem termios (pipe, teste): drena na mao, sem bloquear.
            while self._tem_mais():
                if not os.read(self._fd, 64):
                    break

    def ler(self):
        """Devolve o texto da tecla, ou ``"q"`` se a entrada acabou."""
        if self._fd is None:
            return "\n" if sys.stdin.readline() else "q"
        # os.read e NAO sys.stdin.read: o buffer do TextIOWrapper engoliria o
        # resto da sequencia da seta e sobraria um \x1b solto, que vira "sair".
        dados = os.read(self._fd, 3)
        if not dados:
            return "q"                      # entrada fechada: sai limpo
        if dados == b"\x1b" and self._tem_mais():
            dados += os.read(self._fd, 2)
        # latin-1 nunca levanta excecao e mapeia byte a byte, que e o que as
        # tabelas acima esperam. utf-8 poderia estourar num byte perdido e
        # derrubar a sonda no meio da pista.
        return dados.decode("latin-1", "replace")
