"""Deteccao de carro encravado durante a coleta (Fase 6c).

MEDIDO, duas vezes: o ego encrava num empurrao de recuperacao e nunca mais sai.
Como o veiculo nao e recriado entre episodios, TODOS os seguintes gravam 1200
quadros de um carro imovel -- com o expert mandando trava total, porque ele
continua tentando voltar a linha. O relatorio dizia "kept 1200, dropped 0" nos
doze episodios, e so a velocidade media agregada denunciou.

Quadros desses nao sao neutros: ensinam "esta imagem -> esterco maximo" para
uma imagem que nao muda. Pior que perder a coleta e nao perceber que perdeu.

Puro: aritmetica de passos, testavel sem simulador.
"""


class StuckDetector:
    """Avisa quando a velocidade ficou abaixo do limiar por N passos seguidos.

    Dispara UMA vez por encrave: depois de avisar, so volta a avisar se o carro
    tiver andado no meio. Sem isso um encrave viraria uma enxurrada de
    reposicionamentos, um por passo.
    """

    def __init__(self, v_min=0.3, steps=40):
        if int(steps) < 1:
            raise ValueError("janela tem de ser >= 1 passo (recebi %r)" % steps)
        self.v_min = float(v_min)
        self.steps = int(steps)
        self._parado = 0
        self._avisado = False

    def update(self, speed):
        """Registra a velocidade deste passo; True quando acaba de encravar."""
        if float(speed) >= self.v_min:
            self._parado = 0
            self._avisado = False
            return False
        self._parado += 1
        if self._parado >= self.steps and not self._avisado:
            self._avisado = True
            return True
        return False
