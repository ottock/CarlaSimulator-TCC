"""Quando uma corrida em malha fechada conta como limpa (Fase 6c).

MEDIDO: o eval_track reportou "1/1 pistas limpas" para uma corrida em que o
carro nao saiu do lugar -- mean_speed=0.0, mean_dev=0.00, offlane=0,
collisions=0. O criterio era so "nao bateu e nao saiu da pista", e um carro
parado satisfaz os dois.

Era a mesma familia de falha silenciosa que a coleta ja tinha mostrado duas
vezes: o pipeline reporta sucesso sobre nada. Aqui o criterio passa a exigir
que o carro tenha ANDADO.

Puro: so le o dicionario de metricas, testavel sem simulador.
"""

V_MIN_MS = 0.5        # abaixo disto e arrasto, nao pilotagem (o expert faz 1,74)
DIST_MIN_M = 20.0     # andou pouco demais para dizer qualquer coisa


def motivo(resumo):
    """Por que a corrida NAO e limpa; string vazia quando ela e."""
    if int(resumo.get("collisions", 0)) > 0:
        return "colisoes: %d" % int(resumo["collisions"])
    if int(resumo.get("offlane", 0)) > 0:
        return "saiu da pista em %d passos" % int(resumo["offlane"])
    if "chegou_ao_fim" in resumo:
        # Pista com fim (2026-10-09): andar pouco e o normal numa pista curta; o
        # que conta e ter parado no fim, e nao antes.
        if not resumo["chegou_ao_fim"]:
            return "nao chegou ao fim (parou a %.1f m)" % float(resumo.get("restante_m", -1.0))
        return ""
    if float(resumo.get("mean_speed", 0.0)) < V_MIN_MS:
        return "carro parado ou arrastando (%.2f m/s)" % float(resumo.get("mean_speed", 0.0))
    if float(resumo.get("distance_m", 0.0)) < DIST_MIN_M:
        return "andou so %.1f m" % float(resumo.get("distance_m", 0.0))
    return ""


def corrida_limpa(resumo):
    """True quando a corrida realmente dirigiu a pista sem bater nem sair."""
    return motivo(resumo) == ""
