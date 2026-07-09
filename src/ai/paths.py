"""Resolucao de caminhos do projeto e injecao do pacote `agents` do CARLA no
`sys.path`.

O pacote `agents` (que traz o `BasicAgent`, nosso expert de direcao) NAO vem no
wheel `carla` instalado pelo pip — ele mora dentro da instalacao do simulador em
`CARLA_0.9.16/PythonAPI/carla/agents`. Este modulo localiza esse diretorio e o
adiciona ao `sys.path` sob demanda, para que `from agents.navigation.basic_agent
import BasicAgent` funcione.
"""

# imports
import os
import sys

# constants
_THIS_FILE = os.path.abspath(__file__)
SRC_DIR = os.path.dirname(os.path.dirname(_THIS_FILE))          # .../src
PROJECT_ROOT = os.path.dirname(SRC_DIR)                          # raiz do projeto
CARLA_PYTHONAPI_DIR = os.path.join(
    PROJECT_ROOT, "CARLA_0.9.16", "PythonAPI", "carla"
)


# functions
def ensure_src_on_path() -> None:
    """Garante que `.../src` esteja no `sys.path` (para `import core`, `import ai`).

    Necessario quando o script e executado como `python src/ai/eval_closedloop.py`,
    caso em que o `sys.path[0]` aponta para `src/ai`, nao para `src`.
    """
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)


def ensure_carla_agents_on_path() -> None:
    """Adiciona o diretorio do pacote `agents` do CARLA ao `sys.path`.

    Raises:
        FileNotFoundError: Se a instalacao do CARLA nao for encontrada na raiz.
    """
    if not os.path.isdir(CARLA_PYTHONAPI_DIR):
        raise FileNotFoundError(
            "Pacote 'agents' do CARLA nao encontrado em '%s'. "
            "Confirme que o CARLA_0.9.16 esta extraido na raiz do projeto."
            % CARLA_PYTHONAPI_DIR
        )
    if CARLA_PYTHONAPI_DIR not in sys.path:
        sys.path.insert(0, CARLA_PYTHONAPI_DIR)
