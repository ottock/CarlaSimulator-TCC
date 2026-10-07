"""Os nomes que os scripts de hardware pedem ao jetson_runtime existem?

Por que este arquivo existe: `hardware/sonda_estatica.py` foi para a pista
chamando `jr.TensorRTEngine` e `jr.Actuator`. A classe se chama `TrtEngine` e
`Pca9685Actuator`. O erro so apareceu no carro, DEPOIS de subir camera, LiDAR e
engine -- com o operador agachado na pista, sem ver a tela.

Nenhum teste podia ter pegado isso importando o modulo: `jetson_runtime` puxa
tensorrt, pycuda, serial e Adafruit_PCA9685, que nao existem no PC de
desenvolvimento. Entao a verificacao aqui e ESTATICA, por AST: le os dois
arquivos como texto, junta os nomes de topo que o runtime define e confere que
todo `jr.<algo>` referenciado existe. Nao importa nada, logo roda no Windows.

O que este teste NAO cobre: assinatura de argumentos e nome de metodo
(`lidar.read_points()`). Para isso seria preciso importar de verdade. Ele cobre
a classe de erro que de fato nos custou uma ida a pista.
"""
import ast
import io
import os

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RUNTIME = os.path.join(_REPO, "hardware", "jetson_runtime.py")


def _arvore(caminho):
    with io.open(caminho, encoding="utf-8") as fh:
        return ast.parse(fh.read(), filename=caminho)


def _nomes_de_topo(arvore):
    """Tudo que `import jetson_runtime` passa a expor: classes, funcoes e
    constantes de modulo. Inclui os imports do proprio runtime, porque
    `jr.np` tambem funcionaria."""
    nomes = set()
    for no in arvore.body:
        if isinstance(no, (ast.ClassDef, ast.FunctionDef)):
            nomes.add(no.name)
        elif isinstance(no, ast.Assign):
            for alvo in no.targets:
                if isinstance(alvo, ast.Name):
                    nomes.add(alvo.id)
        elif isinstance(no, ast.AnnAssign) and isinstance(no.target, ast.Name):
            nomes.add(no.target.id)
        elif isinstance(no, (ast.Import, ast.ImportFrom)):
            for alias in no.names:
                nomes.add(alias.asname or alias.name.split(".")[0])
    return nomes


def _apelidos(arvore):
    """Como este arquivo chamou o modulo: `import jetson_runtime as jr` -> {'jr'}."""
    apelidos = set()
    for no in ast.walk(arvore):
        if isinstance(no, ast.Import):
            for alias in no.names:
                if alias.name == "jetson_runtime":
                    apelidos.add(alias.asname or alias.name)
    return apelidos


def _atributos_pedidos(arvore, apelidos):
    """Todo `jr.X` do arquivo, com a linha, para o erro dizer onde esta."""
    pedidos = []
    for no in ast.walk(arvore):
        if (isinstance(no, ast.Attribute) and isinstance(no.value, ast.Name)
                and no.value.id in apelidos):
            pedidos.append((no.attr, no.lineno))
    return pedidos


def _scripts_que_usam_o_runtime():
    pasta = os.path.join(_REPO, "hardware")
    achados = []
    for nome in sorted(os.listdir(pasta)):
        if not nome.endswith(".py") or nome == "jetson_runtime.py":
            continue
        caminho = os.path.join(pasta, nome)
        arvore = _arvore(caminho)
        if _apelidos(arvore):
            achados.append((nome, caminho, arvore))
    return achados


def test_the_runtime_file_parses():
    assert _nomes_de_topo(_arvore(_RUNTIME)), "jetson_runtime.py nao expoe nada?"


def test_someone_actually_uses_the_runtime():
    # Se este teste cair, o de baixo passou a nao verificar nada -- provavelmente
    # porque o `import jetson_runtime as jr` mudou de forma.
    assert _scripts_que_usam_o_runtime(), (
        "nenhum script em hardware/ importa jetson_runtime; o teste abaixo virou vazio")


@pytest.mark.parametrize("nome,caminho,arvore", _scripts_que_usam_o_runtime(),
                         ids=lambda v: v if isinstance(v, str) and v.endswith(".py") else "")
def test_every_runtime_name_the_script_asks_for_exists(nome, caminho, arvore):
    existentes = _nomes_de_topo(_arvore(_RUNTIME))
    apelidos = _apelidos(arvore)
    faltando = [(attr, linha) for attr, linha in _atributos_pedidos(arvore, apelidos)
                if attr not in existentes]
    assert not faltando, "\n".join(
        ["%s pede nomes que jetson_runtime nao tem:" % nome]
        + ["  linha %d: jr.%s" % (linha, attr) for attr, linha in faltando]
        + ["nomes disponiveis: %s" % ", ".join(sorted(n for n in existentes
                                                      if n[0].isupper()))])
