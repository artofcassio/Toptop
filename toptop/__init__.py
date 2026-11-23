"""TopTop: retopologia automática de hard surface para Maya (100% quads).

Os módulos em ``toptop.core``, ``toptop.io`` e ``toptop.engine.base`` são
Python/NumPy puros e rodam sem Maya. Tudo que importa ``maya`` faz isso dentro
das funções (ou em módulos carregados só dentro do Maya), para que o pacote
possa ser importado e testado fora dele.
"""

__version__ = "0.1.0"
