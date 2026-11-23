"""Botão do TopTop na prateleira (shelf) do Maya.

Instalar uma vez, colando no Script Editor (aba Python) e trocando o caminho
pela pasta deste repositório (a que contém a pasta ``toptop``)::

    import sys
    TOPTOP = r"C:/caminho/para/Toptop"
    if TOPTOP not in sys.path:
        sys.path.insert(0, TOPTOP)
    import toptop.shelf
    toptop.shelf.install()

Isso cria (ou reaproveita) a aba "TopTop" com um botão que abre a janela. O
Maya grava a prateleira nas preferências, então o botão continua lá nas
próximas sessões; o próprio botão põe o repositório no ``sys.path``.

Para ter ``import toptop`` disponível desde a abertura do Maya (para usar no
Script Editor sem o botão), ponha no ``userSetup.py`` (em
``Documentos/maya/2025/scripts/userSetup.py``)::

    import sys
    TOPTOP = r"C:/caminho/para/Toptop"
    if TOPTOP not in sys.path:
        sys.path.append(TOPTOP)

Se o repositório mudar de lugar, rode ``install()`` de novo a partir do lugar
novo: o botão existente é atualizado, não duplicado.
"""

from __future__ import annotations

import os
import re

DEFAULT_SHELF = "TopTop"
BUTTON_LABEL = "TopTop"
#: identifica o botão do TopTop entre os outros da prateleira
BUTTON_TAG = "toptop/janela"
BUTTON_IMAGE = "pythonFamily.png"
BUTTON_ANNOTATION = "TopTop: retopologia hard surface (abre a janela)"


def repo_root() -> str:
    """Pasta que contém o pacote ``toptop`` (o que vai para o ``sys.path``)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def button_command(root: str | None = None) -> str:
    """Código Python do botão: põe ``root`` no ``sys.path`` se faltar e abre a janela."""
    root = repo_root() if root is None else str(root)
    return (
        "import sys\n"
        f"_toptop_root = {root!r}\n"
        "if _toptop_root not in sys.path:\n"
        "    sys.path.insert(0, _toptop_root)\n"
        "import toptop.ui.window\n"
        "toptop.ui.window.show()\n"
    )


def button_flags(command: str) -> dict[str, str]:
    """Flags do ``shelfButton`` (criar e editar): rótulo, dica, ícone e o código Python."""
    return dict(
        label=BUTTON_LABEL,
        annotation=BUTTON_ANNOTATION,
        image=BUTTON_IMAGE,
        imageOverlayLabel=BUTTON_LABEL,
        command=command,
        sourceType="python",
    )


def _find_button(shelf: str) -> str | None:
    """Botão do TopTop já existente na prateleira (pela ``docTag``)."""
    from maya import cmds

    for child in cmds.shelfLayout(shelf, query=True, childArray=True) or []:
        if not cmds.shelfButton(child, exists=True):
            continue
        if cmds.shelfButton(child, query=True, docTag=True) == BUTTON_TAG:
            return child
    return None


def install(shelf_name: str = DEFAULT_SHELF, root: str | None = None) -> str:
    """Cria (ou reaproveita) a aba ``shelf_name`` e o botão que abre a janela do TopTop.

    Devolve o nome do botão (vazio no Maya sem interface, onde não há prateleira).
    """
    from maya import cmds, mel

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", shelf_name or ""):
        raise ValueError(f"nome de prateleira inválido: {shelf_name!r} (letras, números e _)")
    if cmds.about(batch=True):
        print("TopTop: o Maya está sem interface; não há prateleira para instalar o botão.")
        return ""
    top = mel.eval("$toptopShelfTop = $gShelfTopLevel")
    if not top or not cmds.shelfTabLayout(top, exists=True):
        raise RuntimeError("TopTop: não achei a área das prateleiras do Maya ($gShelfTopLevel).")

    if cmds.shelfLayout(shelf_name, exists=True):
        shelf = shelf_name
    else:
        # addNewShelfTab registra a aba nas preferências (ela volta na próxima sessão)
        shelf = mel.eval(f'addNewShelfTab "{shelf_name}"') or shelf_name
        if not cmds.shelfLayout(shelf, exists=True):
            shelf = cmds.shelfLayout(shelf_name, parent=top)

    flags = button_flags(button_command(root))
    button = _find_button(shelf)
    if button is None:
        button = cmds.shelfButton(parent=shelf, docTag=BUTTON_TAG, **flags)
        action = "criado"
    else:
        cmds.shelfButton(button, edit=True, **flags)
        action = "atualizado"
    cmds.shelfTabLayout(top, edit=True, selectTab=shelf.split("|")[-1])
    cmds.saveAllShelves(top)
    print(
        f"TopTop: botão {action} na prateleira '{shelf.split('|')[-1]}'.\n"
        f"  Clique nele para abrir a janela (repositório: {root or repo_root()}).\n"
        "  Selecione uma malha, escolha o modo e o nível de detalhe e clique em Retopologizar.\n"
        "  Para ter 'import toptop' desde a abertura do Maya, veja o userSetup.py na ajuda de toptop.shelf."
    )
    return button
