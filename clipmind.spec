# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the ClipMind macOS menu-bar app.

Build with ``scripts/build_macos.sh`` (recommended) or directly::

    pyinstaller --clean --noconfirm clipmind.spec

Environment switches:

``CLIPMIND_BUNDLE_MODEL=1``
    Bundle the default model snapshot (``all-MiniLM-L6-v2``) inside the app (see
    ``packaging/clipmind_launcher.py`` for the ``HF_HOME`` wiring). Off by
    default: the app downloads the model once on first run instead. Override the
    model with ``CLIPMIND_BUNDLE_MODEL_NAME``.

Notes:
    * ``console=False`` + ``LSUIElement`` make this a menu-bar-only app.
    * The dependency set (PySide6 + torch + sentence-transformers) is large;
      a built ``.app`` is on the order of a gigabyte uncompressed. PyInstaller
      prunes unused Qt plugins, but the bulk is torch and PySide6.
    * This spec was written against PyInstaller 6.x. If a hook fails on a newer
      toolchain, the failing module is almost always a *lazily imported* one
      (``transformers``, ``sentence_transformers``, a ``pynput`` backend); add
      it to ``hiddenimports`` below.
"""

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(os.environ.get("SPECPATH", os.getcwd())).resolve()
APP_NAME = "ClipMind"
BUNDLE_ID = "com.clipmind.app"
ICON = ROOT / "packaging" / "ClipMind.icns"

BUNDLE_MODEL = os.environ.get("CLIPMIND_BUNDLE_MODEL") == "1"
# Keep in sync with clipmind.config.DEFAULT_EMBEDDING_MODEL.
BUNDLE_MODEL_NAME = os.environ.get(
    "CLIPMIND_BUNDLE_MODEL_NAME", "all-MiniLM-L6-v2"
)

# --------------------------------------------------------------------- analysis

# Lazily imported modules are invisible to PyInstaller's static trace, so the
# ones the app needs at runtime are listed explicitly.
hiddenimports = []
hiddenimports += collect_submodules("clipmind")
hiddenimports += collect_submodules("pynput")
hiddenimports += collect_submodules("sentence_transformers")
hiddenimports += [
    "pyperclip",
    # macOS integration: the Dock-icon policy and the hotkey listener.
    "objc",
    "AppKit",
    "Foundation",
    "Quartz",
    "CoreFoundation",
    "PyObjCTools",
    # The embedding stack (torch/transformers have PyInstaller hooks).
    "torch",
    "transformers",
    "scipy",
    "sklearn",
    "huggingface_hub",
]

datas = []
if BUNDLE_MODEL:
    hf_home = Path(
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    )
    # Bare names resolve under sentence-transformers' own org in the HF cache.
    _repo_id = BUNDLE_MODEL_NAME if "/" in BUNDLE_MODEL_NAME else (
        "sentence-transformers/" + BUNDLE_MODEL_NAME
    )
    repo_dir = "models--" + _repo_id.replace("/", "--")
    snapshot = hf_home / "hub" / repo_dir
    if not snapshot.is_dir():
        raise SystemExit(
            f"CLIPMIND_BUNDLE_MODEL=1 but {snapshot} is missing. Download the "
            "model once (e.g. `python -c \"from sentence_transformers import "
            f"SentenceTransformer; SentenceTransformer('{BUNDLE_MODEL_NAME}')\"`) "
            "or unset CLIPMIND_BUNDLE_MODEL."
        )
    datas.append((str(snapshot), f"clipmind_hf/hub/{repo_dir}"))

# Keep the bundle from pulling in unrelated heavy packages.
excludes = [
    "tkinter",
    "matplotlib",
    "IPython",
    "notebook",
    "pytest",
    "pytestqt",
    # torch's optional tensorboard integration is not installed and is not used.
    "torch.utils.tensorboard",
    "tensorboard",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtQml",
    "PySide6.QtQuick",
    "PySide6.Qt3DCore",
    "PySide6.QtMultimedia",
    "PySide6.QtDesigner",
    "PySide6.QtOpenGL",
]

a = Analysis(
    [str(ROOT / "packaging" / "clipmind_launcher.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # menu-bar app: never open a terminal window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)

app = BUNDLE(
    coll,
    name=f"{APP_NAME}.app",
    icon=str(ICON) if ICON.exists() else None,
    bundle_identifier=BUNDLE_ID,
    version="0.6.0",
    info_plist={
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleShortVersionString": "0.6.0",
        "CFBundleVersion": "0.6.0",
        "LSMinimumSystemVersion": "11.0",
        # Menu-bar-only: no Dock icon, no app switcher entry.
        "LSUIElement": True,
        "NSHumanReadableCopyright": "MIT licensed. See LICENSE.",
        # Prompt text shown if the OS ever asks for input monitoring.
        "NSAppleEventsUsageDescription": "ClipMind reads the clipboard to store it locally.",
        "NSHighResolutionCapable": True,
    },
)
