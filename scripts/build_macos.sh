#!/usr/bin/env bash
#
# Build ClipMind.app for macOS with PyInstaller, then package it.
#
# Usage:
#   scripts/build_macos.sh                 # build + zip
#   scripts/build_macos.sh --dmg           # build + DMG (macOS only)
#   scripts/build_macos.sh --bundle-model  # bundle the ~87 MB embedding model
#   scripts/build_macos.sh --no-install    # skip the PyInstaller pip install
#   scripts/build_macos.sh --codesign      # ad-hoc codesign the .app
#
# What it does:
#   1. finds the project venv (or falls back to python3),
#   2. makes sure PyInstaller is installed,
#   3. generates packaging/ClipMind.icns if it is missing,
#   4. runs PyInstaller with clipmind.spec,
#   5. verifies dist/ClipMind.app exists and reports its size,
#   6. packages it as a .zip and/or .dmg.
#
# It deliberately does NOT notarise or sign with a Developer ID; see the README
# for the unsigned-app workaround (right-click -> Open).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MAKE_DMG=0
BUNDLE_MODEL=0
DO_INSTALL=1
DO_CODESIGN=0

for arg in "$@"; do
  case "$arg" in
    --dmg) MAKE_DMG=1 ;;
    --zip) MAKE_DMG=0 ;;
    --bundle-model) BUNDLE_MODEL=1 ;;
    --no-install) DO_INSTALL=0 ;;
    --codesign) DO_CODESIGN=1 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: this script builds the macOS bundle; run it on macOS." >&2
  exit 1
fi

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="$(command -v python3)"
  echo "note: no .venv found; using $PY"
fi
echo "using $PY ($("$PY" --version))"

if [[ "$DO_INSTALL" == "1" ]]; then
  echo "==> installing build dependencies"
  "$PY" -m pip install --quiet --upgrade "pyinstaller>=6.0"
fi

if ! "$PY" -c "import PyInstaller" >/dev/null 2>&1; then
  echo "error: PyInstaller is not importable; re-run without --no-install." >&2
  exit 1
fi

if [[ ! -f "$ROOT/packaging/ClipMind.icns" ]]; then
  echo "==> generating placeholder icon"
  "$PY" "$ROOT/scripts/make_icon.py" || true
fi

echo "==> cleaning previous build"
rm -rf "$ROOT/build" "$ROOT/dist"

if [[ "$BUNDLE_MODEL" == "1" ]]; then
  export CLIPMIND_BUNDLE_MODEL=1
  echo "==> bundling the embedding model into the app"
fi

echo "==> running PyInstaller"
"$PY" -m PyInstaller --clean --noconfirm "$ROOT/clipmind.spec"

APP="$ROOT/dist/ClipMind.app"
if [[ ! -d "$APP" ]]; then
  echo "error: expected $APP was not produced." >&2
  exit 1
fi

if [[ "$DO_CODESIGN" == "1" ]] && command -v codesign >/dev/null 2>&1; then
  echo "==> ad-hoc signing"
  codesign --force --deep --sign - "$APP"
fi

echo "==> bundle built: $APP"
du -sh "$APP"
/usr/libexec/PlistBuddy -c "Print :LSUIElement" "$APP/Contents/Info.plist" 2>/dev/null \
  | sed 's/^/    LSUIElement: /' || true

mkdir -p "$ROOT/dist"
ZIP="$ROOT/dist/ClipMind.zip"
echo "==> writing $ZIP"
rm -f "$ZIP"
ditto -c -k --sequesterRsrc --keepParent "$APP" "$ZIP"
du -sh "$ZIP"

if [[ "$MAKE_DMG" == "1" ]]; then
  DMG="$ROOT/dist/ClipMind.dmg"
  echo "==> writing $DMG"
  rm -f "$DMG"
  hdiutil create -volname "ClipMind" -srcfolder "$APP" -ov -format UDZO "$DMG" >/dev/null
  du -sh "$DMG"
fi

cat <<'EOF'

Done. Verify the bundle by hand:
  1. open dist/ClipMind.app          # first launch: right-click -> Open
  2. a clipboard icon should appear in the menu bar (no Dock icon)
  3. grant Accessibility / Input Monitoring if the hotkey is used
  4. copy text, press the hotkey, confirm the popup opens
  5. check the data directory:
       ~/Library/Application Support/clipmind/   (clipmind.db, app.lock)
       ~/.config/clipmind/config.toml
  6. quit from the tray menu and confirm no stray process remains:
       pgrep -fl ClipMind
EOF
