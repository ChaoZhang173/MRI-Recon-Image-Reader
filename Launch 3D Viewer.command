#!/bin/zsh
set -eu
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  print "First install the optional 3D environment as described in README.md."
  read -r "reply?Press Return to close. "
  exit 1
fi
exec .venv/bin/python viewer3d.py "$@"
