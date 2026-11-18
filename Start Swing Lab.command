#!/bin/zsh
cd -- "${0:A:h}" || exit 1
if [[ ! -x .venv/bin/python ]]; then
  print 'The project Python environment is missing. Follow README setup instructions.'
  read 'reply?Press Return to close.'
  exit 1
fi
.venv/bin/python scripts/local_app.py
