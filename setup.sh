#!/usr/bin/env bash
set -e  # exit on first error

command -v uv >/dev/null 2>&1 || pip install uv

VENV="difflogic_verification"

# create the venv if it doesn't exist
uv venv "$VENV"

# activate it so uv installs inside
source "$VENV/bin/activate"

# install project dependencies
uv pip install -r requirements.txt

# build Kissat if missing
if ! [ -x kissat/build/kissat ]; then
    echo "kissat not found, building kissat"
    git clone https://github.com/arminbiere/kissat.git
    cd kissat
    ./configure
    make
    cd ..
fi

echo "✅ Setup complete. Run 'source $VENV/bin/activate'"
