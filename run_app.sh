#!/usr/bin/env bash
cd "$(dirname "$0")"

PYTHON=""
for candidate in py python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" -c "import sys" >/dev/null 2>&1; then
      PYTHON="$candidate"
      [[ "$candidate" == "py" ]] && PYTHON="py -3"
      break
    fi
  fi
done

if [[ -z "$PYTHON" && -f "$LOCALAPPDATA/Programs/Python/Python312/python.exe" ]]; then
  PYTHON="$LOCALAPPDATA/Programs/Python/Python312/python.exe"
fi

if [[ -z "$PYTHON" ]]; then
  echo "Python 3 nao encontrado. Instale Python 3.10+ e adicione ao PATH."
  exit 1
fi

if [[ ! -f "app.py" ]]; then
  echo "app.py nao encontrado. Execute este script dentro da pasta do projeto."
  exit 1
fi

if [[ ! -f "steam_dataset/steamds.csv" ]]; then
  echo "Dataset nao encontrado: steam_dataset/steamds.csv"
  exit 1
fi

if [[ ! -f "models/random_forest.joblib" ]]; then
  echo "Modelo nao treinado. Treinando agora..."
  eval "$PYTHON -m src.train" || exit 1
fi

echo "Abrindo interface Streamlit..."
eval "$PYTHON -m streamlit run app.py"
