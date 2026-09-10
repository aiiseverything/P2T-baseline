#!/usr/bin/env bash
set -euo pipefail
INTERNAL_PYPI=http://mirrors.i.h.pjlab.org.cn/repository/pypi-proxy/simple/
python3 -m pip install --quiet --retries 5 --timeout 120 \
  --index-url "$INTERNAL_PYPI" --trusted-host mirrors.i.h.pjlab.org.cn \
  peft==0.20.0 'pyarrow>=15,<22'
python3 -c 'import peft, pyarrow; print("peft", peft.__version__, "pyarrow", pyarrow.__version__)'
