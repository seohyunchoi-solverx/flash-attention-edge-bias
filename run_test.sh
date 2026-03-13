ㅡ #!/bin/bash
source .venv/bin/activate
python3 tests/test_edge_bias.py
cd /home/seohyun/flash-attention
source .venv/bin/activate
python3 tests/test_edge_bias.py
