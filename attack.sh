# textfooler
python3 attack.py --datapath-prefix $1 --attack-method textfooler --target-model roberta-base

# fast
python3 attack.py --datapath-prefix $1 --attack-method fast-alzantot --target-model roberta-base

# iga
python3 attack.py --datapath-prefix $1 --attack-method iga --target-model roberta-base

# pwws
python3 attack.py --datapath-prefix $1 --attack-method pwws --target-model roberta-base
