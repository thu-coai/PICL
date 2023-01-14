a=${1-10}
b=${2-20}
c=${3-30}

OPTS=""
OPTS+=" --a ${a}"
OPTS+=" --b ${b}"
OPTS+=" --c ${c}"

python3 test_args.py ${OPTS} $@