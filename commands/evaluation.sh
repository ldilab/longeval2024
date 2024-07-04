set -e

RANKING_PATH=$1
YEAR=$2
TERM=$3
DTYPE=$4
OUTPUT=$5
CLEAN=$6

python3 -m src.utils.evaluation \
  --clean \
  $CLEAN \
  --ranking \
  $RANKING_PATH \
  --year \
  $YEAR \
  --term \
  $TERM \
  --dtype \
  $DTYPE \
  --output \
  $OUTPUT
