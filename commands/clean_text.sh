YEAR=$1
TERM=$2
DTYPE=$3

echo "============================"
echo "YEAR: $YEAR"
echo "TERM: $TERM"
echo "DTYPE: $DTYPE"
echo "============================"

python3 -m src.preprocessing.clean_text --year $YEAR --term $TERM --dtype $DTYPE