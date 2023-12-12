set -eu
FILE=test_aadiff
nvcc ${FILE}.cu -o ${FILE} -std=c++14 -lcublas
date
./${FILE}
date
