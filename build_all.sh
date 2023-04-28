DIR=$(pwd)

cd $DIR/mycpp/ && mkdir -p build && cd build && cmake .. && make -j11
cd /kaolin && rm -rf build *egg* && pip install -e .
cd $DIR/bundlesdf/mycuda && rm -rf build *egg* && pip install -e .

cd ${DIR}