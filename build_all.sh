DIR=$(pwd)

cd $DIR/mycpp/ && mkdir -p build && cd build && cmake .. -DPYTHON_EXECUTABLE=$(which python) && make -j11
cd $DIR/bundlesdf/mycuda && rm -rf build *egg* && pip install -e .

cd ${DIR}
