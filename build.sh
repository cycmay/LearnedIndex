mkdir build
# rm -rf build/*
mkdir bin
mkdir lib
cd build
cmake ..
make

cd ../
cd bin
./LearnedIndex
