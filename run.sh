#!/bin/bash
mkdir -p build
cd build
cmake ..
make

echo "----------------------------------------"
echo "Running the project..."
echo "----------------------------------------"
./ObjectDetection
