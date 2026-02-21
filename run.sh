#!/bin/bash

echo "Building the project with CMake..."
mkdir -p build
cd build
cmake ..
make

echo "----------------------------------------"
echo "Running the project..."
echo "----------------------------------------"
./ObjectDetection
