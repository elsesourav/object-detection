cmake_minimum_required(VERSION 3.10)
project(ObjectDetection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add potential Homebrew installation paths for Apple Silicon (M1/M2/M3)
set(OpenCV_DIR "/opt/homebrew/lib/cmake/opencv4")
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include OpenCV directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable
add_executable(ObjectDetection main.cpp)

# Link OpenCV libraries
target_link_libraries(ObjectDetection ${OpenCV_LIBS})

# Print OpenCV information
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
