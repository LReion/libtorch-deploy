cmake_minimum_required(VERSION 3.24)
project(python2cpp)
set(CMAKE_PREFIX_PATH "C:/Users/reion/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(OpenCV REQUIRED)
include_directories("C:/Users/reion/libtorch/include"
        "C:/Users/reion/libtorch/include/torch/csrc/api/include"
        "C:/Users/reion/opencv/build/include"
        "C:/Users/reion/opencv/build/include/opencv2"
        "C:/Users/reion/Desktop/include")

set(CMAKE_CXX_STANDARD 17)

add_executable(main main.cpp)
target_link_libraries(main ${TORCH_LIBRARIES} ${OpenCV_LIBS})


# add_executable(demo demo.cpp)
