cmake_minimum_required(VERSION 3.15)
project(mycpp)


set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fopenmp -g3 -O3")


find_package(Boost REQUIRED COMPONENTS system program_options)
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(
  include
  ${BLAS_INCLUDE_DIR}
)

file(GLOB MY_SRC ${PROJECT_SOURCE_DIR}/src/*.cpp)

set(PYBIND11_CPP_STANDARD -std=c++14)

pybind11_add_module(mycpp src/app/pybind_api.cpp ${MY_SRC})
target_link_libraries(mycpp PRIVATE ${Boost_LIBRARIES} ${OpenMP_CXX_FLAGS} Eigen3::Eigen)