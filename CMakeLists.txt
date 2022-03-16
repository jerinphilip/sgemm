cmake_minimum_required(VERSION 3.22)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
project(bergamot_translator CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MKL)
message(STATUS "MKL: ${MKL_FOUND}, ${MKL_LIBRARIES}, ${MKL_INCLUDE_DIR}")

add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE ${MKL_LIBRARIES})
target_compile_definitions(main PUBLIC MKL_FOUND BLAS_FOUND)
target_include_directories(main PRIVATE ${MKL_INCLUDE_DIR})

