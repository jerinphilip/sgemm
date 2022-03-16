cmake_minimum_required(VERSION 3.10)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
project(bergamot_translator CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MKL)
find_package(Threads REQUIRED)
message(STATUS "MKL: ${MKL_FOUND}, ${MKL_LIBRARIES}, ${MKL_INCLUDE_DIR}")

add_subdirectory(3rd-party)

add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE ${MKL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS} ruy)
target_compile_definitions(main PUBLIC MKL_FOUND BLAS_FOUND)
target_include_directories(main PRIVATE ${MKL_INCLUDE_DIR})

