cmake_minimum_required(VERSION 3.10)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
project(bergamot_translator CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(MARIAN_USE_MKL "Compile with MKL support" ON)
option(MARIAN_USE_RUY_SGEMM "Compile with ruy sgemm" ON)
option(MARIAN_USE_ONNX_SGEMM "Compile with ONNX sgemm" OFF)
option(MARIAN_USE_APPLE_ACCELERATE "Compile with apple-accelerate support" ON)

add_subdirectory(3rd-party)

add_library(marian_sgemm STATIC src/tensor.cpp)
target_include_directories(marian_sgemm PUBLIC ${CMAKE_SOURCE_DIR})


if(MARIAN_USE_MKL)
	set(BLA_VENDOR Intel10_64ilp)
	find_package(MKL)
  target_compile_definitions(marian_sgemm PUBLIC MARIAN_USE_MKL)
  target_link_libraries(marian_sgemm PUBLIC MKL::MKL)
endif(MARIAN_USE_MKL)


if(MARIAN_USE_RUY_SGEMM)
	target_link_libraries(marian_sgemm PUBLIC ruy)
  target_compile_definitions(marian_sgemm PUBLIC MARIAN_USE_RUY_SGEMM)
endif(MARIAN_USE_RUY_SGEMM)


find_package(Threads REQUIRED)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address -static-libasan")

add_executable(main src/main.cpp)
target_link_libraries(main PUBLIC marian_sgemm)

add_executable(tests src/tests.cpp)
target_link_libraries(tests PUBLIC gtest_main gtest marian_sgemm)

