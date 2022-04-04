cmake_minimum_required(VERSION 3.10)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
project(bergamot_translator CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MKL)
find_package(Threads REQUIRED)
message(STATUS "MKL: ${MKL_FOUND}, ${MKL_LIBRARIES}, ${MKL_INCLUDE_DIR}")

# Enable ASAN in debug builds
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address -static-libasan")

add_subdirectory(3rd-party)

set(EXT_LIBS ${MKL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS} ruy)
set(INCLUDE_DIRS ${MKL_INCLUDE_DIR} ${CMAKE_SOURCE_DIR})

list(APPEND BINS main)
foreach(BIN IN LISTS BINS)
  add_executable(${BIN} src/${BIN}.cpp)
  target_link_libraries(${BIN} PRIVATE ${EXT_LIBS})
  target_compile_definitions(${BIN} PUBLIC MKL_FOUND BLAS_FOUND)
  target_include_directories(${BIN} PRIVATE ${INCLUDE_DIRS})
endforeach()

add_executable(tests src/tests.cpp)
target_link_libraries(tests gtest_main gtest ${EXT_LIBS})
target_compile_definitions(tests PUBLIC MKL_FOUND BLAS_FOUND)
target_include_directories(tests PRIVATE ${INCLUDE_DIRS})

