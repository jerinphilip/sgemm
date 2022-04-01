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


add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE ${MKL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS} ruy)
target_compile_definitions(main PUBLIC MKL_FOUND BLAS_FOUND)
target_include_directories(main PRIVATE ${MKL_INCLUDE_DIR})

add_executable(tests src/tests.cpp)
target_link_libraries(tests gtest_main gtest ruy ${MKL_LIBRARIES} ${CMAKE_DL_LIBS} ${CMAKE_THREAD_LIBS_INIT})
target_compile_definitions(tests PUBLIC MKL_FOUND BLAS_FOUND)
target_include_directories(tests PRIVATE ${MKL_INCLUDE_DIR})

