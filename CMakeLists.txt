project(bergamot_translator CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package( BLAS REQUIRED )
message( STATUS BLAS found: ${BLAS_LIBRARIES} )

add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE ${BLAS_LIBRARIES})

