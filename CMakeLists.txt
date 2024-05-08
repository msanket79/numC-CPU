cmake_minimum_required(VERSION 3.22)

project(numC LANGUAGES CXX )
find_package(OpenMP REQUIRED)


set(SOURCES 
    main.cpp
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp:llvm")

include_directories(include
                    openblas/include
		   
)

link_directories(openblas/lib)

add_executable(main ${SOURCES})

set_target_properties( main PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/../bin
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/../bin
)

target_link_libraries(main libopenblas.lib  OpenMP::OpenMP_CXX)