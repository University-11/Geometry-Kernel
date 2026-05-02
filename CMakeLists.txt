cmake_minimum_required(VERSION 3.16)
project(GeometryKernel LANGUAGES CXX)

add_library(geometry_kernel INTERFACE)
target_include_directories(geometry_kernel INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(geometry_kernel INTERFACE cxx_std_17)

enable_testing()

add_executable(geomemory_test tests/cpp/geomemory_test.cpp)
target_link_libraries(geomemory_test PRIVATE geometry_kernel)
add_test(NAME geomemory_test COMMAND geomemory_test)

if(MSVC)
    target_compile_options(geomemory_test PRIVATE /W4)
else()
    target_compile_options(geomemory_test PRIVATE -Wall -Wextra -Wpedantic)
endif()
