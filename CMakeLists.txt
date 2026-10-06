cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(CLEAVE LANGUAGES C CXX)

message(STATUS "CMake Version: ${CMAKE_VERSION}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_TESTS "Build C++ unit tests" OFF)
option(CLEAVE_SUPEROPTIMIZER_BUILD_UNIT_TEST "Build superoptimizer C++ unit tests" OFF)

set(CLEAVE_SUPEROPTIMIZER_ENABLE_DEBUG_LOGGING_DEFAULT OFF)
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  set(CLEAVE_SUPEROPTIMIZER_ENABLE_DEBUG_LOGGING_DEFAULT ON)
endif()
option(CLEAVE_SUPEROPTIMIZER_ENABLE_DEBUG_LOGGING
       "Compile Cleave Superoptimizer debug and VLOG logging"
       ${CLEAVE_SUPEROPTIMIZER_ENABLE_DEBUG_LOGGING_DEFAULT})

find_package(OpenMP REQUIRED)

if(NOT Z3_CXX_INCLUDE_DIRS OR NOT Z3_LIBRARIES)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
            "from pathlib import Path; import z3; root=Path(z3.__file__).resolve().parent; libdir=root/'lib'; libs=[libdir/'libz3.so', libdir/'libz3.dylib', libdir/'z3.dll']; lib=next((p for p in libs if p.exists()), None); assert (root/'include').exists(), root/'include'; assert lib is not None, libdir; print(root/'include'); print(lib)"
    OUTPUT_VARIABLE CLEAVE_Z3_INFO
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE CLEAVE_Z3_RESULT)
  if(NOT CLEAVE_Z3_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to locate z3-solver headers and library")
  endif()
  string(REPLACE "\n" ";" CLEAVE_Z3_INFO_LIST "${CLEAVE_Z3_INFO}")
  list(GET CLEAVE_Z3_INFO_LIST 0 Z3_CXX_INCLUDE_DIRS)
  list(GET CLEAVE_Z3_INFO_LIST 1 Z3_LIBRARIES)
endif()

message(STATUS "Z3_CXX_INCLUDE_DIRS: ${Z3_CXX_INCLUDE_DIRS}")
message(STATUS "Z3_LIBRARIES: ${Z3_LIBRARIES}")

file(GLOB_RECURSE CLEAVE_SUPEROPTIMIZER_SRCS CONFIGURE_DEPENDS
  src/superoptimizer/*.cc
)

add_library(cleave_superoptimizer STATIC ${CLEAVE_SUPEROPTIMIZER_SRCS})

target_compile_features(cleave_superoptimizer PUBLIC cxx_std_17)
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  target_compile_options(cleave_superoptimizer PRIVATE -O0 -g -Wall)
else()
  target_compile_options(cleave_superoptimizer PRIVATE -O2 -Wall)
endif()
target_compile_definitions(cleave_superoptimizer
  PUBLIC
    $<$<BOOL:${CLEAVE_SUPEROPTIMIZER_ENABLE_DEBUG_LOGGING}>:CLEAVE_SUPEROPTIMIZER_ENABLE_DEBUG_LOGGING=1>)

target_include_directories(cleave_superoptimizer
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/3rdparty/json/include
    ${Z3_CXX_INCLUDE_DIRS})

target_link_libraries(cleave_superoptimizer
  PUBLIC
    ${Z3_LIBRARIES}
    OpenMP::OpenMP_CXX)

install(TARGETS cleave_superoptimizer
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include
    DESTINATION .)

if (BUILD_TESTS OR CLEAVE_SUPEROPTIMIZER_BUILD_UNIT_TEST)
  enable_testing()
  add_subdirectory(tests/superoptimizer/cpp)
endif()
