cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(hiptt CXX)

if ("x" STREQUAL "x${CMAKE_BUILD_TYPE}")
set(CMAKE_BUILD_TYPE Release)
endif()

if (UNIX)
# Use $ORIGIN or @loader_path, which allow to pick up
# dependencies relative to the executable folder
if (APPLE)
set(CMAKE_INSTALL_RPATH "@loader_path")
else()
set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()
endif()

option(ENABLE_NVTOOLS "Enable nvvp profiling of CPU code" OFF)

if ("x${Python3_VERSION}" STREQUAL "x")
find_package(Python3 COMPONENTS Interpreter Development)
else()
find_package(Python3 ${Python3_VERSION} EXACT COMPONENTS Interpreter Development)
endif()
set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
message(STATUS "PYTHON_EXECUTABLE = ${PYTHON_EXECUTABLE}")

set(PYBIND11_PYTHON_VERSION ${Python3_VERSION})
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/pybind11)

file(GLOB ${PROJECT_NAME}_SOURCE_FILES "src/*.cu")
file(GLOB ${PROJECT_NAME}_bench_SOURCE_FILES "src/bench/bench.cu")
file(GLOB ${PROJECT_NAME}_test_SOURCE_FILES "src/test/test.cu")

include(CheckLanguage)
check_language(HIP)
check_language(CUDA)

if(CMAKE_HIP_COMPILER)

enable_language(HIP)

if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
endif()
set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
find_package(hip REQUIRED)

set(CMAKE_HIP_FLAGS_DEBUG "-ggdb -fstandalone-debug -Xclang -O0 -Xclang -gcodeview" ${CMAKE_HIP_FLAGS})

set_source_files_properties(${${PROJECT_NAME}_SOURCE_FILES} PROPERTIES LANGUAGE HIP)
set_source_files_properties(${${PROJECT_NAME}_bench_SOURCE_FILES} PROPERTIES LANGUAGE HIP)
set_source_files_properties(${${PROJECT_NAME}_test_SOURCE_FILES} PROPERTIES LANGUAGE HIP)

elseif(CMAKE_CUDA_COMPILER)

enable_language(CUDA)

set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0" ${CMAKE_CUDA_FLAGS})

else()

message(FATAL_ERROR "CUDA or HIP backend must be available, aborting")

endif()

# ENABLE_NVTOOLS
if(ENABLE_NVTOOLS)
    add_definitions(-DENABLE_NVTOOLS)
    list(APPEND CUDA_NVCC_FLAGS -lnvToolsExt)
    link_libraries(-lnvToolsExt)
endif()

add_library(${PROJECT_NAME} STATIC ${${PROJECT_NAME}_SOURCE_FILES})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE TRUE)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES native)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
if(CMAKE_HIP_COMPILER)
target_link_libraries(${PROJECT_NAME} hip::amdhip64)
endif()

add_executable(${PROJECT_NAME}_bench ${${PROJECT_NAME}_bench_SOURCE_FILES})
target_compile_features(${PROJECT_NAME}_bench PUBLIC cxx_std_17)
target_link_libraries(${PROJECT_NAME}_bench PUBLIC ${PROJECT_NAME})

add_executable(${PROJECT_NAME}_test ${${PROJECT_NAME}_test_SOURCE_FILES})
target_compile_features(${PROJECT_NAME}_test PUBLIC cxx_std_17)
target_link_libraries(${PROJECT_NAME}_test PUBLIC ${PROJECT_NAME})

pybind11_add_module(${PROJECT_NAME}_python "src/python/${PROJECT_NAME}.cu" "src/python/${PROJECT_NAME}_module.cu")
target_include_directories(${PROJECT_NAME}_python PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(${PROJECT_NAME}_python PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_compile_definitions(${PROJECT_NAME}_python PRIVATE ${PROJECT_NAME}_EXPORTS)
target_link_libraries(${PROJECT_NAME}_python PRIVATE ${PROJECT_NAME})
set_target_properties(${PROJECT_NAME}_python PROPERTIES LIBRARY_OUTPUT_NAME "${PROJECT_NAME}")

install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION . LIBRARY DESTINATION .)
install(FILES include/hiptt.h DESTINATION include)
install(TARGETS ${PROJECT_NAME}_python ${PROJECT_NAME}_bench ${PROJECT_NAME}_test DESTINATION .)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/examples DESTINATION .)

