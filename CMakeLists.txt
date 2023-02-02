cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(hiptt)

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

# enable CXX
enable_language(CXX)
if (NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 14)
endif (NOT CMAKE_CXX_STANDARD)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_CXX_EXTENSIONS)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif (NOT CMAKE_CXX_EXTENSIONS)

# enable HIP
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

if (NOT CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
endif (NOT CMAKE_CUDA_STANDARD)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if (NOT CMAKE_CUDA_EXTENSIONS)
    set(CMAKE_CUDA_EXTENSIONS OFF)
endif (NOT CMAKE_CUDA_EXTENSIONS)

message(STATUS "CUDA Include Dirs: " ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
message(STATUS "CUDA Host Compiler: " ${CMAKE_CUDA_HOST_COMPILER})

if ("x${Python3_VERSION}" STREQUAL "x")
find_package(Python3 COMPONENTS Interpreter Development)
else()
find_package(Python3 ${Python3_VERSION} EXACT COMPONENTS Interpreter Development)
endif()
set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
message(STATUS "PYTHON_EXECUTABLE = ${PYTHON_EXECUTABLE}")

set(PYBIND11_PYTHON_VERSION ${Python3_VERSION})
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/pybind11)

option(ENABLE_NVTOOLS "Enable nvvp profiling of CPU code" OFF)
option(ENABLE_NO_ALIGNED_ALLOC "Enable aligned_alloc() function implemented in hipTT" OFF)
option(ENABLE_UMPIRE "Enable umpire for memory management" OFF)
include(CheckFunctionExists)

# ENABLE_NVTOOLS
if(ENABLE_NVTOOLS)
    add_definitions(-DENABLE_NVTOOLS)
    list(APPEND CUDA_NVCC_FLAGS -lnvToolsExt)
    link_libraries(-lnvToolsExt)
endif()

message(STATUS "Current CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")

# ENABLE_NO_ALIGNED_ALLOC
if(ENABLE_NO_ALIGNED_ALLOC)
    add_definitions(-DNO_ALIGNED_ALLOC)
else()
    # this is available in C11 N.B. std::aligned_alloc is available in C++17
    CHECK_FUNCTION_EXISTS(aligned_alloc HAVE_ALIGNED_ALLOC)
    if(NOT HAVE_ALIGNED_ALLOC)
        add_definitions(-DNO_ALIGNED_ALLOC)
    endif()
endif()

# ENABLE_UMPIRE
if (ENABLE_UMPIRE)
    find_package(umpire REQUIRED)
    if (NOT CUTT_USES_THIS_UMPIRE_ALLOCATOR)
        set(CUTT_USES_THIS_UMPIRE_ALLOCATOR "UM")
    endif(NOT CUTT_USES_THIS_UMPIRE_ALLOCATOR)
    message(STATUS "Will use Umpire allocator named \"${CUTT_USES_THIS_UMPIRE_ALLOCATOR}\"")
endif ()

file(GLOB CUTT_SOURCE_FILES "src/*.hip" "src/*.cpp" "include/*.h")

add_library(${PROJECT_NAME} STATIC ${CUTT_SOURCE_FILES})
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE TRUE)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES native)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(${PROJECT_NAME} hip::amdhip64)
if(ENABLE_UMPIRE)
    target_link_libraries(${PROJECT_NAME} umpire)
    target_compile_definitions(${PROJECT_NAME} PUBLIC CUTT_HAS_UMPIRE CUTT_USES_THIS_UMPIRE_ALLOCATOR=${CUTT_USES_THIS_UMPIRE_ALLOCATOR})
endif()

add_executable(${PROJECT_NAME}_bench "src/bench/bench.hip")
target_link_libraries(${PROJECT_NAME}_bench PUBLIC ${PROJECT_NAME})

add_executable(${PROJECT_NAME}_test "src/test/test.hip")
target_link_libraries(${PROJECT_NAME}_test PUBLIC ${PROJECT_NAME})

pybind11_add_module(${PROJECT_NAME}_python "src/python/${PROJECT_NAME}.hip" "src/python/${PROJECT_NAME}_module.hip")
target_include_directories(${PROJECT_NAME}_python PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(${PROJECT_NAME}_python PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_compile_definitions(${PROJECT_NAME}_python PRIVATE ${PROJECT_NAME}_EXPORTS)
target_link_libraries(${PROJECT_NAME}_python PRIVATE ${PROJECT_NAME})
set_target_properties(${PROJECT_NAME}_python PROPERTIES LIBRARY_OUTPUT_NAME "${PROJECT_NAME}")

install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION . LIBRARY DESTINATION .)
install(FILES include/hiptt.h DESTINATION include)
install(TARGETS ${PROJECT_NAME}_python ${PROJECT_NAME}_bench ${PROJECT_NAME}_test DESTINATION .)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/examples DESTINATION .)

