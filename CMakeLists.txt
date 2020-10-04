cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(cuTT)

# enable CXX
enable_language(CXX)
if (NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 11)
endif (NOT CMAKE_CXX_STANDARD)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_CXX_EXTENSIONS)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif (NOT CMAKE_CXX_EXTENSIONS)

# enable cuda
enable_language(CUDA)

if (NOT CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
endif (NOT CMAKE_CUDA_STANDARD)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if (NOT CMAKE_CUDA_EXTENSIONS)
    set(CMAKE_CUDA_EXTENSIONS OFF)
endif (NOT CMAKE_CUDA_EXTENSIONS)

message(STATUS "CUDA Include Dirs: " ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
message(STATUS "CUDA Host Compiler: " ${CMAKE_CUDA_HOST_COMPILER})

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

option(ENABLE_NVTOOLS "Enable nvvp profiling of CPU code" OFF)
option(ENABLE_NO_ALIGNED_ALLOC "Enable aligned_alloc() function implemented in cuTT" OFF)
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

file(GLOB CUTT_SOURCE_FILES "src/*.cu" "src/*.cpp" "include/*.h")

add_library(cutt ${CUTT_SOURCE_FILES})
target_include_directories(cutt PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
if(ENABLE_UMPIRE)
    target_link_libraries(cutt umpire)
    target_compile_definitions(cutt PUBLIC CUTT_HAS_UMPIRE -DCUTT_USES_THIS_UMPIRE_ALLOCATOR=${CUTT_USES_THIS_UMPIRE_ALLOCATOR})
endif()

# install library and include
install(TARGETS cutt
        ARCHIVE
            DESTINATION lib
        LIBRARY
            DESTINATION lib
        )
install(FILES cutt.h
        DESTINATION include
        )

add_executable(cutt_bench "src/cutt_bench.cpp")
target_link_libraries(cutt_bench PUBLIC cutt)

add_executable(cutt_test "src/cutt_test.cpp")
target_link_libraries(cutt_test PUBLIC cutt)
