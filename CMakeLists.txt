cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(gputt LANGUAGES CXX VERSION 0.1)

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

option(GPUTT_BUILD_PYTHON "Build Python bindings" ON)
option(ENABLE_NVTOOLS "Enable nvvp profiling of CPU code" OFF)

if (GPUTT_BUILD_PYTHON)
if ("x${Python3_VERSION}" STREQUAL "x")
find_package(Python3 COMPONENTS Interpreter Development)
else()
find_package(Python3 ${Python3_VERSION} EXACT COMPONENTS Interpreter Development)
endif()
set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
message(STATUS "PYTHON_EXECUTABLE = ${PYTHON_EXECUTABLE}")

set(PYBIND11_PYTHON_VERSION ${Python3_VERSION})
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/pybind11)
endif()

file(GLOB ${PROJECT_NAME}_SOURCE_FILES "src/*.cu")
file(GLOB ${PROJECT_NAME}_bench_SOURCE_FILES "src/bench/bench.cu")
file(GLOB ${PROJECT_NAME}_test_SOURCE_FILES "src/test/test.cu")
file(GLOB ${PROJECT_NAME}_example_SOURCE_FILES "src/example/example.cu")

include(CheckLanguage)
if (CMAKE_VERSION GREATER_EQUAL 3.24)
check_language(HIP)
else()
message(WARNING "Prior to 3.24, the HIP language is not fully integrated into CMake, "
	"we have to use more ugly ways...")
set(HIP_COMPAT_ENABLED 1)
endif()
check_language(CUDA)

if(CMAKE_HIP_COMPILER OR HIP_COMPAT_ENABLED)

if (NOT HIP_COMPAT_ENABLED)
enable_language(HIP)
endif()

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

if (HIP_COMPAT_ENABLED)
set(HIP_LANGUAGE CXX)
set_source_files_properties(${${PROJECT_NAME}_SOURCE_FILES} PROPERTIES COMPILE_FLAGS "-x hip")
set_source_files_properties(${${PROJECT_NAME}_bench_SOURCE_FILES} PROPERTIES COMPILE_FLAGS "-x hip")
set_source_files_properties(${${PROJECT_NAME}_test_SOURCE_FILES} PROPERTIES COMPILE_FLAGS "-x hip")
set_source_files_properties(${${PROJECT_NAME}_example_SOURCE_FILES} PROPERTIES COMPILE_FLAGS "-x hip")
else()
set(HIP_LANGUAGE HIP)
endif()
set_source_files_properties(${${PROJECT_NAME}_SOURCE_FILES} PROPERTIES LANGUAGE ${HIP_LANGUAGE})
set_source_files_properties(${${PROJECT_NAME}_bench_SOURCE_FILES} PROPERTIES LANGUAGE ${HIP_LANGUAGE})
set_source_files_properties(${${PROJECT_NAME}_test_SOURCE_FILES} PROPERTIES LANGUAGE ${HIP_LANGUAGE})
set_source_files_properties(${${PROJECT_NAME}_example_SOURCE_FILES} PROPERTIES LANGUAGE ${HIP_LANGUAGE})
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
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/gputt>  
    $<INSTALL_INTERFACE:include/gputt>)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(hip_FOUND)
target_link_libraries(${PROJECT_NAME} hip::amdhip64)
endif()

add_executable(${PROJECT_NAME}_bench ${${PROJECT_NAME}_bench_SOURCE_FILES})
target_compile_features(${PROJECT_NAME}_bench PUBLIC cxx_std_17)
target_include_directories(${PROJECT_NAME}_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME}_bench PUBLIC ${PROJECT_NAME})

add_executable(${PROJECT_NAME}_test ${${PROJECT_NAME}_test_SOURCE_FILES})
target_compile_features(${PROJECT_NAME}_test PUBLIC cxx_std_17)
target_include_directories(${PROJECT_NAME}_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME}_test PUBLIC ${PROJECT_NAME})

add_executable(${PROJECT_NAME}_example ${${PROJECT_NAME}_example_SOURCE_FILES})
target_compile_features(${PROJECT_NAME}_example PUBLIC cxx_std_17)
target_include_directories(${PROJECT_NAME}_example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME}_example PUBLIC ${PROJECT_NAME})

if (GPUTT_BUILD_PYTHON)
pybind11_add_module(${PROJECT_NAME}_python "src/python/${PROJECT_NAME}.cu" "src/python/${PROJECT_NAME}_module.cu")
target_include_directories(${PROJECT_NAME}_python PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(${PROJECT_NAME}_python PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_compile_definitions(${PROJECT_NAME}_python PRIVATE ${PROJECT_NAME}_EXPORTS)
target_link_libraries(${PROJECT_NAME}_python PRIVATE ${PROJECT_NAME})
set_target_properties(${PROJECT_NAME}_python PROPERTIES LIBRARY_OUTPUT_NAME "${PROJECT_NAME}")
endif()

set(GPUTT_TARGETS ${PROJECT_NAME})
if (GPUTT_BUILD_PYTHON)
list(APPEND GPUTT_TARGETS ${PROJECT_NAME}_python)
endif()
install(TARGETS ${GPUTT_TARGETS} EXPORT gputt-targets ARCHIVE DESTINATION . LIBRARY DESTINATION .)
install(FILES include/gputt/gputt.h include/gputt/gputt_runtime.h DESTINATION include/gputt)

install(EXPORT gputt-targets DESTINATION lib/cmake/gputt)

include(CMakePackageConfigHelpers)

include(CMakePackageConfigHelpers)
# generate the config file that includes the exports
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/gputtConfig.cmake"
  INSTALL_DESTINATION "lib/cmake/gputt"
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/gputtConfigVersion.cmake"
  VERSION "${${PROJECT_NAME}_VERSION_MAJOR}.${${PROJECT_NAME}_VERSION_MINOR}"
  COMPATIBILITY AnyNewerVersion)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/gputtConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/gputtConfigVersion.cmake
  DESTINATION lib/cmake/gputt)
