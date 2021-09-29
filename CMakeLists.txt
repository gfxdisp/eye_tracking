cmake_minimum_required(VERSION 3.12)
project(eye_tracker)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE release)

if(NOT CMAKE_PREFIX_PATH)
    set(CMAKE_PREFIX_PATH /usr/local)
endif()

# flags
option(HEADLESS "Headless (no GUI)" OFF)
if(HEADLESS)
    message(STATUS "Building in headless mode (without GUI).")
    add_compile_definitions(HEADLESS)
else()
    message(STATUS "Building with GUI support.")
endif()

# ccache
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

# uEye
option(ENABLEUEYE "Enable IDS uEye support" ON)
if(ENABLEUEYE AND EXISTS /opt/ids/ueye)
    set(CMAKE_PREFIX_PATH /opt/ids/ueye)
    include_directories(/opt/ids/ueye/include)
    file(GLOB SOURCES *.cpp idslib/*.cpp)
    add_executable(eye_tracker ${SOURCES})
    target_link_libraries(eye_tracker ueye_api)
    add_compile_definitions(HAVE_UEYE)
else() # Don't build idslib
    message(STATUS "Building without IDS uEye support.")
    file(GLOB SOURCES *.cpp)
    add_executable(eye_tracker ${SOURCES})
endif()

# xtensor
list(APPEND CMAKE_MODULE_PATH "${CMAKE_PREFIX_PATH}/share/cmake" "${CMAKE_PREFIX_PATH}/lib/cmake")
find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtensor-blas REQUIRED)
include_directories(${xtensor_INCLUDE_DIRS} ${xtl_INCLUDE_DIRS} ${xtensor-blas_INCLUDE_DIRS})
find_package(xsimd) # SIMD extension for xtensor - optional dependency - very unlikely to make a difference
if(XSIMD_FOUND)
    include_directories(${xsimd_INCLUDE_DIRS})
    set(XTENSOR_USE_XSIMD 1)
    link_libraries(eye_tracker xtensor::use_xsimd)
endif()
# xtensor::optimize makes the build use platform-specific instructions
# (i.e. g++ -march=native). This can improve performance but makes the resulting builds non-portable.
target_link_libraries(eye_tracker xtensor xtensor::optimize)

# OpenCV
set(OpenCV_DIR ${CMAKE_PREFIX_PATH}/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(eye_tracker ${OpenCV_LIBS})

# BLAS/LAPACK
find_package(BLAS)
if(NOT BLAS_FOUND)
    set(BLAS_DIR ${CMAKE_PREFIX_PATH}/lib/cmake/openblas)
endif()
add_definitions(-DHAVE_CBLAS=1)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
target_link_libraries(eye_tracker ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
