cmake_minimum_required(VERSION 3.12)
project(eye_tracker)
set(CMAKE_CXX_STANDARD 20)
file(GLOB SOURCES *.cpp)
add_executable(eye_tracker ${SOURCES})

if(NOT CMAKE_PREFIX_PATH)
    set(CMAKE_PREFIX_PATH /usr/local)
endif()
# ccache
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

# OpenCV
set(OpenCV_DIR ${CMAKE_PREFIX_PATH}/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
list(APPEND OpenCV_INCLUDE_DIRS ${CMAKE_PREFIX_PATH}/include/opencv4)
target_include_directories(eye_tracker PUBLIC ${OpenCV_INCLUDE_DIRS})
target_include_directories(eye_tracker PUBLIC ${OpenCV_INCLUDE_DIRS}/install/include/opencv4)
target_link_libraries(eye_tracker PUBLIC debug ${OpenCV_LIBS})
