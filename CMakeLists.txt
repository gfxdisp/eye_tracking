cmake_minimum_required(VERSION 3.12)
project(eye_tracker)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

add_subdirectory(eye_tracker)
add_subdirectory(apps)