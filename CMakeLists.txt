cmake_minimum_required(VERSION 3.12)
project(eye_tracker)

if (UNIX)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
    set(CMAKE_PREFIX_PATH /usr/local)

    # OpenCV
    set(OpenCV_DIR ${CMAKE_PREFIX_PATH}/lib/cmake/opencv4)
    find_package(OpenCV REQUIRED)
    list(APPEND OpenCV_INCLUDE_DIRS ${CMAKE_PREFIX_PATH}/include/opencv4)

    # uEye
    link_directories(/opt/ids/ueye/lib/)

    message("Included: ${OpenCV_INCLUDE_DIRS}")
    message("Libs: ${OpenCV_LIBS}")

    add_subdirectory(eye_tracker)
endif ()

add_subdirectory(apps)