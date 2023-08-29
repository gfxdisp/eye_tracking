cmake_minimum_required(VERSION 3.12)
project(hdrmfs_eye_tracker)
set(CMAKE_CXX_STANDARD 20)
file(GLOB SOURCES src/*.cpp)
add_executable(hdrmfs_eye_tracker ${SOURCES})
add_executable(estimator simul_src/estimate.cpp)
add_executable(meta_model simul_src/meta_model.cpp src/PolynomialFit.cpp src/Utils.cpp)


# If linux
if(UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

    if(NOT CMAKE_PREFIX_PATH)
        set(CMAKE_PREFIX_PATH /usr/local)
    endif()
    # ccache
    find_program(CCACHE_FOUND ccache)
    if(CCACHE_FOUND)
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
        set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
    endif(CCACHE_FOUND)

    install(TARGETS hdrmfs_eye_tracker
            CONFIGURATIONS Release
            RUNTIME DESTINATION bin)

    # OpenCV
    set(OpenCV_DIR ${CMAKE_PREFIX_PATH}/lib/cmake/opencv4)
    find_package(OpenCV REQUIRED)
    list(APPEND OpenCV_INCLUDE_DIRS ${CMAKE_PREFIX_PATH}/include/opencv4)
    target_include_directories(hdrmfs_eye_tracker PUBLIC ${OpenCV_INCLUDE_DIRS})

    # uEye
    link_directories(/opt/ids/ueye/lib/)

    message("Included: ${OpenCV_INCLUDE_DIRS}")
    message("Libs: ${OpenCV_LIBS}")
    target_include_directories(hdrmfs_eye_tracker PUBLIC ${OpenCV_INCLUDE_DIRS}/install/include/opencv4 /opt/ids/ueye/include)
    target_link_libraries(hdrmfs_eye_tracker PUBLIC ${OpenCV_LIBS} ueye_api)

    target_include_directories(meta_model PUBLIC ${OpenCV_INCLUDE_DIRS}/install/include/opencv4 /opt/ids/ueye/include)
    target_link_libraries(meta_model PUBLIC ${OpenCV_LIBS} ueye_api)
endif ()