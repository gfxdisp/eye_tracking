file(GLOB_RECURSE SOURCES src/*.cpp)

add_library(eye_tracker ${SOURCES})
target_include_directories(eye_tracker PUBLIC include)

find_package(OpenCV REQUIRED)
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found")
else ()
    target_include_directories(eye_tracker PUBLIC ${OpenCV_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS}/install/include/opencv4)
    target_link_libraries(eye_tracker PUBLIC ${OpenCV_LIBS})
endif ()

find_library(ueye_api NAMES ueye_api PATHS /opt/ids/ueye/lib)
if (NOT ueye_api)
    message(WARNING "ueye_api library not found, the IDS uEye camera module will not be available")
else ()
    message(STATUS "IDS uEye camera module found")
    target_include_directories(eye_tracker PUBLIC /opt/ids/ueye/include)
    target_link_libraries(eye_tracker PUBLIC ueye_api)
    target_compile_definitions(eye_tracker PUBLIC UEYE_INCLUDED)
endif ()

install(TARGETS eye_tracker DESTINATION lib)