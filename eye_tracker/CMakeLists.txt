file(GLOB_RECURSE SOURCES src/*.cpp)
add_library(eye_tracker ${SOURCES})

target_include_directories(eye_tracker PUBLIC ${OpenCV_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS}/install/include/opencv4)
target_include_directories(eye_tracker PUBLIC /opt/ids/ueye/include)
target_include_directories(eye_tracker PUBLIC include)
target_link_libraries(eye_tracker PUBLIC ${OpenCV_LIBS} ueye_api)

install(TARGETS eye_tracker DESTINATION lib)