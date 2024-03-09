add_executable(hdrmfs_eye_tracker main.cpp)
add_dependencies(hdrmfs_eye_tracker eye_tracker)
target_include_directories(hdrmfs_eye_tracker PUBLIC ${OpenCV_INCLUDE_DIRS}/install/include/opencv4 ${CMAKE_SOURCE_DIR}/eye_tracker/include)
target_link_libraries(hdrmfs_eye_tracker PRIVATE eye_tracker)

install(TARGETS hdrmfs_eye_tracker DESTINATION bin)