add_executable(testbed main.cpp)
add_dependencies(testbed eye_tracker)
target_include_directories(testbed PUBLIC ${OpenCV_INCLUDE_DIRS}/install/include/opencv4 ${CMAKE_SOURCE_DIR}/eye_tracker/include)
target_link_libraries(testbed PRIVATE eye_tracker)

install(TARGETS testbed DESTINATION bin)