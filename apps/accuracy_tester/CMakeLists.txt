add_executable(accuracy_tester main.cpp)
add_dependencies(accuracy_tester eye_tracker)
target_include_directories(accuracy_tester PUBLIC ${OpenCV_INCLUDE_DIRS}/install/include/opencv4 ${CMAKE_SOURCE_DIR}/eye_tracker/include)
target_link_libraries(accuracy_tester PRIVATE eye_tracker)

install(TARGETS accuracy_tester DESTINATION bin)