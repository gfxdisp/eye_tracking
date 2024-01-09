add_executable(data_generator main.cpp)
add_dependencies(data_generator eye_tracker)
target_include_directories(data_generator PUBLIC ${OpenCV_INCLUDE_DIRS}/install/include/opencv4 ${CMAKE_SOURCE_DIR}/eye_tracker/include)
target_link_libraries(data_generator PRIVATE eye_tracker)

install(TARGETS data_generator DESTINATION bin)