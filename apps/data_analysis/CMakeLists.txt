add_executable(data_analysis main.cpp)
add_dependencies(data_analysis eye_tracker)
target_include_directories(data_analysis PUBLIC ${OpenCV_INCLUDE_DIRS}/install/include/opencv4 ${CMAKE_SOURCE_DIR}/eye_tracker/include)
target_link_libraries(data_analysis PRIVATE eye_tracker)

install(TARGETS data_analysis DESTINATION bin)