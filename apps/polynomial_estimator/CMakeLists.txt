add_executable(polynomial_estimator main.cpp)
add_dependencies(polynomial_estimator eye_tracker)
target_include_directories(polynomial_estimator PUBLIC ${OpenCV_INCLUDE_DIRS}/install/include/opencv4 ${CMAKE_SOURCE_DIR}/eye_tracker/include)
target_link_libraries(polynomial_estimator PRIVATE eye_tracker)

install(TARGETS polynomial_estimator DESTINATION bin)