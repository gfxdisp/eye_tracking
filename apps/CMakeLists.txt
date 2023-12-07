add_subdirectory(blender_renderer)

if (UNIX)
    add_subdirectory(hdrmfs_eye_tracker)
    add_subdirectory(polynomial_estimator)
    add_subdirectory(accuracy_tester)
endif (UNIX)
