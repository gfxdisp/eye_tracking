#!/usr/bin/env bash

resources_path="resources/"
calibration_path="examples/calibration.mp4"
video_path="examples/gaze_4.mp4"
user="default"
hdrmfs_eye_tracker -s $resources_path -u $user -v $video_path -c $calibration_path
