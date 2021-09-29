# eye_tracking

Gaze and eye position tracking, originally designed for the HDR-MF-S display.

## Dependencies

- CUDA
- OpenCV
- xtl
- xtensor
- xtensor-blas
- IDS Software Suite (required for uEye camera support)
- xsimd (optional)

TODO: link to dependencies; how to build and install dependencies (particularly OpenCV); dependency versions

## Configuring

    cmake -DXTENSOR_PREFIX=/path/to/xtensor/installation/ -DHEADLESS={ON|OFF} .

TODO: explain the options

## Building

    make -j8

## Running

    # Run on a pre-recorded video file
    ./eye_tracker file /path/to/file.mp4
    # Run on a stream from an IDS uEye camera, if only one is connected
    ./eye_tracker ueye
    # Run on a stream from a specific uEye camera
    ./eye_tracker ueye 2
    # Run on a stream from a specific uEye camera, using a specific GPU for CUDA
    ./eye_tracker ueye 2 1

## References

TODO: cite Guestrin & Eizenman, the source for the eye diameter, the HDR-MF-S paper?
