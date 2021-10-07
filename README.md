# eye_tracking

Gaze and eye position tracking, originally designed for the HDR-MF-S display.

## Installing dependencies

### [CUDA](https://developer.nvidia.com/cuda-toolkit)

CUDA can be installed from the NVIDIA website, or from the Ubuntu repositories:

    sudo apt install nvidia-cuda-toolkit

Tested with version 10.1.243-3. Use `nvcc --version` to check whether CUDA is installed.

### [IDS Software Suite](https://en.ids-imaging.com/ids-software-suite.html) (required for IDS uEye support)

Select your camera model, create an account, download and install the 64-bit Linux Debian packages.
You will need all of the included packages, as well as `libomp5` (OpenMP) from the Ubuntu repositories.

### FFmpeg

Required to load videos in OpenCV.

TODO: which exact package is needed?

### OpenCV

OpenCV can be found in the Ubuntu repositories, but it seems to be built without CUDA support.

    git clone --depth=1 https://github.com/opencv/opencv.git
    git clone --depth=1 https://github.com/opencv/opencv_contrib.git
    cd opencv
    mkdir build
    cd build
    sudo apt install gcc-8 g++-8 # Required on newer Ubuntu versions
    # Minimal build of OpenCV - includes only the required modules
    cmake -DCMAKE_INSTALL_PREFIX=/path/to/libraries \
          -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
          -DBUILD_LIST=cudaarithm,cudafilters,cudaimgproc,cudev,imgcodecs,highgui,video,videoio \
          -DENABLE_FAST_MATH=ON -DWITH_CUDA=ON -DCUDA_FAST_MATH=ON -DWITH_FFMPEG=ON -DWITH_LAPACK=ON \
          -DINSTALL_C_EXAMPLES=OFF -DINSTALL_PYTHON_EXAMPLES=OFF \
          -DCMAKE_C_COMPILER=/usr/bin/gcc-8 -DCMAKE_CXX_COMPILER=/usr/bin/g++-8 \
          -DWITH_UEYE=ON -DUEYE_ROOT=/opt/ids/ueye ../
    # Remove WITH_UEYE if not using uEye
    make -j8
    make -j8 install

Tested with OpenCV commit c1148c4 (version 4.5.3-dev) and OpenCV-contrib commit c3de0d3.

## Building

    cd /path/to/eye_tracking
    cmake -DCMAKE_PREFIX_PATH=/path/to/libraries -DHEADLESS={ON|OFF} .
    make -j8

The option `HEADLESS` specifies whether to display a GUI (if it is `OFF`),
or to print results to standard output (if it is `ON`)

## Running

    # Run on a pre-recorded video file
    ./eye_tracker file /path/to/file.mp4
    # Run on a stream from an IDS uEye camera, if only one is connected
    ./eye_tracker ueye
    # Run on a stream from a specific uEye camera
    ./eye_tracker ueye 2
    # Run on a stream from a specific uEye camera, using a specific GPU for CUDA
    ./eye_tracker ueye 2 1

## Known issues/further work

- The Kálmán filter settings are very arbitrarily chosen, particularly the noise covariances. The filter would be much more useful if they were determined in some rigorous way.
- Blinks are not handled at all. Probably, a good way to detect them would be to calculate PERCLOS (percentage closure) of the eye, i.e. the percentage of the iris covered by the eyelids. We already calculate the area of the iris as part of the circle detection algorithm.
- Currently, the algorithm can run at 60 FPS. If you ever need to increase performance, you can try any of the following optimisations:
    - make the ROI smaller
    - apply the ROI to the camera (i.e. only read out the ROI from the image sensor)
    - use a mask for CUDA operations to reduce the area processed even further (e.g. use a rectangular ROI and then a circular mask corresponding to the eye hole in the VR headset)
    - (when showing the GUI) to eliminate a GPU->CPU download:
        - build OpenCV with OpenGL support (-DWITH_OPENGL=ON)
        - call cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE | cv::WINDOW_OPENGL) before the main loop
        - you will now be able to call cv::imshow on a cv::cuda::GpuMat directly, without downloading
        - to draw the overlay, you will need to use direct OpenGL calls

## References

- Eye model: [E. D. Guestrin and M. Eizenman, "General theory of remote gaze estimation using the pupil center and corneal reflections," in _IEEE Transactions on Biomedical Engineering_, vol. 53, no. 6, pp. 1124-1133, June 2006, doi: 10.1109/TBME.2005.863952.](https://ieeexplore.ieee.org/document/1634506)
- Eyeball diameter: [I. Bekerman, P. Gottlieb and M. Vaiman, "Variations in eyeball diameters of the healthy adults," in _Journal of Ophthalmology_, vol. 2014, art. 503645, November 2014, doi: 10.1155/2014/503645.](https://www.hindawi.com/journals/joph/2014/503645/)
- HDR-MF-S display: [F. Zhong, A. Jindal, A. Ö. Yöntem, P. Hanji, S. J. Watt and R. K. Mantiuk, "Reproducing reality with a high-dynamic-range multi-focal stereo display," in _ACM Transactions on Graphics_, vol. 40, no. 6, art. 241, December 2021, doi: 10.1145/3478513.3480513](https://www.cl.cam.ac.uk/research/rainbow/projects/hdrmfs/Reproducing_reality_HDR_MF_S_display.pdf)
