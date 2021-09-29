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
          -DBUILD_LIST=cudaarithm,cudafilters,cudaimgproc,cudev,highgui,video,videoio \
          -DENABLE_FAST_MATH=ON \
          -DWITH_CUDA=ON -DCUDA_FAST_MATH=ON -DWITH_FFMPEG=ON \
          -DWITH_OPENGL=ON -DWITH_LAPACK=ON \
          -DINSTALL_C_EXAMPLES=OFF -DINSTALL_PYTHON_EXAMPLES=OFF \
          -DCMAKE_C_COMPILER=/usr/bin/gcc-8 -DCMAKE_CXX_COMPILER=/usr/bin/g++-8 \
          -DWITH_UEYE=ON -DUEYE_ROOT=/opt/ids/ueye ../
    # Remove WITH_UEYE if not using uEye
    make -j8
    make -j8 install

Tested with OpenCV commit c1148c4 (version 4.5.3-dev) and OpenCV-contrib commit c3de0d3.

### [xtl](https://github.com/xtensor-stack/xtl/)

    git clone --depth 1 https://github.com/xtensor-stack/xtl.git
    cd xtl
    cmake -DCMAKE_INSTALL_PREFIX=/path/to/libraries
    make install

Tested with commit 7d775ef (after release 0.7.2).

### [xtensor](https://github.com/xtensor-stack/xtensor/)

    git clone --depth 1 https://github.com/xtensor-stack/xtensor.git
    cd xtensor
    cmake -DCMAKE_INSTALL_PREFIX=/path/to/libraries
    make install

Tested with commit d5a5c63 (after release 0.23.10).

### [xtensor-blas](https://github.com/xtensor-stack/xtensor-blas)

You will also need to install OpenBLAS and LAPACK.

    sudo apt install libopenblas64-dev liblapack-dev # Unsure if these are the right libraries
    git clone --depth 1 https://github.com/xtensor-stack/xtensor-blas.git
    cd xtensor-blas
    cmake -DCMAKE_INSTALL_PREFIX=/path/to/libraries
    make install

Tested with libopenblas64-dev version 0.3.8+ds-1ubuntu0.20.04.1,
liblapack-dev version 3.9.0-1build1,
and xtensor-blas commit 7ceb791 (after release 0.19.1).

### [xsimd](https://github.com/xtensor-stack/xsimd) (optional)

    git clone --depth 1 -b 7.x https://github.com/xtensor-stack/xsimd.git
    cd xsimd
    cmake -DCMAKE_INSTALL_PREFIX=/path/to/libraries
    make install

Tested with release 7.6.0.

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

## References

- Eye model: [E. D. Guestrin and M. Eizenman, "General theory of remote gaze estimation using the pupil center and corneal reflections," in _IEEE Transactions on Biomedical Engineering_, vol. 53, no. 6, pp. 1124-1133, June 2006, doi: 10.1109/TBME.2005.863952.](https://ieeexplore.ieee.org/document/1634506)
- Eyeball diameter: [I. Bekerman, P. Gottlieb and M. Vaiman, "Variations in eyeball diameters of the healthy adults," in _Journal of Ophthalmology_, vol. 2014, art. 503645, November 2014, doi: 10.1155/2014/503645.](https://www.hindawi.com/journals/joph/2014/503645/)
- HDR-MF-S display: [F. Zhong, A. Jindal, A. Ö. Yöntem, P. Hanji, S. J. Watt and R. K. Mantiuk, "Reproducing reality with a high-dynamic-range multi-focal stereo display," in _ACM Transactions on Graphics_, vol. 40, no. 6, art. 241, December 2021, doi: 10.1145/3478513.3480513](https://www.cl.cam.ac.uk/research/rainbow/projects/hdrmfs/Reproducing_reality_HDR_MF_S_display.pdf)
