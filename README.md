# eye_tracking

Gaze and eye position tracking, originally designed for the HDR-MF-S display.

## Installing dependencies

The application has been run and tested on Ubuntu 20.04. Any other version might be incompatible.

### [CUDA](https://developer.nvidia.com/cuda-toolkit)

CUDA can be installed from the NVIDIA website, or from the Ubuntu repositories:

    sudo apt install nvidia-cuda-toolkit

Tested with version 10.1.243-3. Use `nvcc --version` to check whether CUDA is installed.

### [IDS Software Suite](https://en.ids-imaging.com/ids-software-suite.html) (required for IDS uEye support)

Select your camera model, create an account, download and install the 64-bit Linux Debian packages.
You will need all of the included packages.

### OpenCV

OpenCV has to be installed with CUDA and uEye support, so it cannot be simply downloaded from Ubuntu repositories. Repository contains a `run.sh` file which installs all the necessary files. Before running it, make sure that CUDA is properly installed (`nvcc --version`) and uEye is available in `/opt/ids/ueye`.

## Building

    cd /path/to/eye_tracking
    cmake -DCMAKE_PREFIX_PATH=/path/to/libraries .
    make -j8

To install the application on the machine, run the `sudo make install` command. 

## Running

    hdrmfs_eye_tracker path_to_settings_file file|ids [path_to_file] [user_id]

settings.json file contains all the parameters used for eye-tracking and user profiles. Whenever an unknown user runs the application, their profile is created based on default values.

### Run on a pre-recorded video file
    hdrmfs_eye_tracker settings.json file /path/to/file.mp4 user_id
    
### Run on a stream from an IDS uEye camera
    hdrmfs_eye_tracker settings.json ids user_id

## Controls
- **ESC** - exits the application
- **V** - starts the video capture that will be saved to the `videos` directory.
- **S** - enabled 'slow mode' that slows down frame capture for easier analysis.
- **P** - saves current frame to the `images` directory.
- **Q** - disables window output to improve the performance.
- **W** - shows camera image used for pupil detection.
- **E** - shows camera image used for glint detection.
- **R** - shows thresholded image used for pupil detection as a video output.
- **T** - shows thresholded image used for glint detection as a video output.
- **+/-** - increases/decreases the radius of a circle used as an area of interest for pupil and glint detection.
- **Arrow keys** - moves a circle used as an area of interest for pupil and glint detection.

## Known issues/further work

- The Kálmán filter settings are very arbitrarily chosen, particularly the noise covariances. The filter would be much more useful if they were determined in some rigorous way.
- Blinks are not handled at all. Probably, a good way to detect them would be to calculate PERCLOS (percentage closure) of the eye, i.e. the percentage of the iris covered by the eyelids. We already calculate the area of the iris as part of the circle detection algorithm.
- Pupil detection is problematic for users with glasses, as a number of secondary reflections might hide a pupil.

## References

- Eye model: [E. D. Guestrin and M. Eizenman, "General theory of remote gaze estimation using the pupil center and corneal reflections," in _IEEE Transactions on Biomedical Engineering_, vol. 53, no. 6, pp. 1124-1133, June 2006, doi: 10.1109/TBME.2005.863952.](https://ieeexplore.ieee.org/document/1634506)
- Eyeball diameter: [I. Bekerman, P. Gottlieb and M. Vaiman, "Variations in eyeball diameters of the healthy adults," in _Journal of Ophthalmology_, vol. 2014, art. 503645, November 2014, doi: 10.1155/2014/503645.](https://www.hindawi.com/journals/joph/2014/503645/)
- HDR-MF-S display: [F. Zhong, A. Jindal, A. Ö. Yöntem, P. Hanji, S. J. Watt and R. K. Mantiuk, "Reproducing reality with a high-dynamic-range multi-focal stereo display," in _ACM Transactions on Graphics_, vol. 40, no. 6, art. 241, December 2021, doi: 10.1145/3478513.3480513](https://www.cl.cam.ac.uk/research/rainbow/projects/hdrmfs/Reproducing_reality_HDR_MF_S_display.pdf)
