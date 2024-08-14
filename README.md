# Tracking eye position and gaze direction in near-eye volumetric displays

This repository contains the source code for the eye-tracking application used in the paper 
"Tracking eye position and gaze direction in near-eye volumetric displays" (DOI: [XX.XXX/XX.XXX](https://doi.org/XX.XXX/XX.XXX)). 
The application uses an IDS uEye camera to track the user's eye and gaze direction, and displays the results on the screen.
In the case of missing camera, the application can be run on a pre-recorded video file.

## Installing dependencies

The application has been run and tested on Ubuntu 20.04. Any other version might be incompatible.

### CUDA

CUDA should be installed according to the [official instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html). 
Application was tested with version 11.7. Use `nvcc --version` to check whether CUDA is installed.

### IDS uEye camera (optional)

Select your camera model, create an account, and download the [IDS Software Suite](https://en.ids-imaging.com/ids-software-suite.html) (archive file). In order to install it in the `/opt/ids/`
directory, run the following commands:
```bash
tar -zxvf ids-software-suite-linux-64-4.96.1-archive.tgz
cd /opt/
sudo /path/to/extracted/folder/ueye_4.96.1.2054_amd64.run
```


### OpenCV

OpenCV has to be installed with CUDA support. Repository contains a `get_opencv.sh` file which installs all the necessary files. Before running it, make sure that CUDA is properly installed (`nvcc --version`).

## Building

Run `install.sh` script to build the application. It will create a `build` directory, compile the source code, and copy the `hdrmfs_eye_tracker` executable to the `/usr/local/bin/` directory.

## Running

Too run the application, you can use the `run.sh` script: it will start the application with the default settings,
fine-tune the calibration parameters using the provided video and csv files, and present the tracking from the specified video on the screen.

The application can be manually run with the following parameters:
```
- -s, --settings-path [path] - path to the folder with all resource file, specifically the `settings.json` which contains all the necessary parameters.
- -u, --user [user_id] - user ID, used to record the fine-tuned calibration parameters for the specified user.
- -v, --video [path] - path to the video file that will be used to demonstrate the tracking. If not provided, the application will use the IDS camera stream.
- -c, --calibration [path] - path to the video file containing the recording of the calibration process. If not provided, the application will use already available calibration parameters.
- -h, --headless - run the application without the window output.
```

## Controls
- **ESC** - exits the application
- **V** - starts the video capture that will be saved to the `videos` directory.
- **P** - saves current frame to the `images` directory.
- **Q** - disables window output to improve the performance.
- **W** - shows camera image used for pupil detection.
- **E** - shows thresholded image used for pupil detection as a video output.
- **R** - shows thresholded image used for glint detection as a video output.
