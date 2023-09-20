# Running script:
# blender --background --python main.py -- -e left -n 1000

import importlib.util
import pip
import os, sys
def check_modules():
    any_install = False
    needed_modules = ["bpy", "argparse", "scipy", "pandas", "json", "contextlib", "io", "time", "numpy"]
    install_names = ["bpy", "argparse", "scipy", "pandas", "json", "contextlib", "io", "time", "numpy"]
    for i, module in enumerate(needed_modules):
        if importlib.util.find_spec(module) is None:
            print(f"Module {module} not loaded, installing...")
            pip.main(["install", install_names[i]])
            any_install = True
    return not any_install

def main():
    import bpy
    from argparse import ArgumentParser
    from scipy.io import loadmat
    import json

    script_path = os.path.dirname(bpy.data.filepath)
    sys.path.append(script_path)

    from setup_parameters import SetupParameters
    import renderer
    import numpy as np

    # create a parser ignoring everything before the first '--' in sys.argv
    argv = sys.argv
    script_name = argv[argv.index("--") - 1]
    needed_args = [[script_name], argv[argv.index("--") + 1:]]
    sys.argv = [item for sublist in needed_args for item in sublist]
    parser = ArgumentParser()
    parser.add_argument("-e", "--eye", dest="eye", default="", type=str, required=True,
                        help="eye to calibrate, either 'left' or 'right'")
    parser.add_argument("-s", "--setup", dest="setup", type=str, required=True)
    parser.add_argument("-n", "--num-images", dest="num_images", default=1000, type=int, required=False)
    parser.add_argument("-i", "--index", dest="index", type=int, required=True)
    parser.add_argument("-o", "--output", dest="output", default=script_path, type=str, required=False)
    parser.add_argument("-c", "--csv", dest="csv", default="", type=str, required=False)
    parser.add_argument("-r", "--random-hardware", dest="random_hardware", action="store_true", required=False)


    args = parser.parse_args()
    eye = args.eye
    num_images = args.num_images
    output_path = args.output
    setup_idx = args.index
    setup_path = args.setup
    csv_path = args.csv
    random_hardware = args.random_hardware

    hdrmfs_camera_calib_path = os.path.join(setup_path, "hdrmfs_camera_calib.mat")
    capture_params_path = os.path.join(setup_path, "capture_params.json")

    hdrmfs_camera_calib = loadmat(hdrmfs_camera_calib_path)

    with open(capture_params_path) as capture_params_file:
        capture_params = json.load(capture_params_file)

    setup_parameters = SetupParameters(hdrmfs_camera_calib, capture_params, eye)
    setup_parameters.randomize_params(random_hardware=random_hardware)

    setup_images_path = os.path.join(output_path, f"setups_{eye}", f"{setup_idx:05d}")

    if not os.path.exists(setup_images_path):
        os.makedirs(setup_images_path)

    if not os.path.exists(os.path.join(setup_images_path, "setup_params.csv")):
        with open(os.path.join(setup_images_path, "setup_params.csv"), "a") as f:
            f.write("idx")
            f.write(",cornea_centre_distance,cornea_curvature_radius,cornea_refraction_index,alpha,beta\n")
            f.write(f"{setup_idx}")
            f.write(f",{setup_parameters.cornea_centre_distance * 1000},{setup_parameters.cornea_curvature_radius * 1000},{setup_parameters.cornea_refraction_index},{setup_parameters.alpha},{setup_parameters.beta}\n")
    else:
        with open(os.path.join(setup_images_path, "setup_params.csv"), "r") as f:
            last_line = f.readlines()[-1]
            values = last_line.split(",")
            setup_parameters.cornea_centre_distance = float(values[1]) / 1000
            setup_parameters.cornea_curvature_radius = float(values[2]) / 1000
            setup_parameters.cornea_refraction_index = float(values[3])
            setup_parameters.alpha = float(values[4])
            setup_parameters.beta = float(values[5])
            print(f"Loaded parameters from {os.path.join(setup_images_path, 'setup_params.csv')}:")
            print(f"Cornea centre distance: {setup_parameters.cornea_centre_distance}")
            print(f"Cornea curvature radius: {setup_parameters.cornea_curvature_radius}")
            print(f"Cornea refraction index: {setup_parameters.cornea_refraction_index}")
            print(f"Alpha: {setup_parameters.alpha}")
            print(f"Beta: {setup_parameters.beta}")

    renderer.render_images(setup_parameters, num_images, setup_images_path, csv_path)


if __name__ == "__main__":
    if check_modules():
        main()
    else:
        print("Relaunch the script to apply the changes in the modules.")