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
    parser.add_argument("-n", "--num-images", dest="num_images", default=1000, type=int, required=False)
    parser.add_argument("-i", "--index", dest="index", type=int, required=True)
    parser.add_argument("-o", "--output", dest="output", default=script_path, type=str, required=False)

    args = parser.parse_args()
    eye = args.eye
    num_images = args.num_images
    output_path = args.output
    setup_idx = args.index

    hdrmfs_camera_calib_path = os.path.join(script_path, "hdrmfs_camera_calib.mat")
    capture_params_path = os.path.join(script_path, "capture_params.json")

    hdrmfs_camera_calib = loadmat(hdrmfs_camera_calib_path)

    with open(capture_params_path) as capture_params_file:
        capture_params = json.load(capture_params_file)

    setup_parameters = SetupParameters(hdrmfs_camera_calib, capture_params, eye)
    setup_parameters.randomize_params()

    setup_images_path = os.path.join(output_path, f"setups_{eye}", f"{setup_idx:05d}")

    if not os.path.exists(setup_images_path):
        os.makedirs(setup_images_path)

    if not os.path.exists(os.path.join(setup_images_path, "setup_params.csv")):
        with open(os.path.join(setup_images_path, "setup_params.csv"), "a") as f:
            f.write("idx")
            for i in range(setup_parameters.leds_world_pos.shape[0]):
                f.write(f",led_{i}_x,led_{i}_y,led_{i}_z")
            for i in range(4):
                for j in range(4):
                    f.write(f",M_extr_{i}{j}")
            for i in range(4):
                for j in range(4):
                    f.write(f",M_disp_rsb_{i}{j}")
            for i in range(3):
                for j in range(3):
                    f.write(f",M_intr_{i}{j}")
            f.write(",eyeball_to_cornea_dist,cornea_radius,alpha,beta\n")
            f.write(f"{setup_idx}")
            for i in range(setup_parameters.leds_world_pos.shape[0]):
                f.write(f",{setup_parameters.leds_world_pos[i, 0]},{setup_parameters.leds_world_pos[i, 1]},{setup_parameters.leds_world_pos[i, 2]}")
            for i in range(4):
                for j in range(4):
                    f.write(f",{setup_parameters.M_extr[i, j]}")
            for i in range(4):
                for j in range(4):
                    f.write(f",{setup_parameters.M_disp_rsb[i, j]}")
            for i in range(3):
                for j in range(3):
                    f.write(f",{setup_parameters.M_intr[i, j]}")
            f.write(f",{setup_parameters.eyeball_to_cornea_dist},{setup_parameters.cornea_radius},{setup_parameters.alpha},{setup_parameters.beta}\n")
    else:
        with open(os.path.join(setup_images_path, "setup_params.csv"), "r") as f:
            last_line = f.readlines()[-1]
            values = last_line.split(",")
            setup_parameters.leds_world_pos = np.zeros((setup_parameters.leds_world_pos.shape[0], 3))
            for i in range(setup_parameters.leds_world_pos.shape[0]):
                setup_parameters.leds_world_pos[i, :] = np.array([float(values[1 + 3 * i]), float(values[2 + 3 * i]), float(values[3 + 3 * i])])
            for i in range(4):
                for j in range(4):
                    setup_parameters.M_extr[i, j] = float(values[1 + 3 * setup_parameters.leds_world_pos.shape[0] + i * 4 + j])
            for i in range(4):
                for j in range(4):
                    setup_parameters.M_disp_rsb[i, j] = float(values[1 + 3 * setup_parameters.leds_world_pos.shape[0] + 16 + i * 4 + j])
            for i in range(3):
                for j in range(3):
                    setup_parameters.M_intr[i, j] = float(values[1 + 3 * setup_parameters.leds_world_pos.shape[0] + 32 + i * 3 + j])
            setup_parameters.eyeball_to_cornea_dist = float(values[1 + 3 * setup_parameters.leds_world_pos.shape[0] + 41])
            setup_parameters.cornea_radius = float(values[1 + 3 * setup_parameters.leds_world_pos.shape[0] + 42])
            setup_parameters.alpha = float(values[1 + 3 * setup_parameters.leds_world_pos.shape[0] + 43])
            setup_parameters.beta = float(values[1 + 3 * setup_parameters.leds_world_pos.shape[0] + 44])

    renderer.render_images(setup_parameters, num_images, setup_images_path)


if __name__ == "__main__":
    if check_modules():
        main()
    else:
        print("Relaunch the script to apply the changes in the modules.")