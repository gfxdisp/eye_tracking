import random

import numpy as np
import bpy
import io
import os
import sys
from contextlib import redirect_stdout, contextmanager
import time
from mathutils import Euler
from scipy.optimize import minimize
from mathutils import Vector, Matrix


@contextmanager
def stdout_redirected(to=os.devnull):
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()
        os.dup2(to.fileno(), fd)
        sys.stdout = os.fdopen(fd, 'w')

    old_stdout = os.fdopen(os.dup(fd), 'w')
    with open(to, 'w') as file:
        _redirect_stdout(to=file)
    try:
        yield
    finally:
        _redirect_stdout(to=old_stdout)
        os.close(old_stdout.fileno())


def get_rot_x(angle):
    return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])


def get_rot_y(angle):
    return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])


def get_rot_z(angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])


def axis_and_angle_to_rot_mat(axis, angle):
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3, 3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    return R


def point_to_line_dist(line_point, point, line_direction):
    V = line_point - point
    projection = np.dot(V, line_direction) / np.linalg.norm(line_direction) ** 2 * line_direction
    distance = np.linalg.norm(V - projection)
    return distance


def optical_to_visual(optical_axis, alpha, beta):
    forward = optical_axis / np.linalg.norm(optical_axis)
    theta = np.arctan2(optical_axis[0], optical_axis[2]) - np.pi
    phi = -(np.arccos(optical_axis[1]) - np.pi / 2)

    R = get_rot_y(theta) @ get_rot_x(phi)
    right = R @ np.array([1, 0, 0])
    up = R @ np.array([0, 1, 0])

    R = axis_and_angle_to_rot_mat(up, np.deg2rad(-alpha))
    forward = R @ forward
    right = R @ right

    R = axis_and_angle_to_rot_mat(right, np.deg2rad(-beta))
    visual_axis = R @ forward

    return visual_axis


def visual_to_optical(visual_axis, alpha, beta):
    def calculate_error(angles, visual_axis, alpha, beta):
        phi = angles[0]
        theta = angles[1]
        R = get_rot_y(theta) @ get_rot_x(phi)
        calc_optical_axis = R @ np.array([0, 0, -1])
        calc_visual_axis = optical_to_visual(calc_optical_axis, alpha, beta)
        calc_visual_axis = calc_visual_axis / np.linalg.norm(calc_visual_axis)
        error = np.sum(np.abs(calc_visual_axis - visual_axis))
        return error

    angles = minimize(calculate_error, visual_axis, args=(visual_axis, alpha, beta), method='Nelder-Mead').x
    phi = angles[0]
    theta = angles[1]
    R = get_rot_y(theta) @ get_rot_x(phi)
    optical_axis = R @ np.array([0, 0, -1])
    return optical_axis


def find_optical_axis(angles, eye_pos, focus_point, alpha, beta, eyeball_to_cornea_dist):
    phi = angles[0]
    theta = angles[1]
    R = get_rot_y(theta) @ get_rot_x(phi)
    optical_axis = R @ np.array([0, 0, -1])
    optical_axis = optical_axis / np.linalg.norm(optical_axis)

    visual_axis = optical_to_visual(optical_axis, alpha, beta)
    cornea_pos = eye_pos + eyeball_to_cornea_dist * optical_axis

    distance = point_to_line_dist(focus_point, cornea_pos, visual_axis)
    return distance


def look_at(focus_point, alpha, beta, eyeball_to_cornea_dist):
    eye = bpy.data.objects.get("Eye")
    eye_world_pos = np.array(eye.matrix_world.translation)

    test_axis = focus_point - eye_world_pos
    test_axis /= np.linalg.norm(test_axis)
    theta = np.arctan2(test_axis[0], test_axis[2]) - np.pi
    phi = -(np.arccos(test_axis[1]) - np.pi / 2)

    x0 = np.array([phi, theta])

    angles = minimize(find_optical_axis, x0,
                      args=(eye_world_pos, focus_point, alpha, beta, eyeball_to_cornea_dist)).x

    phi = angles[0]
    theta = angles[1]

    eye.rotation_mode = 'XYZ'
    eye.rotation_euler = (phi, theta, 0)


def render_images(setup_params, images_num, folder_path, csv_path):
    bpy.context.scene.render.image_settings.file_format = 'JPEG'
    bpy.context.scene.render.image_settings.color_mode = 'BW'
    bpy.context.scene.eevee.taa_render_samples = 1
    bpy.context.scene.eevee.taa_samples = 1
    bpy.context.scene.eevee.sss_samples = 1
    bpy.context.scene.eevee.volumetric_samples = 1
    bpy.context.scene.eevee.use_gtao = False
    bpy.data.images["iris.jpg.004"].colorspace_settings.name = 'Raw'
    bpy.data.objects["Global light"].hide_render = True

    using_file = csv_path != ""
    eye_centre_poses = []
    focus_poses = []
    if using_file:
        with open(csv_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                values = line.split(",")
                eye_centre_pos = np.array([float(values[1]), float(values[2]), float(values[3])]) / 1000
                focus_pos = np.array([float(values[4]), float(values[5]), float(values[6])]) / 1000
                eye_centre_poses.append(eye_centre_pos)
                focus_poses.append(focus_pos)

    if not os.path.exists(os.path.join(folder_path, "images")):
        os.makedirs(os.path.join(folder_path, "images"))

    if using_file:
        start_idx = 0
        end_idx = len(eye_centre_poses)
    else:
        images = [f for f in os.listdir(os.path.join(folder_path, "images")) if
                  os.path.isfile(os.path.join(folder_path, "images", f)) and (f.endswith(".jpg") or f.endswith(".png"))]
        if len(images) > 0:
            last_image_idx = np.max([int(f.split("_")[1]) for f in images])
            start_idx = last_image_idx + 1
            end_idx = start_idx + images_num
        else:
            start_idx = 0
            end_idx = images_num

    leds_poses = setup_params.leds_world_pos / 1000
    for i in range(leds_poses.shape[0]):
        led = bpy.data.objects[f"Light.{i:03d}"]
        led.location = leds_poses[i, :]

    eye = bpy.data.objects['Eye']
    eye.rotation_mode = 'XYZ'
    eye.rotation_euler = (0, 0, 0)

    cornea = bpy.data.objects['Cornea']
    cornea.rotation_mode = 'XYZ'
    cornea.rotation_euler = (0, 0, 0)
    cornea.location = (0, 0, -setup_params.cornea_centre_distance)

    bpy.data.materials['Cornea'].node_tree.nodes['Glass BSDF'].inputs[
        2].default_value = setup_params.cornea_refraction_index

    bpy.data.materials['Cornea'].roughness = 0.0

    camera = bpy.data.objects['Camera']
    camera.matrix_world = setup_params.M_extr
    bpy.context.view_layer.update()

    camera.data.lens = setup_params.M_intr[0, 0] * setup_params.sx / setup_params.W
    camera.data.shift_x = (setup_params.W / 2 - setup_params.M_intr[0, 2]) / max(setup_params.W, setup_params.H)
    camera.data.shift_y = (setup_params.M_intr[1, 2] - setup_params.H / 2) / max(setup_params.W, setup_params.H)
    bpy.context.scene.render.use_border = True
    min_x = setup_params.capture_offset[0] / setup_params.W
    max_x = (setup_params.capture_offset[0] + setup_params.roi[0]) / setup_params.W
    min_y = (setup_params.H - setup_params.roi[1] - setup_params.capture_offset[1]) / setup_params.H
    max_y = (setup_params.H - setup_params.capture_offset[1]) / setup_params.H
    bpy.context.scene.render.border_min_x = min_x
    bpy.context.scene.render.border_max_x = max_x
    bpy.context.scene.render.border_min_y = min_y
    bpy.context.scene.render.border_max_y = max_y

    projection_matrix = camera.calc_matrix_camera(bpy.data.scenes["Scene"].view_layers["View Layer"].depsgraph,
                                                  x=setup_params.W, y=setup_params.H,
                                                  scale_x=bpy.context.scene.render.pixel_aspect_x,
                                                  scale_y=bpy.context.scene.render.pixel_aspect_y)

    mean_led_pos = np.mean(leds_poses, axis=0)
    min_eye_pos = mean_led_pos + np.array([0, 0, 0.055]) - np.array([0.015, 0.015, 0.025])
    max_eye_pos = mean_led_pos + np.array([0, 0, 0.055]) + np.array([0.015, 0.015, 0.025])

    d_res = [2160, 3840]
    display_corners = np.array([[1, d_res[0], 0, 1], [d_res[1], d_res[0], 0, 1], [d_res[1], 1, 0, 1], [1, 1, 0, 1]])
    min_screen_x = 100000
    max_screen_x = -100000
    min_screen_y = 100000
    max_screen_y = -100000
    min_screen_z = 100000
    max_screen_z = -100000
    for i in range(4):
        world_corners = (display_corners @ setup_params.M_disp_rsb[i]) / 1000

        min_screen_x = min(np.min(world_corners[:, 0]), min_screen_x)
        max_screen_x = max(np.max(world_corners[:, 0]), max_screen_x)
        min_screen_y = min(np.min(world_corners[:, 1]), min_screen_y)
        max_screen_y = max(np.max(world_corners[:, 1]), max_screen_y)
        min_screen_z = min(np.min(world_corners[:, 2]), min_screen_z)
        max_screen_z = max(np.max(world_corners[:, 2]), max_screen_z)

    start = time.time()
    for i in range(start_idx, end_idx):
        if using_file:
            eye.matrix_world.translation = eye_centre_poses[i]
        else:
            eye.matrix_world.translation = np.array(
                [random.uniform(min_eye_pos[0], max_eye_pos[0]), random.uniform(min_eye_pos[1], max_eye_pos[1]),
                 random.uniform(min_eye_pos[2], max_eye_pos[2])])
        bpy.context.view_layer.update()

        if using_file:
            focus_point = focus_poses[i]
        else:
            focus_point = np.array(
                [random.uniform(min_screen_x, max_screen_x), random.uniform(min_screen_y, max_screen_y),
                 random.uniform(min_screen_z, max_screen_z)])

        look_at(focus_point, setup_params.alpha, setup_params.beta, setup_params.cornea_centre_distance)
        bpy.context.view_layer.update()

        eye_centre_pos = eye.matrix_world.translation
        cornea_pos = cornea.matrix_world.translation

        print(eye_centre_pos, cornea_pos)

        for j in range(leds_poses.shape[0]):
            bpy.data.objects[f"Light.{j:03d}"].hide_render = True
        bpy.data.materials['Cornea'].use_screen_refraction = True
        bpy.data.materials['Cornea'].roughness = 0.0

        bpy.context.scene.render.filepath = os.path.join(folder_path, "images", f"image_{i:010d}_lights_off.jpg")
        with stdout_redirected(os.devnull):
            bpy.ops.render.render(write_still=True, use_viewport=True)

        for j in range(leds_poses.shape[0]):
            bpy.data.objects[f"Light.{j:03d}"].hide_render = False
        bpy.data.objects["Global light"].hide_render = True
        bpy.data.materials['Cornea'].use_screen_refraction = False
        bpy.data.materials['Cornea'].roughness = 0.0

        bpy.context.scene.render.filepath = os.path.join(folder_path, "images", f"image_{i:010d}_lights_on.jpg")
        with stdout_redirected(os.devnull):
            bpy.ops.render.render(write_still=True, use_viewport=True)

        with open(os.path.join(folder_path, "eye_features.csv"), "a") as f:
            f.write(f"{i}")
            for j in cornea_pos:
                f.write(f",{j * 1000}")
            for j in eye_centre_pos:
                f.write(f",{j * 1000}")
            f.write("\n")

        if time.time() - start > 5:
            print(f"Progress: {i}/{end_idx}")
            start = time.time()
