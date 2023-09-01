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


def look_at(focus_point, alpha, beta, eyeball_to_cornea_dist):
    eye = bpy.data.objects.get("Eye")
    eye_world_pos = np.array(eye.matrix_world.translation)
    x0 = np.array([0, 0, -1])

    visual_axis = minimize(find_visual_axis, x0,
                           args=(eye_world_pos, focus_point, alpha, beta, eyeball_to_cornea_dist)).x
    visual_axis = visual_axis / np.linalg.norm(visual_axis)
    optical_axis = get_optical_axis(visual_axis, eye_world_pos, alpha, beta, eyeball_to_cornea_dist)

    point_on_sphere = eye_world_pos + eyeball_to_cornea_dist * optical_axis
    visual_axis = (focus_point - point_on_sphere) / np.linalg.norm(focus_point - point_on_sphere)

    r = np.linalg.norm(point_on_sphere - eye_world_pos)
    theta = np.math.acos((point_on_sphere[2] - eye_world_pos[2]) / r)
    phi = np.math.atan2(point_on_sphere[1] - eye_world_pos[1], point_on_sphere[0] - eye_world_pos[0])

    q = Euler((0, -0.5 * np.pi, 0), 'XYZ').to_quaternion()
    q.rotate(Euler((0, theta, phi), 'XYZ'))
    eye.rotation_mode = 'QUATERNION'
    eye.rotation_quaternion = q
    return visual_axis


def find_visual_axis(visual_axis, eye_pos, focus_point, alpha, beta, eyeball_to_cornea_dist):
    optical_axis = get_optical_axis(visual_axis, eye_pos, alpha, beta, eyeball_to_cornea_dist)
    cornea_pos = eye_pos + eyeball_to_cornea_dist * optical_axis

    distance = point_to_line_dist(focus_point, cornea_pos, visual_axis)
    return distance


def get_optical_axis(visual_axis, eye_pos, alpha, beta, eyeball_to_cornea_dist):
    visual_axis = visual_axis / np.linalg.norm(visual_axis)

    rot_y = get_rot_y(alpha)
    rot_x = get_rot_x(beta)

    optical_axis = rot_x @ rot_y @ visual_axis
    return optical_axis


def point_to_line_dist(line_point, point, line_direction):
    V = line_point - point
    projection = np.dot(V, line_direction) / np.linalg.norm(line_direction) ** 2 * line_direction
    distance = np.linalg.norm(V - projection)
    return distance


def render_images(setup_params, images_num, folder_path):
    bpy.context.scene.render.image_settings.file_format = 'JPEG'
    bpy.context.scene.render.image_settings.color_mode = 'BW'
    bpy.context.scene.eevee.taa_render_samples = 1
    bpy.context.scene.eevee.taa_samples = 1
    bpy.context.scene.eevee.sss_samples = 1
    bpy.context.scene.eevee.volumetric_samples = 1
    bpy.context.scene.eevee.use_gtao = False

    if not os.path.exists(os.path.join(folder_path, "images")):
        os.makedirs(os.path.join(folder_path, "images"))

    images = [f for f in os.listdir(os.path.join(folder_path, "images")) if
              os.path.isfile(os.path.join(folder_path, "images", f)) and (f.endswith(".jpg") or f.endswith(".png"))]
    if len(images) > 0:
        last_image_idx = np.max([int(f.split("_")[1]) for f in images])
        start_idx = last_image_idx + 1
        end_idx = start_idx + images_num
    else:
        start_idx = 0
        end_idx = images_num

    leds_camera_pos = np.zeros((setup_params.leds_world_pos.shape[0], 4))
    for i in range(setup_params.leds_world_pos.shape[0]):
        leds_camera_pos[i, :] = np.append(setup_params.leds_world_pos[i, :], 1) @ setup_params.M_extr
    leds_camera_pos = leds_camera_pos[:, :3] / leds_camera_pos[:, 3:] / 1000
    max_led_depth = np.max(leds_camera_pos[:, 2])
    for i in range(leds_camera_pos.shape[0]):
        led = bpy.data.objects[f"Light.{i:03d}"]
        led.location = leds_camera_pos[i, :]

    eye = bpy.data.objects['Eye']
    eye.rotation_mode = 'XZY'
    eye.rotation_euler = (0, 0.5 * np.pi, 0)

    cornea = bpy.data.objects['Cornea']
    cornea_size = setup_params.cornea_curvature_radius * 2
    cornea_offset = np.array([cornea_size / 2, cornea_size / 2, 0])
    cornea.rotation_mode = 'XYZ'
    cornea.rotation_euler = (0, 0, 0)
    cornea.scale = (cornea_size, cornea_size, cornea_size)
    cornea.matrix_world.translation = np.array(eye.matrix_world.translation) - np.array(
        [0, 0, setup_params.cornea_centre_distance])

    bpy.data.materials['Cornea'].node_tree.nodes['Glass BSDF'].inputs[
        2].default_value = setup_params.cornea_refraction_index

    camera = bpy.data.objects['Camera']
    camera.matrix_world.translation = np.array([0, 0, 0])
    camera.rotation_mode = 'XYZ'
    camera.rotation_euler = (0, np.pi, np.pi)

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

    projection_matrix_inv = np.linalg.inv(projection_matrix)
    camera_matrix = np.array(camera.matrix_world)

    d_res = [2160, 3840]
    display_corners = np.array([[1, d_res[0], 0, 1], [d_res[1], d_res[0], 0, 1], [d_res[1], 1, 0, 1], [1, 1, 0, 1]])
    world_corners = (display_corners @ setup_params.M_disp_rsb @ setup_params.M_extr) / 1000
    min_screen_x = np.min(world_corners[:, 0])
    max_screen_x = np.max(world_corners[:, 0])
    min_screen_y = np.min(world_corners[:, 1])
    max_screen_y = np.max(world_corners[:, 1])
    mean_screen_z = np.mean(world_corners[:, 2])

    start = time.time()
    for i in range(start_idx, end_idx):
        focus_point = np.array(
            [random.uniform(min_screen_x, max_screen_x), random.uniform(min_screen_y, max_screen_y), mean_screen_z])

        visual_axis = look_at(focus_point, setup_params.alpha, setup_params.beta, setup_params.cornea_centre_distance)
        bpy.context.view_layer.update()

        offset = np.array(eye.matrix_world.translation) - np.array(cornea.matrix_world.translation)
        offset[2] = 0
        depth = random.uniform(max_led_depth + 0.02, max_led_depth + 0.07)
        min_px_coords = np.array([min_x * 2 - 1, max_y * 2 - 1, 1, 1]) * depth
        max_px_coords = np.array([max_x * 2 - 1, min_y * 2 - 1, 1, 1]) * depth

        min_camera_coords = (camera_matrix @ projection_matrix_inv @ min_px_coords)[:3] + offset + cornea_offset
        max_camera_coords = (camera_matrix @ projection_matrix_inv @ max_px_coords)[:3] + offset - cornea_offset

        x = random.uniform(min_camera_coords[0], max_camera_coords[0])
        y = random.uniform(min_camera_coords[1], max_camera_coords[1])
        eye.matrix_world.translation = [x, y, depth]
        bpy.context.view_layer.update()

        eye_centre_pos = eye.matrix_world.translation

        for j in range(leds_camera_pos.shape[0]):
            bpy.data.objects[f"Light.{j:03d}"].hide_render = True

        bpy.context.scene.render.filepath = os.path.join(folder_path, "images", f"image_{i:04d}_lights_off.jpg")
        with stdout_redirected(os.devnull):
            bpy.ops.render.render(write_still=True, use_viewport=True)

        for j in range(leds_camera_pos.shape[0]):
            bpy.data.objects[f"Light.{j:03d}"].hide_render = False

        bpy.context.scene.render.filepath = os.path.join(folder_path, "images", f"image_{i:04d}_lights_on.jpg")
        with stdout_redirected(os.devnull):
            bpy.ops.render.render(write_still=True, use_viewport=True)

        with open(os.path.join(folder_path, "eye_features.csv"), "a") as f:
            f.write(f"{i}")
            for j in visual_axis:
                f.write(f",{j}")
            for j in eye_centre_pos:
                f.write(f",{j * 1000}")
            f.write("\n")

        if time.time() - start > 5:
            print(f"Progress: {i}/{end_idx}")
            start = time.time()


def get_rot_x(angle_deg):
    angle_r = np.deg2rad(angle_deg)
    mat = np.array([[1, 0, 0], [0, np.cos(angle_r), -np.sin(angle_r)], [0, np.sin(angle_r), np.cos(angle_r)]])
    return mat


def get_rot_y(angle_deg):
    angle_r = np.deg2rad(angle_deg)
    mat = np.array([[np.cos(angle_r), 0, np.sin(angle_r)], [0, 1, 0], [-np.sin(angle_r), 0, np.cos(angle_r)]])
    return mat


def get_rot_z(angle_deg):
    angle_r = np.deg2rad(angle_deg)
    mat = np.array([[np.cos(angle_r), -np.sin(angle_r), 0], [np.sin(angle_r), np.cos(angle_r), 0], [0, 0, 1]])
    return mat
