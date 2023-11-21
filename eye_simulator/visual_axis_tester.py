import numpy as np
from scipy.optimize import minimize
import time

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
    optical_axis = optical_axis / np.linalg.norm(optical_axis)
    theta = np.arctan2(optical_axis[0], optical_axis[2]) - np.pi
    phi = -(np.arccos(optical_axis[1]) - np.pi / 2)

    R = get_rot_y(theta) @ get_rot_x(phi)
    right = R @ np.array([1, 0, 0])
    up = R @ np.array([0, 1, 0])

    R = axis_and_angle_to_rot_mat(up, np.deg2rad(-alpha))
    forward = R @ optical_axis
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

    angles = minimize(calculate_error, visual_axis, args=(visual_axis, alpha, beta), method='SLSQP').x
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
    eye_world_pos = np.array([200.104, 146.681, 821.42])
    test_axis = focus_point - eye_world_pos
    test_axis /= np.linalg.norm(test_axis)
    theta = np.arctan2(test_axis[0], test_axis[2]) - np.pi
    phi = -(np.arccos(test_axis[1]) - np.pi / 2)

    x0 = np.array([phi, theta])

    angles = minimize(find_optical_axis, x0,
                            args=(eye_world_pos, focus_point, alpha, beta, eyeball_to_cornea_dist)).x

    phi = angles[0]
    theta = angles[1]
    R = get_rot_y(theta) @ get_rot_x(phi)
    optical_axis = R @ np.array([0, 0, -1])

    print(optical_axis, np.rad2deg(phi), np.rad2deg(theta))

def main():
    optical_axes = np.array([[1, 1, 1]])
    optical_axes = optical_axes / np.linalg.norm(optical_axes, axis=1)[:, None]
    for optical_axis in optical_axes:
        visual_axis = optical_to_visual(optical_axis, 15, 20)
        new_optical_axis = visual_to_optical(visual_axis, 15, 20)
        print(optical_axis, visual_axis, new_optical_axis)


    # top_left = np.array([100, 230, 750])
    # top_right = np.array([260, 230, 750])
    # bottom_left = np.array([100, 160, 750])
    # bottom_right = np.array([260, 160, 750])
    #
    # focus_points = np.array([top_left, top_right, bottom_left, bottom_right])
    # for focus_point in focus_points:
    #     look_at(focus_point, 10, 0, 5.3)


if __name__ == "__main__":
    main()