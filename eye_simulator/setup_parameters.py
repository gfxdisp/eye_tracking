import numpy as np


class SetupParameters:
    def __init__(self, hdrmfs_camera_calib, capture_params, eye):
        camera_params = capture_params['camera_params'][eye]
        self.leds_world_pos = hdrmfs_camera_calib[f'leds_world_pos_{eye}']
        matlab_extrinsic_mat = np.array(hdrmfs_camera_calib[f'M_et_{eye}'])

        self.M_extr = np.eye(4)
        self.M_extr[0:3, 0:3] = matlab_extrinsic_mat[0:3, 0:3]
        self.M_extr[0:3, 1:3] = -self.M_extr[0:3, 1:3]
        self.M_extr[0:3, 3] = (np.array([0, 0, 0, 1]) @ np.linalg.inv(matlab_extrinsic_mat))[0:3] / 1000
        self.M_extr = self.M_extr.transpose()

        self.M_disp_rsb = np.array(hdrmfs_camera_calib['M_disp_rsb'][0])
        self.M_intr = np.reshape(camera_params['intrinsic_matrix'], (3, 3))
        self.M_intr[[0, 1], 2] -= 1
        self.sx = 6.144  # Sensor width in mm
        self.sy = 4.915  # Sensor height in mm
        self.W = 1280  # Image width in pixels
        self.H = 1024  # Image height in pixels
        self.capture_offset = camera_params['capture_offset']
        self.roi = camera_params['region_of_interest']

        self.cornea_centre_distance = 5.3 / 1000  # m
        self.cornea_curvature_radius = 8.2 / 1000  # m
        self.cornea_refraction_index = 1.3375
        self.alpha = -2.758 if eye == 'left' else -5.0  # degrees
        self.beta = 4.244  # degrees

        # self.cornea_centre_distance = 6.32233071280933 / 1000  # m
        # self.cornea_curvature_radius = 8.0096467356863 / 1000  # m
        # self.cornea_refraction_index = 1.52780683114967
        # self.alpha = -3.68388813209528 if eye == 'left' else 5.0  # degrees
        # self.beta = -0.449709334328888  # degrees

        # Errors in measurements
        self.translation_error_std = np.array([2.5, 2.5, 5.0]) / 1000
        self.rotation_deg_error_std = np.array([1.0, 1.0, 1.0])
        self.focal_length_error_std = np.array([0.01, 0.01]) * self.M_intr[[0, 1], [0, 1]]
        self.principal_shift_error_std = np.array([0.01, 0.01]) * self.M_intr[[0, 1], [2, 2]]
        self.eyeball_to_cornea_dist_error_std = 0.0005
        self.cornea_radius_error_std = 0.0005
        self.cornea_refraction_index_error_std = 0.2
        self.alpha_error_std = 5.0
        self.beta_error_std = 3.0

    def randomize_params(self, random_hardware=False):
        if random_hardware:
            # Randomize extrinsic matrix
            ang_x = np.random.normal(0, self.rotation_deg_error_std[0])
            ang_y = np.random.normal(0, self.rotation_deg_error_std[1])
            ang_z = np.random.normal(0, self.rotation_deg_error_std[2])

            rot_x = get_rot_x(ang_x)
            rot_y = get_rot_y(ang_y)
            rot_z = get_rot_z(ang_z)
            self.M_extr[0:3, 0:3] = rot_z @ rot_y @ rot_x @ self.M_extr[0:3, 0:3]

            trans_x = np.random.normal(0, self.translation_error_std[0])
            trans_y = np.random.normal(0, self.translation_error_std[1])
            trans_z = np.random.normal(0, self.translation_error_std[2])
            self.M_extr[3, 0:3] += np.array([trans_x, trans_y, trans_z])

            # Randomize intrinsic matrix
            # self.M_intr[[0, 1], [0, 1]] += np.random.normal(0, self.focal_length_error_std)
            # self.M_intr[[0, 1], [2, 2]] += np.random.normal(0, self.principal_shift_error_std)

        # self.cornea_centre_distance += np.random.normal(0, self.eyeball_to_cornea_dist_error_std)
        # self.cornea_curvature_radius += np.random.normal(0, self.cornea_radius_error_std)
        #
        # self.cornea_refraction_index += np.random.normal(0, self.cornea_refraction_index_error_std)

        # self.alpha += np.random.normal(0, self.alpha_error_std)
        # self.beta += np.random.normal(0, self.beta_error_std)


def get_rot_x(angle_deg):
    angle_r = np.deg2rad(angle_deg)
    mat = np.array([[1, 0, 0], [0, np.cos(angle_r), np.sin(angle_r)], [0, -np.sin(angle_r), np.cos(angle_r)]])
    return mat


def get_rot_y(angle_deg):
    angle_r = np.deg2rad(angle_deg)
    mat = np.array([[np.cos(angle_r), 0, -np.sin(angle_r)], [0, 1, 0], [np.sin(angle_r), 0, np.cos(angle_r)]])
    return mat


def get_rot_z(angle_deg):
    angle_r = np.deg2rad(angle_deg)
    mat = np.array([[np.cos(angle_r), -np.sin(angle_r), 0], [np.sin(angle_r), np.cos(angle_r), 0], [0, 0, 1]])
    return mat