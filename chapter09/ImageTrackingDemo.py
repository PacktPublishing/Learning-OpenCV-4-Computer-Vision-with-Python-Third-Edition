#!/usr/bin/env python


"""This demo is adapted from the "Visualizing the Invisible" project by
Joseph Howse (Nummist Media Corporation Limited).

See the project's GitHub page at:
https://github.com/JoeHowse/VisualizingTheInvisible

The project is open-source under the BSD 3-Clause License, as follows.

Copyright (c) 2018-2022, Nummist Media Corporation Limited
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


import math
import timeit

import cv2
import numpy


def convert_bgr_to_gray(src, dst=None):
    weight = 1.0 / 3.0
    m = numpy.array([[weight, weight, weight]], numpy.float32)
    return cv2.transform(src, m, dst)


def map_2D_point_onto_3D_plane(point_2D, image_size, image_scale):
    x, y = point_2D
    w, h = image_size
    return (image_scale * (x - 0.5 * w),
            image_scale * (y - 0.5 * h),
            0.0)


def map_2D_points_onto_3D_plane(points_2D, image_size,
                                image_real_height):

    w, h = image_size
    image_scale = image_real_height / h

    points_3D = [map_2D_point_onto_3D_plane(
                     point_2D, image_size, image_scale)
                 for point_2D in points_2D]
    return numpy.array(points_3D, numpy.float32)


def map_vertices_to_plane(image_size, image_real_height):

    w, h = image_size

    vertices_2D = [(0, 0), (w, 0), (w, h), (0, h)]
    vertex_indices_by_face = [[0, 1, 2, 3]]

    vertices_3D = map_2D_points_onto_3D_plane(
        vertices_2D, image_size, image_real_height)
    return vertices_3D, vertex_indices_by_face


class ImageTrackingDemo():


    def __init__(self, capture, diagonal_fov_degrees=70.0,
                 target_fps=25.0,
                 reference_image_path='reference_image.png',
                 reference_image_real_height=1.0):

        self._capture = capture
        success, trial_image = capture.read()
        if success:
            # Use the actual image dimensions.
            h, w = trial_image.shape[:2]
        else:
            # Use the nominal image dimensions.
            w = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self._image_size = (w, h)

        diagonal_image_size = (w ** 2.0 + h ** 2.0) ** 0.5
        diagonal_fov_radians = \
            diagonal_fov_degrees * math.pi / 180.0
        focal_length = 0.5 * diagonal_image_size / math.tan(
            0.5 * diagonal_fov_radians)
        self._camera_matrix = numpy.array(
            [[focal_length, 0.0, 0.5 * w],
             [0.0, focal_length, 0.5 * h],
             [0.0, 0.0, 1.0]], numpy.float32)

        self._distortion_coefficients = None

        self._rodrigues_rotation_vector = numpy.array(
            [[0.0], [0.0], [1.0]], numpy.float32)
        self._euler_rotation_vector = numpy.zeros((3, 1), numpy.float32)  # Radians
        self._rotation_matrix = numpy.zeros((3, 3), numpy.float64)

        self._translation_vector = numpy.zeros((3, 1), numpy.float32)

        self._kalman = cv2.KalmanFilter(18, 6)

        self._kalman.processNoiseCov = numpy.identity(
            18, numpy.float32) * 1e-5
        self._kalman.measurementNoiseCov = numpy.identity(
            6, numpy.float32) * 1e-2
        self._kalman.errorCovPost = numpy.identity(
            18, numpy.float32)

        self._kalman.measurementMatrix = numpy.array(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            numpy.float32)

        self._init_kalman_transition_matrix(target_fps)

        self._was_tracking = False

        self._reference_image_real_height = \
            reference_image_real_height
        reference_axis_length = 0.5 * reference_image_real_height

        #-----------------------------------------------------------------------------
        # BEWARE!
        #-----------------------------------------------------------------------------
        #
        # OpenCV's coordinate system has non-standard axis directions:
        #   +X:  object's left; viewer's right from frontal view
        #   +Y:  down
        #   +Z:  object's backward; viewer's forward from frontal view
        #
        # Negate them all to convert to right-handed coordinate system (like OpenGL):
        #   +X:  object's right; viewer's left from frontal view
        #   +Y:  up
        #   +Z:  object's forward; viewer's backward from frontal view
        #
        #-----------------------------------------------------------------------------
        self._reference_axis_points_3D = numpy.array(
            [[0.0, 0.0, 0.0],
             [-reference_axis_length, 0.0, 0.0],
             [0.0, -reference_axis_length, 0.0],
             [0.0, 0.0, -reference_axis_length]], numpy.float32)

        self._bgr_image = None
        self._gray_image = None
        self._mask = None

        # Create and configure the feature detector.
        patchSize = 31
        self._feature_detector = cv2.ORB_create(
            nfeatures=250, scaleFactor=1.2, nlevels=16,
            edgeThreshold=patchSize, patchSize=patchSize)

        bgr_reference_image = cv2.imread(
            reference_image_path, cv2.IMREAD_COLOR)
        reference_image_h, reference_image_w = \
            bgr_reference_image.shape[:2]
        reference_image_resize_factor = \
            (2.0 * h) / reference_image_h
        bgr_reference_image = cv2.resize(
            bgr_reference_image, (0, 0), None,
            reference_image_resize_factor,
            reference_image_resize_factor, cv2.INTER_CUBIC)
        gray_reference_image = convert_bgr_to_gray(bgr_reference_image)
        reference_mask = numpy.empty_like(gray_reference_image)

        # Find keypoints and descriptors for multiple segments of
        # the reference image.
        reference_keypoints = []
        self._reference_descriptors = numpy.empty(
            (0, 32), numpy.uint8)
        num_segments_y = 6
        num_segments_x = 6
        for segment_y, segment_x in numpy.ndindex(
                (num_segments_y, num_segments_x)):
            y0 = reference_image_h * \
                segment_y // num_segments_y - patchSize
            x0 = reference_image_w * \
                segment_x // num_segments_x - patchSize
            y1 = reference_image_h * \
                (segment_y + 1) // num_segments_y + patchSize
            x1 = reference_image_w * \
                (segment_x + 1) // num_segments_x + patchSize
            reference_mask.fill(0)
            cv2.rectangle(
                reference_mask, (x0, y0), (x1, y1), 255, cv2.FILLED)
            more_reference_keypoints, more_reference_descriptors = \
                self._feature_detector.detectAndCompute(
                    gray_reference_image, reference_mask)
            if more_reference_descriptors is None:
                # No keypoints were found for this segment.
                continue
            reference_keypoints += more_reference_keypoints
            self._reference_descriptors = numpy.vstack(
                (self._reference_descriptors,
                 more_reference_descriptors))

        cv2.drawKeypoints(
            gray_reference_image, reference_keypoints,
            bgr_reference_image,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        ext_i = reference_image_path.rfind('.')
        reference_image_keypoints_path = \
            reference_image_path[:ext_i] + '_keypoints' + \
            reference_image_path[ext_i:]
        cv2.imwrite(
            reference_image_keypoints_path, bgr_reference_image)

        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6, key_size=12,
                            multi_probe_level=1)
        search_params = dict()
        self._descriptor_matcher = cv2.FlannBasedMatcher(
            index_params, search_params)
        self._descriptor_matcher.add([self._reference_descriptors])

        reference_points_2D = [keypoint.pt
                               for keypoint in reference_keypoints]
        self._reference_points_3D = map_2D_points_onto_3D_plane(
            reference_points_2D, gray_reference_image.shape[::-1],
            reference_image_real_height)

        (self._reference_vertices_3D,
         self._reference_vertex_indices_by_face) = \
            map_vertices_to_plane(
                    gray_reference_image.shape[::-1],
                    reference_image_real_height)


    def run(self):

        num_images_captured = 0
        start_time = timeit.default_timer()

        while cv2.waitKey(1) != 27:  # Escape
            success, self._bgr_image = self._capture.read(
                self._bgr_image)
            if success:
                num_images_captured += 1
                self._track_object()
                cv2.imshow('Image Tracking', self._bgr_image)
            delta_time = timeit.default_timer() - start_time
            if delta_time > 0.0:
                fps = num_images_captured / delta_time
                self._init_kalman_transition_matrix(fps)


    def _track_object(self):

        self._gray_image = convert_bgr_to_gray(
            self._bgr_image, self._gray_image)

        if self._mask is None:
            self._mask = numpy.full_like(self._gray_image, 255)

        keypoints, descriptors = \
            self._feature_detector.detectAndCompute(
                self._gray_image, self._mask)

        # Find the 2 best matches for each descriptor.
        matches = self._descriptor_matcher.knnMatch(descriptors, 2)

        # Filter the matches based on the distance ratio test.
        good_matches = [
            match[0] for match in matches
            if len(match) > 1 and \
                match[0].distance < 0.8 * match[1].distance
        ]

        # Select the good keypoints and draw them in red.
        good_keypoints = [keypoints[match.queryIdx]
                          for match in good_matches]
        cv2.drawKeypoints(self._gray_image, good_keypoints,
                          self._bgr_image, (0, 0, 255))

        min_good_matches_to_start_tracking = 8
        min_good_matches_to_continue_tracking = 6
        num_good_matches = len(good_matches)

        if num_good_matches < min_good_matches_to_continue_tracking:
            self._was_tracking = False
            self._mask.fill(255)

        elif num_good_matches >= \
                min_good_matches_to_start_tracking or \
                    self._was_tracking:

            # Select the 2D coordinates of the good matches.
            # They must be in an array of shape (N, 1, 2).
            good_points_2D = numpy.array(
                [[keypoint.pt] for keypoint in good_keypoints],
                numpy.float32)

            # Select the 3D coordinates of the good matches.
            # They must be in an array of shape (N, 1, 3).
            good_points_3D = numpy.array(
                [[self._reference_points_3D[match.trainIdx]]
                 for match in good_matches],
                numpy.float32)

            # Solve for the pose and find the inlier indices.
            (success, rodrigues_rotation_vector_temp,
             translation_vector_temp, inlier_indices) = \
                cv2.solvePnPRansac(good_points_3D, good_points_2D,
                                   self._camera_matrix,
                                   self._distortion_coefficients,
                                   None, None,
                                   useExtrinsicGuess=False,
                                   iterationsCount=100,
                                   reprojectionError=8.0,
                                   confidence=0.99,
                                   flags=cv2.SOLVEPNP_ITERATIVE)

            if success:

                self._translation_vector[:] = translation_vector_temp
                self._rodrigues_rotation_vector[:] = \
                    rodrigues_rotation_vector_temp
                self._convert_rodrigues_to_euler()

                if not self._was_tracking:
                    self._init_kalman_state_matrices()
                    self._was_tracking = True

                self._apply_kalman()

                # Select the inlier keypoints.
                inlier_keypoints = [good_keypoints[i]
                                    for i in inlier_indices.flat]

                # Draw the inlier keypoints in green.
                cv2.drawKeypoints(self._bgr_image, inlier_keypoints,
                                  self._bgr_image, (0, 255, 0))

                # Draw the axes of the tracked object.
                self._draw_object_axes()

                # Make and draw a mask around the tracked object.
                self._make_and_draw_object_mask()


    def _init_kalman_transition_matrix(self, fps):

        if fps <= 0.0:
            return

        # Velocity transition rate
        vel = 1.0 / fps

        # Acceleration transition rate
        acc = 0.5 * (vel ** 2.0)

        self._kalman.transitionMatrix = numpy.array(
            [[1.0, 0.0, 0.0, vel, 0.0, 0.0, acc, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0, vel, 0.0, 0.0, acc, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0, 0.0, vel, 0.0, 0.0, acc,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, vel, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, vel, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, vel,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              1.0, 0.0, 0.0, vel, 0.0, 0.0, acc, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 1.0, 0.0, 0.0, vel, 0.0, 0.0, acc, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 1.0, 0.0, 0.0, vel, 0.0, 0.0, acc],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 1.0, 0.0, 0.0, vel, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, vel, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, vel],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
            numpy.float32)


    def _init_kalman_state_matrices(self):

        t_x, t_y, t_z = self._translation_vector.flat
        pitch, yaw, roll = self._euler_rotation_vector.flat

        self._kalman.statePre = numpy.array(
            [[t_x],   [t_y],  [t_z],
             [0.0],   [0.0],  [0.0],
             [0.0],   [0.0],  [0.0],
             [pitch], [yaw],  [roll],
             [0.0],   [0.0],  [0.0],
             [0.0],   [0.0],  [0.0]], numpy.float32)
        self._kalman.statePost = numpy.array(
            [[t_x],   [t_y],  [t_z],
             [0.0],   [0.0],  [0.0],
             [0.0],   [0.0],  [0.0],
             [pitch], [yaw],  [roll],
             [0.0],   [0.0],  [0.0],
             [0.0],   [0.0],  [0.0]], numpy.float32)


    def _apply_kalman(self):

        self._kalman.predict()

        t_x, t_y, t_z = self._translation_vector.flat
        pitch, yaw, roll = self._euler_rotation_vector.flat

        estimate = self._kalman.correct(numpy.array(
            [[t_x],   [t_y], [t_z],
             [pitch], [yaw], [roll]], numpy.float32))

        translation_estimate = estimate[0:3]
        euler_rotation_estimate = estimate[9:12]

        self._translation_vector[:] = translation_estimate

        angular_delta = cv2.norm(self._euler_rotation_vector,
                                 euler_rotation_estimate, cv2.NORM_L2)

        MAX_ANGULAR_DELTA = 30.0 * math.pi / 180.0
        if angular_delta > MAX_ANGULAR_DELTA:
            # The rotational motion stabilization seems to be drifting
            # too far, probably due to an Euler angle singularity.

            # Reset the rotational motion stabilization.
            # Let the translational motion stabilization continue as-is.

            self._kalman.statePre[9] = pitch
            self._kalman.statePre[10] = yaw
            self._kalman.statePre[11] = roll
            self._kalman.statePre[12:18] = 0.0

            self._kalman.statePost[9] = pitch
            self._kalman.statePost[10] = yaw
            self._kalman.statePost[11] = roll
            self._kalman.statePost[12:18] = 0.0
        else:
            self._euler_rotation_vector[:] = euler_rotation_estimate
            self._convert_euler_to_rodrigues()


    def _convert_rodrigues_to_euler(self):

        self._rotation_matrix, jacobian = cv2.Rodrigues(
            self._rodrigues_rotation_vector, self._rotation_matrix)

        m00 = self._rotation_matrix[0, 0]
        m02 = self._rotation_matrix[0, 2]
        m10 = self._rotation_matrix[1, 0]
        m11 = self._rotation_matrix[1, 1]
        m12 = self._rotation_matrix[1, 2]
        m20 = self._rotation_matrix[2, 0]
        m22 = self._rotation_matrix[2, 2]

        # Convert to Euler angles using the yaw-pitch-roll
        # Tait-Bryan convention.
        if m10 > 0.998:
            # The rotation is near the "vertical climb" singularity.
            pitch = 0.5 * math.pi
            yaw = math.atan2(m02, m22)
            roll = 0.0
        elif m10 < -0.998:
            # The rotation is near the "nose dive" singularity.
            pitch = -0.5 * math.pi
            yaw = math.atan2(m02, m22)
            roll = 0.0
        else:
            pitch = math.asin(m10)
            yaw = math.atan2(-m20, m00)
            roll = math.atan2(-m12, m11)

        self._euler_rotation_vector[0] = pitch
        self._euler_rotation_vector[1] = yaw
        self._euler_rotation_vector[2] = roll


    def _convert_euler_to_rodrigues(self):

        pitch = self._euler_rotation_vector[0]
        yaw = self._euler_rotation_vector[1]
        roll = self._euler_rotation_vector[2]

        cyaw = math.cos(yaw)
        syaw = math.sin(yaw)
        cpitch = math.cos(pitch)
        spitch = math.sin(pitch)
        croll = math.cos(roll)
        sroll = math.sin(roll)

        # Convert from Euler angles using the yaw-pitch-roll
        # Tait-Bryan convention.
        m00 = cyaw * cpitch
        m01 = syaw * sroll - cyaw * spitch * croll
        m02 = cyaw * spitch * sroll + syaw * croll
        m10 = spitch
        m11 = cpitch * croll
        m12 = -cpitch * sroll
        m20 = -syaw * cpitch
        m21 = syaw * spitch * croll + cyaw * sroll
        m22 = -syaw * spitch * sroll + cyaw * croll

        self._rotation_matrix[0, 0] = m00
        self._rotation_matrix[0, 1] = m01
        self._rotation_matrix[0, 2] = m02
        self._rotation_matrix[1, 0] = m10
        self._rotation_matrix[1, 1] = m11
        self._rotation_matrix[1, 2] = m12
        self._rotation_matrix[2, 0] = m20
        self._rotation_matrix[2, 1] = m21
        self._rotation_matrix[2, 2] = m22

        self._rodrigues_rotation_vector, jacobian = cv2.Rodrigues(
            self._rotation_matrix, self._rodrigues_rotation_vector)


    def _draw_object_axes(self):

        points_2D, jacobian = cv2.projectPoints(
            self._reference_axis_points_3D,
            self._rodrigues_rotation_vector,
            self._translation_vector, self._camera_matrix,
            self._distortion_coefficients)

        origin = (int(points_2D[0, 0, 0]), int(points_2D[0, 0, 1]))
        right = (int(points_2D[1, 0, 0]), int(points_2D[1, 0, 1]))
        up = (int(points_2D[2, 0, 0]), int(points_2D[2, 0, 1]))
        forward = (int(points_2D[3, 0, 0]), int(points_2D[3, 0, 1]))

        # Draw the X axis in red.
        cv2.arrowedLine(
            self._bgr_image, origin, right, (0, 0, 255), 2)

        # Draw the Y axis in green.
        cv2.arrowedLine(
            self._bgr_image, origin, up, (0, 255, 0), 2)

        # Draw the Z axis in blue.
        cv2.arrowedLine(
            self._bgr_image, origin, forward, (255, 0, 0), 2)


    def _make_and_draw_object_mask(self):

        # Project the object's vertices into the scene.
        vertices_2D, jacobian = cv2.projectPoints(
            self._reference_vertices_3D,
            self._rodrigues_rotation_vector,
            self._translation_vector, self._camera_matrix,
            self._distortion_coefficients)
        vertices_2D = vertices_2D.astype(numpy.int32)

        # Make a mask based on the projected vertices.
        self._mask.fill(0)
        for vertex_indices in \
                self._reference_vertex_indices_by_face:
            cv2.fillConvexPoly(
                self._mask, vertices_2D[vertex_indices], 255)

        # Draw the mask in semi-transparent yellow.
        cv2.subtract(
            self._bgr_image, 48, self._bgr_image, self._mask)


def main():

    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    diagonal_fov_degrees = 70.0
    target_fps = 25.0

    demo = ImageTrackingDemo(
        capture, diagonal_fov_degrees, target_fps)
    demo.run()


if __name__ == '__main__':
    main()
