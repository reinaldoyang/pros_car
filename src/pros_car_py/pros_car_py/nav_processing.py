from pros_car_py.nav2_utils import (
    get_yaw_from_quaternion,
    get_direction_vector,
    get_angle_to_target,
    calculate_angle_point,
    cal_distance,
)
import math
import time


class Nav2Processing:
    def __init__(self, ros_communicator, data_processor):
        self.ros_communicator = ros_communicator
        self.data_processor = data_processor
        self.arm_controller = None
        self.finishFlag = False
        self.global_plan_msg = None
        self.index = 0
        self.index_length = 0
        self.recordFlag = 0
        self.goal_published_flag = False
        self.fixed_goal_progress_pose = None
        self.fixed_goal_progress_yaw = None
        self.fixed_goal_progress_time = None
        self.fixed_goal_last_replan_time = 0.0
        self.fixed_goal_stuck_timeout = 5.0
        self.fixed_goal_progress_distance = 0.03
        self.fixed_goal_progress_yaw_degrees = 8.0
        self.fixed_goal_replan_cooldown = 2.0
        self.fixed_goal_periodic_replan_sec = 3.0
        self.fixed_goal_direct_control_distance = 0.35
        self.fixed_goal_empty_path_direct_fallback_distance = 1.2
        self.fixed_goal_last_periodic_replan_time = 0.0
        self.fixed_goal_replan_pending = False

        # Visual servoing state
        self.camera_target_reached = False
        self.camera_target_locked = False
        self.camera_lost_count = 0
        self.camera_reached_count = 0
        self.camera_required_reached_frames = 3
        self.camera_progress_pose = None
        self.camera_progress_yaw = None
        self.camera_progress_time = None
        self.camera_stuck_timeout = 2.0
        self.camera_progress_distance = 0.02
        self.camera_progress_yaw_degrees = 5.0
        self.camera_stuck_forward_start_time = None
        self.camera_stuck_forward_duration = 0.6

        # Standalone bridge crossing visual servo state.
        self.bridge_bear_grasp_distance = 0.4
        self.bridge_bear_center_threshold = 60.0
        self.bridge_bear_align_required_frames = 3
        self.bridge_center_threshold = 60.0
        self.bridge_required_detected_frames = 3
        self.bridge_lost_stop_frames = 5
        self.bridge_align_required_centered_frames = 3
        self.bridge_drive_realign_threshold = 140.0
        self.bridge_drive_realign_frames = 5
        self.bridge_brake_before_grasp_duration = 0.5
        self.bridge_brake_before_grasp_start_time = None
        self.bridge_after_grasp_forward_duration = 4.5
        self.bridge_after_grasp_forward_start_time = None
        self.bridge_grasp_retry_backup_start_time = None
        self.bridge_grasp_retry_backup_duration = 0.8
        self.bridge_crossing_phase = "ALIGN_BEAR"
        self.bridge_bear_centered_count = 0
        self.bridge_grasp_align_centered_count = 0
        self.bridge_align_centered_count = 0
        self.bridge_drive_off_center_count = 0
        self.bridge_grasp_triggered = False
        self.bridge_detected_count = 0
        self.bridge_lost_count = 0
        self.bridge_target_label_published = False
        self.bridge_crossing_last_log_time = 0.0
        self.task2_bridge_started = False

        # Mission-level state machine
        self.mission_state = "INIT"
        self.mission_arm_reset_done = False
        self.mission_bear_reached_announced = False
        self.mission_bear_lock_center_threshold = 50.0
        # self.return_pose = [0.084, 0.028, 178.438]
        self.return_pose = [0.104, 0.125, -147.486]
        self.return_goal_published = False
        self.return_distance_announced = False
        self.return_direct_control_announced = False
        self.return_align_announced = False
        self.return_align_rotation_action = None
        self.return_position_threshold = 0.08
        self.return_yaw_threshold = 10.0
        self.grasp_verify_start_time = None
        self.grasp_verify_wait_time = 1.0

        self.grasp_confirm_required_time = 3.0
        self.grasp_verify_timeout = 5.0
        self.grasp_match_accumulated_time = 0.0
        self.grasp_last_check_time = None
        self.grasp_success_latched = False

        self.grasp_verify_expected_distance = 0.288
        self.grasp_verify_max_distance = 0.33
        self.grasp_verify_distance_tolerance = 0.12
        self.grasp_verify_expected_delta_x = 26.0
        self.grasp_verify_delta_x_tolerance = 80.0
        self.grasp_retry_requested = False
        self.grasp_retry_backup_start_time = None
        self.grasp_retry_backup_duration = 0.8
        self.drop_bear_triggered = False
        self.arm_missing_warned = False

        # Task 3 fixed door-front pose in map frame: [x, y, yaw_deg].
        self.post_task1_pose = [0.894, 0.298, 88.246]
        self.door_front_pose = [2.873, 1.614, 1.849]
        self.task3_after_arm_pose = [3.808, 1.495, -28.078]
        self.door_goal_published = False
        self.door_distance_announced = False
        self.door_direct_control_announced = False
        self.door_position_threshold = 0.08
        self.door_yaw_threshold = 3.0
        self.door_slow_approach_distance = 0.25
        self.direct_pose_heading_threshold = 8.0
        self.door_align_announced = False
        self.door_align_rotation_action = None
        self.task3_ready_announced = False
        self.doorknob_target_label_published = False
        self.doorknob_servo_reached_count = 0
        self.doorknob_target_depth = 0.33
        self.doorknob_depth_tolerance = 0.03
        self.doorknob_center_threshold = 35.0
        self.doorknob_center_slow_threshold = 80.0
        self.doorknob_required_reached_frames = 3
        self.doorknob_rotation_action = None
        self.task3_arm_sequence_triggered = False

    def reset_nav_process(self):
        self.finishFlag = False
        self.recordFlag = 0
        self.goal_published_flag = False
        self.reset_fixed_goal_progress()

        self.reset_camera_nav_state()

    def reset_camera_nav_state(self):
        """Reset visual-servoing state used by camera_nav()."""
        self.camera_target_reached = False
        self.camera_target_locked = False
        self.camera_lost_count = 0
        self.camera_reached_count = 0
        self.camera_progress_pose = None
        self.camera_progress_yaw = None
        self.camera_progress_time = None
        self.camera_stuck_forward_start_time = None

    def reset_bridge_crossing_nav(self):
        self.bridge_crossing_phase = "ALIGN_BEAR"
        self.bridge_bear_centered_count = 0
        self.bridge_grasp_align_centered_count = 0
        self.bridge_align_centered_count = 0
        self.bridge_drive_off_center_count = 0
        self.bridge_brake_before_grasp_start_time = None
        self.bridge_after_grasp_forward_start_time = None
        self.bridge_grasp_retry_backup_start_time = None
        self.bridge_grasp_triggered = False
        self.bridge_detected_count = 0
        self.bridge_lost_count = 0
        self.bridge_target_label_published = False
        self.bridge_crossing_last_log_time = 0.0
        self.grasp_verify_start_time = None
        self.grasp_match_accumulated_time = 0.0
        self.grasp_last_check_time = None
        self.grasp_success_latched = False
        self.reset_camera_nav_state()

    def reset_mission_state(self):
        """Reset the mission state machine to its initial state."""
        self.mission_state = "INIT"
        self.mission_arm_reset_done = False
        self.mission_bear_reached_announced = False
        self.return_goal_published = False
        self.return_distance_announced = False
        self.return_direct_control_announced = False
        self.return_align_announced = False
        self.return_align_rotation_action = None
        self.grasp_verify_start_time = None
        self.grasp_retry_requested = False
        self.grasp_retry_backup_start_time = None
        self.drop_bear_triggered = False
        self.arm_missing_warned = False
        self.global_plan_msg = None
        self.index = 0
        self.reset_fixed_goal_progress()
        self.grasp_verify_start_time = None
        self.grasp_match_accumulated_time = 0.0
        self.grasp_last_check_time = None
        self.grasp_success_latched = False
        self.door_goal_published = False
        self.door_distance_announced = False
        self.door_direct_control_announced = False
        self.door_align_announced = False
        self.door_align_rotation_action = None
        self.task3_ready_announced = False
        self.doorknob_target_label_published = False
        self.doorknob_servo_reached_count = 0
        self.doorknob_rotation_action = None
        self.task3_arm_sequence_triggered = False
        self.task2_bridge_started = False

    def set_arm_controller(self, arm_controller):
        self.arm_controller = arm_controller

    def set_mission_state(self, next_state):
        if self.mission_state != next_state:
            print(f"[mission_nav] {self.mission_state} -> {next_state}")
            self.mission_state = next_state

    def reset_arm_for_mission_start(self):
        if self.mission_arm_reset_done:
            return

        if self.arm_controller is None:
            if not self.arm_missing_warned:
                print("[mission_nav] Cannot reset arm at mission start: arm_controller is None")
                self.arm_missing_warned = True
            self.mission_arm_reset_done = True
            return

        print("[mission_nav] Resetting arm for mission start")
        if hasattr(self.arm_controller, "reset_to_initial_pose"):
            self.arm_controller.reset_to_initial_pose()
        else:
            self.arm_controller.manual_control(0, "b")
        self.mission_arm_reset_done = True

    def consume_grasp_retry_requested(self):
        if not self.grasp_retry_requested:
            return False

        self.grasp_retry_requested = False
        return True

    def prepare_new_grasp_attempt(self):
        self.grasp_verify_start_time = None
        self.grasp_match_accumulated_time = 0.0
        self.grasp_last_check_time = None
        self.grasp_success_latched = False
        print("[mission_nav] New grasp attempt: reset grasp verification latch")

    def finish_nav_process(self):
        self.finishFlag = True
        self.recordFlag = 1

    def get_finish_flag(self):
        return self.finishFlag

    def get_action_from_nav2_plan(self, goal_coordinates=None):
        if goal_coordinates is not None and not self.goal_published_flag:
            self.ros_communicator.publish_goal_pose(goal_coordinates)
            self.goal_published_flag = True
        orientation_points, coordinates = (
            self.data_processor.get_processed_received_global_plan()
        )
        action_key = "STOP"
        if not orientation_points or not coordinates:
            action_key = "STOP"
        else:
            try:
                z, w = orientation_points[0]
                plan_yaw = get_yaw_from_quaternion(z, w)
                car_position, car_orientation = (
                    self.data_processor.get_processed_amcl_pose()
                )
                car_orientation_z, car_orientation_w = (
                    car_orientation[2],
                    car_orientation[3],
                )
                goal_position = self.ros_communicator.get_latest_goal()
                target_distance = cal_distance(car_position, goal_position)
                if target_distance < 0.5:
                    action_key = "STOP"
                    self.finishFlag = True
                else:
                    car_yaw = get_yaw_from_quaternion(
                        car_orientation_z, car_orientation_w
                    )
                    diff_angle = (plan_yaw - car_yaw) % 360.0
                    if diff_angle < 30.0 or (diff_angle > 330 and diff_angle < 360):
                        action_key = "FORWARD"
                    elif diff_angle > 30.0 and diff_angle < 180.0:
                        action_key = "COUNTERCLOCKWISE_ROTATION"
                    elif diff_angle > 180.0 and diff_angle < 330.0:
                        action_key = "CLOCKWISE_ROTATION"
                    else:
                        action_key = "STOP"
            except:
                action_key = "STOP"
        return action_key

    def get_action_from_nav2_plan_no_dynamic_p_2_p(self, goal_coordinates=None):
        if goal_coordinates is not None and not self.goal_published_flag:
            self.ros_communicator.publish_goal_pose(goal_coordinates)
            self.goal_published_flag = True

        # 只抓第一次路径
        if self.recordFlag == 0:
            if not self.check_data_availability():
                return "STOP"
            else:
                print("Get first path")
                self.index = 0
                self.global_plan_msg = (
                    self.data_processor.get_processed_received_global_plan_no_dynamic()
                )
                self.recordFlag = 1
                action_key = "STOP"

        car_position, car_orientation = self.data_processor.get_processed_amcl_pose()

        goal_position = self.ros_communicator.get_latest_goal()
        target_distance = cal_distance(car_position, goal_position)

        # 抓最近的物標(可調距離)
        target_x, target_y = self.get_next_target_point(car_position)

        if target_x is None or target_distance < 0.5:
            self.ros_communicator.reset_nav2()
            self.finish_nav_process()
            return "STOP"

        # 計算角度誤差
        diff_angle = self.calculate_diff_angle(
            car_position, car_orientation, target_x, target_y
        )
        if diff_angle < 20 and diff_angle > -20:
            action_key = "FORWARD"
        elif diff_angle < -20 and diff_angle > -180:
            action_key = "CLOCKWISE_ROTATION"
        elif diff_angle > 20 and diff_angle < 180:
            action_key = "COUNTERCLOCKWISE_ROTATION"
        return action_key

    def get_action_from_nav2_plan_tf_p_2_p(
        self,
        goal_coordinates=None,
        finish_distance=0.5,
        mark_finished=True,
    ):
        if goal_coordinates is not None and not self.goal_published_flag:
            self.ros_communicator.publish_goal_pose(goal_coordinates)
            self.goal_published_flag = True

        current_pose = self.get_current_tf_pose_map()
        goal_position = self.ros_communicator.get_latest_goal()
        if current_pose is None or goal_position is None:
            return "STOP"

        # 只抓第一次路径
        if self.recordFlag == 0:
            global_plan_msg = (
                self.data_processor.get_processed_received_global_plan_no_dynamic()
            )
            if global_plan_msg is None:
                return "STOP"

            print("Get first TF path")
            self.index = 0
            self.global_plan_msg = global_plan_msg
            self.recordFlag = 1

        if self.global_plan_msg is None:
            return "STOP"

        car_x = current_pose[0]
        car_y = current_pose[1]
        car_yaw = current_pose[2]
        car_position = [car_x, car_y, 0.0]

        goal_x = goal_position[0]
        goal_y = goal_position[1]
        target_distance = math.sqrt(
            (car_x - goal_x) ** 2
            + (car_y - goal_y) ** 2
        )

        if target_distance < finish_distance:
            self.ros_communicator.reset_nav2()
            if mark_finished:
                self.finish_nav_process()
            return "STOP"

        target_x, target_y = self.get_next_target_point(car_position)
        if target_x is None:
            target_x = goal_x
            target_y = goal_y

        target_yaw = math.degrees(math.atan2(target_y - car_y, target_x - car_x)) % 360.0
        diff_angle = self.angle_diff_deg(target_yaw, car_yaw)

        if abs(diff_angle) < 20.0:
            return "FORWARD"
        if diff_angle < 0.0:
            return "CLOCKWISE_ROTATION"
        if diff_angle > 0.0:
            return "COUNTERCLOCKWISE_ROTATION"

        return "STOP"

    def check_data_availability(self):
        return (
            self.data_processor.get_processed_received_global_plan_no_dynamic()
            and self.data_processor.get_processed_amcl_pose()
            and self.ros_communicator.get_latest_goal()
        )

    def get_next_target_point(self, car_position, min_required_distance=0.5):
        """
        選擇距離車輛 min_required_distance 以上最短路徑然後返回 target_x, target_y
        """
        if self.global_plan_msg is None or self.global_plan_msg.poses is None:
            print("Error: global_plan_msg is None or poses is missing!")
            return None, None
        while self.index < len(self.global_plan_msg.poses) - 1:
            target_x = self.global_plan_msg.poses[self.index].pose.position.x
            target_y = self.global_plan_msg.poses[self.index].pose.position.y
            distance_to_target = cal_distance(car_position, (target_x, target_y))

            if distance_to_target < min_required_distance:
                self.index += 1
            else:
                self.ros_communicator.publish_selected_target_marker(
                    x=target_x, y=target_y
                )
                return target_x, target_y

        return None, None

    def calculate_diff_angle(self, car_position, car_orientation, target_x, target_y):
        target_pos = [target_x, target_y]
        diff_angle = calculate_angle_point(
            car_orientation[2], car_orientation[3], car_position[:2], target_pos
        )
        return diff_angle

    def filter_negative_one(self, depth_list):
        return [depth for depth in depth_list if depth != -1.0]

    def get_camera_stuck_forward_action(self, close_enough, invalid_depth):
        if close_enough or invalid_depth:
            self.camera_progress_pose = None
            self.camera_progress_yaw = None
            self.camera_progress_time = None
            self.camera_stuck_forward_start_time = None
            return None

        now = time.time()
        if self.camera_stuck_forward_start_time is not None:
            if (
                now - self.camera_stuck_forward_start_time
                < self.camera_stuck_forward_duration
            ):
                return "FORWARD"

            self.camera_stuck_forward_start_time = None
            self.camera_progress_pose = None
            self.camera_progress_yaw = None
            self.camera_progress_time = None
            return None

        current_pose = self.get_current_tf_pose_map()
        if current_pose is None:
            return None

        current_xy = (current_pose[0], current_pose[1])
        current_yaw = current_pose[2]
        if self.camera_progress_pose is None:
            self.camera_progress_pose = current_xy
            self.camera_progress_yaw = current_yaw
            self.camera_progress_time = now
            return None

        moved = math.sqrt(
            (current_xy[0] - self.camera_progress_pose[0]) ** 2
            + (current_xy[1] - self.camera_progress_pose[1]) ** 2
        )
        yaw_changed = abs(self.angle_diff_deg(current_yaw, self.camera_progress_yaw))
        if (
            moved >= self.camera_progress_distance
            or yaw_changed >= self.camera_progress_yaw_degrees
        ):
            self.camera_progress_pose = current_xy
            self.camera_progress_yaw = current_yaw
            self.camera_progress_time = now
            return None

        if now - self.camera_progress_time < self.camera_stuck_timeout:
            return None

        print(
            "[camera_nav] Bear approach appears stuck; moving forward briefly "
            "before continuing alignment"
        )
        self.camera_stuck_forward_start_time = now
        self.camera_progress_pose = current_xy
        self.camera_progress_yaw = current_yaw
        self.camera_progress_time = now
        return "FORWARD"

    def bridge_crossing_log(self, message):
        now = time.time()
        if now - self.bridge_crossing_last_log_time >= 0.5:
            self.bridge_crossing_last_log_time = now
            print(message)

    def get_valid_bridge_status(self):
        segmentation_status = self.data_processor.get_yolo_segmentation_status()
        if segmentation_status is None:
            return None

        bridge_detected = bool(segmentation_status.get("bridge_detected", False))
        bridge_center_x = float(segmentation_status.get("bridge_center_x", -1.0))
        image_width = float(segmentation_status.get("image_width", 0.0))
        bridge_area_ratio = float(segmentation_status.get("bridge_area_ratio", 0.0))

        if (
            not bridge_detected
            or bridge_center_x < 0.0
            or image_width <= 0.0
            or bridge_area_ratio <= 0.001
        ):
            return None

        image_center_x = image_width / 2.0
        return {
            "center_error": bridge_center_x - image_center_x,
            "bridge_area_ratio": bridge_area_ratio,
        }

    def get_action_to_bridge_grasp_bear(self):
        if self.arm_controller is None:
            if not self.arm_missing_warned:
                print("[bridge_crossing] Cannot trigger grasp: arm_controller is None")
                self.arm_missing_warned = True
            return "STOP"

        if not self.bridge_grasp_triggered:
            print("[bridge_crossing] Triggering bear grasp")
            self.prepare_new_grasp_attempt()
            self.arm_controller.trigger_auto_grasp_at_base()
            self.bridge_grasp_triggered = True
            return "STOP"

        if self.arm_controller.grasp_done:
            print("[bridge_crossing] Grasp motion finished; verifying grasp")
            self.grasp_verify_start_time = None
            self.bridge_crossing_phase = "VERIFY_GRASP"

        return "STOP"

    def get_action_to_bridge_brake_before_grasp(self):
        now = time.time()
        if self.bridge_brake_before_grasp_start_time is None:
            self.bridge_brake_before_grasp_start_time = now
            print(
                "[bridge_crossing] Bear reached grasp depth; braking before grasp for "
                f"{self.bridge_brake_before_grasp_duration:.1f}s"
            )
            return "BRAKE"

        elapsed = now - self.bridge_brake_before_grasp_start_time
        if elapsed < self.bridge_brake_before_grasp_duration:
            return "BRAKE"

        self.bridge_brake_before_grasp_start_time = None
        self.bridge_grasp_triggered = False
        self.bridge_grasp_align_centered_count = 0
        self.bridge_crossing_phase = "ALIGN_BEAR_FOR_GRASP"
        print("[bridge_crossing] Brake complete; aligning bear before grasp")
        return "STOP"

    def get_action_to_bridge_align_bear_for_grasp(self):
        yolo_target_info = self.data_processor.get_yolo_target_info()
        if yolo_target_info is None or len(yolo_target_info) < 3:
            self.bridge_grasp_align_centered_count = 0
            self.bridge_crossing_log(
                "[bridge_crossing] phase=ALIGN_BEAR_FOR_GRASP, bear_detected=False, action=STOP"
            )
            return "STOP"

        found = int(yolo_target_info[0]) == 1
        bear_depth = float(yolo_target_info[1])
        bear_delta_x = float(yolo_target_info[2])

        if not found:
            self.bridge_grasp_align_centered_count = 0
            action = "CLOCKWISE_ROTATION_SLOW"
        elif abs(bear_delta_x) <= self.bridge_bear_center_threshold:
            self.bridge_grasp_align_centered_count += 1
            if self.bridge_grasp_align_centered_count >= self.bridge_bear_align_required_frames:
                self.bridge_crossing_phase = "GRASP_BEAR"
                self.bridge_grasp_triggered = False
                action = "STOP"
            else:
                action = "STOP"
        else:
            self.bridge_grasp_align_centered_count = 0
            if bear_delta_x > 0.0:
                action = "CLOCKWISE_ROTATION_SLOW"
            else:
                action = "COUNTERCLOCKWISE_ROTATION_SLOW"

        self.bridge_crossing_log(
            "[bridge_crossing] "
            f"phase=ALIGN_BEAR_FOR_GRASP, bear_detected={found}, "
            f"bear_depth={bear_depth:.2f}, bear_delta_x={bear_delta_x:.1f}, "
            f"centered_count={self.bridge_grasp_align_centered_count}, "
            f"action={action}"
        )
        return action

    def get_action_to_bridge_verify_grasp(self):
        if self.grasp_success_latched:
            print("[bridge_crossing] Grasp already latched as success; driving forward")
            self.bridge_crossing_phase = "DRIVE_FORWARD_AFTER_GRASP"
            self.bridge_after_grasp_forward_start_time = None
            return "STOP"

        now = time.time()
        if self.grasp_verify_start_time is None:
            self.grasp_verify_start_time = now
            self.grasp_match_accumulated_time = 0.0
            self.grasp_last_check_time = now
            print(
                "[bridge_crossing] Verifying grasp until bear is seen near grasp pose for "
                f"{self.grasp_confirm_required_time:.1f}s"
            )
            return "STOP"

        elapsed = now - self.grasp_verify_start_time
        dt = now - self.grasp_last_check_time if self.grasp_last_check_time else 0.0
        self.grasp_last_check_time = now

        if self.current_yolo_matches_grasped_bear():
            self.grasp_match_accumulated_time += max(0.0, dt)
            print(
                "[bridge_crossing] Bear still near grasp pose: "
                f"{self.grasp_match_accumulated_time:.1f}/"
                f"{self.grasp_confirm_required_time:.1f}s"
            )

        if self.grasp_match_accumulated_time >= self.grasp_confirm_required_time:
            print("[bridge_crossing] Grasp accepted; driving forward")
            self.grasp_success_latched = True
            self.grasp_verify_start_time = None
            self.grasp_match_accumulated_time = 0.0
            self.grasp_last_check_time = None
            self.bridge_crossing_phase = "DRIVE_FORWARD_AFTER_GRASP"
            self.bridge_after_grasp_forward_start_time = None
            return "STOP"

        if elapsed < self.grasp_verify_timeout:
            return "STOP"

        print("[bridge_crossing] Grasp verification timed out; backing up before retry")
        self.grasp_verify_start_time = None
        self.grasp_match_accumulated_time = 0.0
        self.grasp_last_check_time = None
        self.bridge_grasp_triggered = False
        self.grasp_success_latched = False

        if self.arm_controller is None:
            if not self.arm_missing_warned:
                print("[bridge_crossing] Cannot open gripper before retry: arm_controller is None")
                self.arm_missing_warned = True
        elif hasattr(self.arm_controller, "release_bear"):
            self.arm_controller.release_bear()
        else:
            self.arm_controller.manual_control(0, "b")

        self.reset_camera_nav_state()
        self.bridge_grasp_retry_backup_start_time = None
        self.bridge_crossing_phase = "BACK_UP_AFTER_GRASP_FAIL"
        return "STOP"

    def get_action_to_bridge_back_up_after_grasp_fail(self):
        now = time.time()
        if self.bridge_grasp_retry_backup_start_time is None:
            self.bridge_grasp_retry_backup_start_time = now
            print("[bridge_crossing] Grasp failed; backing up before finding bear again")
            return "BACKWARD_SLOW"

        if now - self.bridge_grasp_retry_backup_start_time < self.bridge_grasp_retry_backup_duration:
            return "BACKWARD_SLOW"

        self.bridge_grasp_retry_backup_start_time = None
        self.bridge_bear_centered_count = 0
        self.bridge_grasp_align_centered_count = 0
        self.bridge_drive_off_center_count = 0
        self.bridge_detected_count = 0
        self.bridge_lost_count = 0
        self.bridge_crossing_phase = "ALIGN_BEAR"
        print("[bridge_crossing] Retry backup complete; finding bear again")
        return "STOP"

    def get_action_to_bridge_drive_forward_after_grasp(self):
        now = time.time()
        if self.bridge_after_grasp_forward_start_time is None:
            self.bridge_after_grasp_forward_start_time = now
            print(
                "[bridge_crossing] Driving forward after grasp for "
                f"{self.bridge_after_grasp_forward_duration:.1f}s"
            )
            return "FORWARD"

        elapsed = now - self.bridge_after_grasp_forward_start_time
        if elapsed >= self.bridge_after_grasp_forward_duration:
            self.bridge_crossing_phase = "DONE"
            self.bridge_after_grasp_forward_start_time = None
            print("[bridge_crossing] Finished forward drive after grasp")
            return "STOP"

        action = "FORWARD"
        self.bridge_crossing_log(
            "[bridge_crossing] "
            "phase=DRIVE_FORWARD_AFTER_GRASP, "
            f"elapsed={elapsed:.1f}/"
            f"{self.bridge_after_grasp_forward_duration:.1f}s, "
            f"action={action}"
        )
        return action

    def bridge_crossing_nav(self):
        """Standalone bridge deck following mode.

        This is intentionally separate from mission_nav. It first aligns to the
        bridge deck, then drives forward with minimal rotation so the robot does
        not lose momentum while climbing.
        """
        if not self.bridge_target_label_published:
            self.publish_yolo_target_label("bear")
            self.bridge_target_label_published = True

        yolo_target_info = self.data_processor.get_yolo_target_info()
        bear_found = False
        bear_depth = -1.0
        bear_delta_x = 0.0
        if yolo_target_info is not None and len(yolo_target_info) >= 3:
            bear_found = int(yolo_target_info[0]) == 1
            bear_depth = float(yolo_target_info[1])
            bear_delta_x = float(yolo_target_info[2])

        if self.bridge_crossing_phase == "DONE":
            return "STOP"

        if self.bridge_crossing_phase == "DRIVE_FORWARD_AFTER_GRASP":
            return self.get_action_to_bridge_drive_forward_after_grasp()

        if self.bridge_crossing_phase == "VERIFY_GRASP":
            return self.get_action_to_bridge_verify_grasp()

        if self.bridge_crossing_phase == "BACK_UP_AFTER_GRASP_FAIL":
            return self.get_action_to_bridge_back_up_after_grasp_fail()

        if self.bridge_crossing_phase == "BRAKE_BEFORE_GRASP":
            return self.get_action_to_bridge_brake_before_grasp()

        if self.bridge_crossing_phase == "ALIGN_BEAR_FOR_GRASP":
            return self.get_action_to_bridge_align_bear_for_grasp()

        if self.bridge_crossing_phase == "GRASP_BEAR":
            return self.get_action_to_bridge_grasp_bear()

        if self.bridge_crossing_phase == "APPROACH_BEAR":
            self.bridge_crossing_phase = "BRAKE_BEFORE_GRASP"
            self.bridge_brake_before_grasp_start_time = None
            return "STOP"

        if self.bridge_crossing_phase == "ALIGN_BEAR":
            if not bear_found:
                self.bridge_bear_centered_count = 0
                action = "CLOCKWISE_ROTATION_SLOW"
            elif abs(bear_delta_x) <= self.bridge_bear_center_threshold:
                self.bridge_bear_centered_count += 1
                if self.bridge_bear_centered_count >= self.bridge_bear_align_required_frames:
                    self.bridge_crossing_phase = "DRIVE"
                    action = "FORWARD"
                else:
                    action = "STOP"
            else:
                self.bridge_bear_centered_count = 0
                if bear_delta_x > 0.0:
                    action = "CLOCKWISE_ROTATION_SLOW"
                else:
                    action = "COUNTERCLOCKWISE_ROTATION_SLOW"

            self.bridge_crossing_log(
                "[bridge_crossing] "
                f"phase=ALIGN_BEAR, bear_detected={bear_found}, "
                f"bear_depth={bear_depth:.2f}, bear_delta_x={bear_delta_x:.1f}, "
                f"centered_count={self.bridge_bear_centered_count}, "
                f"action={action}"
            )
            return action

        if (
            bear_found
            and bear_depth > 0.0
            and bear_depth <= self.bridge_bear_grasp_distance
        ):
            self.bridge_crossing_phase = "BRAKE_BEFORE_GRASP"
            self.bridge_brake_before_grasp_start_time = None
            self.bridge_crossing_log(
                "[bridge_crossing] "
                f"bear_depth={bear_depth:.2f} <= "
                f"{self.bridge_bear_grasp_distance:.2f}; "
                "braking before grasp"
            )
            return "BRAKE"

        if self.bridge_crossing_phase == "DRIVE":
            action = "FORWARD"
            self.bridge_crossing_log(
                "[bridge_crossing] "
                f"phase=DRIVE, bear_detected={bear_found}, "
                f"bear_depth={bear_depth:.2f}, action={action}"
            )
            return action

        bridge_status = self.get_valid_bridge_status()
        if bridge_status is None:
            self.bridge_lost_count += 1
            self.bridge_detected_count = 0
            self.bridge_align_centered_count = 0

            if (
                self.bridge_crossing_phase == "DRIVE"
                and self.bridge_lost_count < self.bridge_lost_stop_frames
            ):
                action = "FORWARD"
            else:
                self.bridge_crossing_phase = "ALIGN"
                self.bridge_drive_off_center_count = 0
                action = "STOP"

            self.bridge_crossing_log(
                "[bridge_crossing] "
                f"phase={self.bridge_crossing_phase}, bridge_lost=True, "
                f"lost_count={self.bridge_lost_count}, action={action}"
            )
            return action

        self.bridge_detected_count += 1
        self.bridge_lost_count = 0
        center_error = bridge_status["center_error"]

        if self.bridge_crossing_phase == "ALIGN":
            self.bridge_drive_off_center_count = 0
            if self.bridge_detected_count < self.bridge_required_detected_frames:
                action = "STOP"
            elif abs(center_error) <= self.bridge_center_threshold:
                self.bridge_align_centered_count += 1
                if (
                    self.bridge_align_centered_count
                    >= self.bridge_align_required_centered_frames
                ):
                    self.bridge_crossing_phase = "DRIVE"
                    action = "FORWARD"
                else:
                    action = "STOP"
            else:
                self.bridge_align_centered_count = 0
                if center_error > 0.0:
                    action = "CLOCKWISE_ROTATION_SLOW"
                else:
                    action = "COUNTERCLOCKWISE_ROTATION_SLOW"
        else:
            self.bridge_align_centered_count = 0
            if abs(center_error) > self.bridge_drive_realign_threshold:
                self.bridge_drive_off_center_count += 1
            else:
                self.bridge_drive_off_center_count = 0

            if self.bridge_drive_off_center_count >= self.bridge_drive_realign_frames:
                self.bridge_crossing_phase = "ALIGN"
                self.bridge_align_centered_count = 0
                action = "STOP"
            else:
                action = "FORWARD"

        self.bridge_crossing_log(
            "[bridge_crossing] "
            f"phase={self.bridge_crossing_phase}, "
            f"bridge_center_error={center_error:.1f}, "
            f"detected_count={self.bridge_detected_count}, "
            f"lost_count={self.bridge_lost_count}, "
            f"align_centered_count={self.bridge_align_centered_count}, "
            f"drive_off_center_count={self.bridge_drive_off_center_count}, "
            f"bear_detected={bear_found}, bear_depth={bear_depth:.2f}, "
            f"action={action}"
        )
        return action

    # def camera_nav(self):
    #     """
    #     YOLO 目標資訊 (yolo_target_info) 說明：

    #     - 索引 0 (index 0)：
    #         - 表示是否成功偵測到目標
    #         - 0：未偵測到目標
    #         - 1：成功偵測到目標

    #     - 索引 1 (index 1)：
    #         - 目標的深度距離 (與相機的距離，單位為公尺)，如果沒偵測到目標就回傳 0
    #         - 與目標過近時(大約 40 公分以內)會回傳 -1

    #     - 索引 2 (index 2)：
    #         - 目標相對於畫面正中心的像素偏移量
    #         - 若目標位於畫面中心右側，數值為正
    #         - 若目標位於畫面中心左側，數值為負
    #         - 若沒有目標則回傳 0

    #     畫面 n 個等分點深度 (camera_multi_depth) 說明 :

    #     - 儲存相機畫面中央高度上 n 個等距水平點的深度值。
    #     - 若距離過遠、過近（小於 40 公分）或是實體相機有時候深度會出一些問題，則該點的深度值將設定為 -1。
    #     """
    #     yolo_target_info = self.data_processor.get_yolo_target_info()
    #     camera_multi_depth = self.data_processor.get_camera_x_multi_depth()
    #     if camera_multi_depth == None or yolo_target_info == None:
    #         return "STOP"

    #     camera_forward_depth = self.filter_negative_one(camera_multi_depth[7:13])
    #     camera_left_depth = self.filter_negative_one(camera_multi_depth[0:7])
    #     camera_right_depth = self.filter_negative_one(camera_multi_depth[13:20])

    #     action = "STOP"
    #     limit_distance = 0.7

    #     # if all(depth > limit_distance for depth in camera_forward_depth):
    #     if yolo_target_info[0] == 1:
    #         if yolo_target_info[2] > 200.0:
    #             action = "CLOCKWISE_ROTATION_SLOW"
    #         elif yolo_target_info[2] < -200.0:
    #             action = "COUNTERCLOCKWISE_ROTATION_SLOW"
    #         else:
    #             if yolo_target_info[1] < 0.5:
    #                 action = "STOP"
    #             else:
    #                 action = "FORWARD_SLOW"
    #     else:
    #         action = "CLOCKWISE_ROTATION"
    #     # elif any(depth < limit_distance for depth in camera_left_depth):
    #     #     action = "CLOCKWISE_ROTATION"
    #     # elif any(depth < limit_distance for depth in camera_right_depth):
    #     #     action = "COUNTERCLOCKWISE_ROTATION"
    #     return action

    def camera_nav(self):
        """
        Visual servoing behavior.

        /yolo/target_info:
            index 0: found target, 1 = found, 0 = not found
            index 1: target depth distance in meters
            index 2: horizontal offset delta_x

        Behavior:
            - If target has already been reached, keep stopping.
            - If no target is found, rotate to search.
            - If target is found but not centered, rotate slowly.
            - If target is centered and far, move forward slowly.
            - If target is close enough, latch target_reached and stop.
        """
        yolo_target_info = self.data_processor.get_yolo_target_info()
        camera_multi_depth = self.data_processor.get_camera_x_multi_depth()

        # Once reached, stay stopped until reset_nav_process() is called.
        if self.camera_target_reached:
            return "STOP"

        if camera_multi_depth is None or yolo_target_info is None:
            return "STOP"

        found = int(yolo_target_info[0])
        distance = float(yolo_target_info[1])
        delta_x = float(yolo_target_info[2])

        # Tunable parameters
        x_threshold = 50.0
        far_forward_x_threshold = 260.0
        final_align_distance = 0.65
        stop_distance = 0.39

        # If target is not found, keep searching.
        # Later, in the full mission state machine, this should switch back to exploration.
        # if found != 1:
        #     self.camera_lost_count += 1
        #     return "CLOCKWISE_ROTATION"
        max_lost_frames = 60

        if found != 1:
            self.camera_reached_count = 0
            self.camera_progress_pose = None
            self.camera_progress_yaw = None
            self.camera_progress_time = None
            self.camera_stuck_forward_start_time = None
            if self.camera_target_locked:
                self.camera_lost_count += 1

                if self.camera_lost_count <= max_lost_frames:
                    print(f"[camera_nav] Target temporarily lost: {self.camera_lost_count}/{max_lost_frames}. Stop and wait.")
                    return "STOP"

                print("[camera_nav] Target lost for too long. Unlock and search again.")
                self.camera_target_locked = False
                self.camera_lost_count = 0
                return "CLOCKWISE_ROTATION"

            return "CLOCKWISE_ROTATION"

        # Target found, reset lost counter.
        self.camera_target_locked = True
        self.camera_lost_count = 0

        target_centered = abs(delta_x) <= x_threshold
        valid_depth = distance > 0.0 and distance != -1.0
        close_enough = valid_depth and distance <= stop_distance
        invalid_depth = distance == -1.0
        final_align_zone = invalid_depth or not valid_depth or distance <= final_align_distance

        stuck_action = self.get_camera_stuck_forward_action(close_enough, invalid_depth)
        if stuck_action is not None:
            self.camera_reached_count = 0
            return stuck_action

        # When still far from the bear, prefer forward motion for moderate
        # horizontal error so the car does not get trapped rotating near walls.
        # Near the grasp distance, switch back to stricter centering.
        if not target_centered:
            self.camera_reached_count = 0
            if not final_align_zone and abs(delta_x) <= far_forward_x_threshold:
                return "FORWARD_SLOW"
            if delta_x > x_threshold:
                return "CLOCKWISE_ROTATION_SLOW"
            return "COUNTERCLOCKWISE_ROTATION_SLOW"

        # Target is centered. Confirm close/reached over several frames before
        # latching so one noisy depth frame does not end the approach too early.
        if close_enough or invalid_depth:
            self.camera_reached_count += 1
            print(
                "[camera_nav] Reached confirmation: "
                f"{self.camera_reached_count}/{self.camera_required_reached_frames}, "
                f"distance={distance:.2f}"
            )

            if self.camera_reached_count >= self.camera_required_reached_frames:
                self.camera_target_reached = True
                print(f"[camera_nav] Target reached confirmed: distance={distance:.2f} m. Stop.")

            return "STOP"

        self.camera_reached_count = 0

        # Target is centered but still far.
        return "FORWARD_SLOW"

    def get_current_tf_yaw(self):
        if not hasattr(self.ros_communicator, "get_tf_yaw"):
            return None

        return self.ros_communicator.get_tf_yaw(
            parent_frame="odom",
            child_frame="base_footprint",
        )

    def get_current_tf_pose_map(self):
        if not hasattr(self.ros_communicator, "get_tf_pose"):
            return None

        return self.ros_communicator.get_tf_pose(
            parent_frame="map",
            child_frame="base_footprint",
        )

    def angle_diff_deg(self, target_yaw, current_yaw):
        return (target_yaw - current_yaw + 180.0) % 360.0 - 180.0

    def reset_fixed_goal_progress(self):
        self.fixed_goal_progress_pose = None
        self.fixed_goal_progress_yaw = None
        self.fixed_goal_progress_time = None
        self.fixed_goal_last_replan_time = 0.0
        self.fixed_goal_last_periodic_replan_time = time.time()
        self.fixed_goal_replan_pending = False

    def handle_fixed_goal_empty_path(self, goal_pose, goal_name, goal_flag_attr):
        print(
            f"[mission_nav] {goal_name} returned an empty Nav2 path; "
            "requesting a fresh Nav2 path"
        )
        if hasattr(self.ros_communicator, "clear_computed_path"):
            self.ros_communicator.clear_computed_path()
        self.global_plan_msg = None
        self.recordFlag = 0
        self.index = 0
        self.fixed_goal_replan_pending = False
        self.fixed_goal_last_periodic_replan_time = time.time()
        self.request_fixed_goal_replan(goal_pose, goal_flag_attr)
        return "STOP"

    def use_direct_fixed_pose_control(
        self,
        current_pose,
        target_pose,
        distance,
        position_threshold,
        goal_name,
    ):
        return self.get_direct_action_to_pose(
            current_pose,
            target_pose,
            distance,
            position_threshold,
        )

    def publish_yolo_target_label(self, label):
        if hasattr(self.ros_communicator, "publish_target_label"):
            self.ros_communicator.publish_target_label(label)

    def request_fixed_goal_replan(self, goal_pose, goal_flag_attr):
        self.global_plan_msg = None
        self.index = 0
        self.recordFlag = 0
        self.fixed_goal_replan_pending = False
        setattr(self, goal_flag_attr, True)

        if hasattr(self.ros_communicator, "clear_computed_path"):
            self.ros_communicator.clear_computed_path()
        if hasattr(self.ros_communicator, "request_compute_path_to_pose"):
            path_requested = self.ros_communicator.request_compute_path_to_pose(goal_pose)
            if not path_requested:
                setattr(self, goal_flag_attr, False)
        else:
            self.ros_communicator.publish_goal_pose(goal_pose)
            self.goal_published_flag = True

    def check_fixed_goal_stuck_and_replan(self, current_pose, goal_pose, goal_name, goal_flag_attr):
        now = time.time()
        current_xy = (current_pose[0], current_pose[1])
        current_yaw = current_pose[2]

        if self.fixed_goal_progress_pose is None:
            self.fixed_goal_progress_pose = current_xy
            self.fixed_goal_progress_yaw = current_yaw
            self.fixed_goal_progress_time = now
            return False

        moved = math.sqrt(
            (current_xy[0] - self.fixed_goal_progress_pose[0]) ** 2
            + (current_xy[1] - self.fixed_goal_progress_pose[1]) ** 2
        )
        yaw_changed = 0.0
        if self.fixed_goal_progress_yaw is not None:
            yaw_changed = abs(self.angle_diff_deg(current_yaw, self.fixed_goal_progress_yaw))
        if (
            moved >= self.fixed_goal_progress_distance
            or yaw_changed >= self.fixed_goal_progress_yaw_degrees
        ):
            self.fixed_goal_progress_pose = current_xy
            self.fixed_goal_progress_yaw = current_yaw
            self.fixed_goal_progress_time = now
            return False

        if self.fixed_goal_progress_time is None:
            self.fixed_goal_progress_time = now
            return False

        stuck_time = now - self.fixed_goal_progress_time
        replan_age = now - self.fixed_goal_last_replan_time
        if stuck_time < self.fixed_goal_stuck_timeout:
            return False
        if replan_age < self.fixed_goal_replan_cooldown:
            return False

        print(
            f"[mission_nav] {goal_name} appears stuck for {stuck_time:.1f}s; "
            "requesting a fresh Nav2 path"
        )
        self.global_plan_msg = None
        self.index = 0
        self.recordFlag = 0
        self.fixed_goal_progress_pose = current_xy
        self.fixed_goal_progress_yaw = current_yaw
        self.fixed_goal_progress_time = now
        self.fixed_goal_last_replan_time = now
        self.fixed_goal_last_periodic_replan_time = now
        self.fixed_goal_replan_pending = False
        self.request_fixed_goal_replan(goal_pose, goal_flag_attr)
        return True

    def check_fixed_goal_periodic_replan(self, goal_pose, goal_name, goal_flag_attr):
        now = time.time()
        if (
            now - self.fixed_goal_last_periodic_replan_time
            < self.fixed_goal_periodic_replan_sec
        ):
            return False

        current_pose = self.get_current_tf_pose_map()
        if current_pose is not None:
            distance = math.sqrt(
                (current_pose[0] - goal_pose[0]) ** 2
                + (current_pose[1] - goal_pose[1]) ** 2
            )
            if distance <= self.fixed_goal_empty_path_direct_fallback_distance:
                return False

        if getattr(self.ros_communicator, "compute_path_request_active", False):
            return False

        print(
            f"[mission_nav] Refreshing {goal_name} Nav2 path after "
            f"{self.fixed_goal_periodic_replan_sec:.1f}s"
        )
        self.fixed_goal_last_periodic_replan_time = now
        self.fixed_goal_last_replan_time = now
        self.fixed_goal_replan_pending = True

        if hasattr(self.ros_communicator, "clear_computed_path"):
            self.ros_communicator.clear_computed_path()
        if hasattr(self.ros_communicator, "request_compute_path_to_pose"):
            path_requested = self.ros_communicator.request_compute_path_to_pose(
                goal_pose,
                clear_existing_path=True,
            )
            if not path_requested:
                self.fixed_goal_replan_pending = False
                setattr(self, goal_flag_attr, False)
        else:
            self.ros_communicator.publish_goal_pose(goal_pose)
            self.goal_published_flag = True
        return True

    def maybe_replace_fixed_goal_plan(self, goal_name, goal_pose=None, goal_flag_attr=None):
        if not self.fixed_goal_replan_pending:
            return False
        if not hasattr(self.ros_communicator, "get_latest_computed_path"):
            return False

        computed_path = self.ros_communicator.get_latest_computed_path()
        if computed_path is None:
            return False
        if not computed_path.poses:
            print(
                f"[mission_nav] Refreshed {goal_name} path was empty; "
                "requesting another Nav2 path"
            )
            if goal_pose is not None and goal_flag_attr is not None:
                self.handle_fixed_goal_empty_path(
                    goal_pose,
                    goal_name,
                    goal_flag_attr,
                )
            else:
                self.fixed_goal_replan_pending = False
                if hasattr(self.ros_communicator, "clear_computed_path"):
                    self.ros_communicator.clear_computed_path()
            return True

        self.global_plan_msg = computed_path
        self.recordFlag = 1
        self.index = 0
        self.fixed_goal_replan_pending = False
        print(
            f"[mission_nav] Replaced {goal_name} path with refreshed Nav2 path: "
            f"{len(computed_path.poses)} poses"
        )
        return True

    def get_action_to_return_fixed_pose(self):
        current_pose = self.get_current_tf_pose_map()
        if current_pose is None:
            return "STOP"

        return_x, return_y, return_yaw = self.return_pose
        distance = math.sqrt(
            (current_pose[0] - return_x) ** 2
            + (current_pose[1] - return_y) ** 2
        )

        if not self.return_distance_announced:
            print(f"[mission_nav] Distance to fixed return pose: {distance:.2f}")
            self.return_distance_announced = True

        if distance <= self.return_position_threshold:
            self.ros_communicator.reset_nav2()
            self.reset_fixed_goal_progress()
            self.return_align_announced = False
            self.return_align_rotation_action = None
            self.set_mission_state("ALIGN_FIXED_POSE")
            return "STOP"

        if distance <= self.fixed_goal_direct_control_distance:
            if not self.return_direct_control_announced:
                print(
                    "[mission_nav] Return-home goal is too close for reliable Nav2; "
                    "using direct fixed-pose control"
                )
                self.return_direct_control_announced = True
                self.return_goal_published = False
                self.global_plan_msg = None
                self.index = 0
                self.recordFlag = 0
                self.ros_communicator.reset_nav2()
                if hasattr(self.ros_communicator, "clear_computed_path"):
                    self.ros_communicator.clear_computed_path()

            return self.use_direct_fixed_pose_control(
                current_pose,
                self.return_pose,
                distance,
                self.return_position_threshold,
                "return-home navigation",
            )

        self.return_direct_control_announced = False

        if not self.return_goal_published:
            print(
                "[mission_nav] Publishing Nav2 fixed return goal: "
                f"x={return_x:.3f}, y={return_y:.3f}, yaw={return_yaw:.3f}"
            )
            self.goal_published_flag = False
            self.recordFlag = 0
            self.global_plan_msg = None
            self.index = 0
            self.return_goal_published = True
            self.reset_fixed_goal_progress()
            if hasattr(self.ros_communicator, "clear_computed_path"):
                self.ros_communicator.clear_computed_path()
            if hasattr(self.ros_communicator, "request_compute_path_to_pose"):
                path_requested = self.ros_communicator.request_compute_path_to_pose(
                    self.return_pose
                )
                if not path_requested:
                    self.return_goal_published = False
            else:
                self.ros_communicator.publish_goal_pose(self.return_pose)
            return "STOP"

        if self.global_plan_msg is None:
            if not hasattr(self.ros_communicator, "get_latest_computed_path"):
                return "STOP"

            computed_path = self.ros_communicator.get_latest_computed_path()
            if computed_path is None:
                self.check_fixed_goal_periodic_replan(
                    self.return_pose,
                    "return-home navigation",
                    "return_goal_published",
                )
                return "STOP"

            if not computed_path.poses:
                if distance <= self.fixed_goal_empty_path_direct_fallback_distance:
                    print(
                        "[mission_nav] Empty Nav2 return-home path; "
                        "using direct fixed-pose control"
                    )
                    if hasattr(self.ros_communicator, "clear_computed_path"):
                        self.ros_communicator.clear_computed_path()
                    return self.use_direct_fixed_pose_control(
                        current_pose,
                        self.return_pose,
                        distance,
                        self.return_position_threshold,
                        "return-home navigation",
                    )
                return self.handle_fixed_goal_empty_path(
                    self.return_pose,
                    "return-home navigation",
                    "return_goal_published",
                )

            self.global_plan_msg = computed_path
            self.recordFlag = 1
            self.index = 0
            print(
                "[mission_nav] Following Nav2 fixed return path: "
                f"{len(computed_path.poses)} poses"
            )
            self.reset_fixed_goal_progress()

        if self.maybe_replace_fixed_goal_plan(
            "return-home navigation",
            self.return_pose,
            "return_goal_published",
        ):
            return "STOP"

        if self.check_fixed_goal_stuck_and_replan(
            current_pose,
            self.return_pose,
            "return-home navigation",
            "return_goal_published",
        ):
            return "STOP"

        if self.check_fixed_goal_periodic_replan(
            self.return_pose,
            "return-home navigation",
            "return_goal_published",
        ):
            return "STOP"

        car_x = current_pose[0]
        car_y = current_pose[1]
        car_yaw = current_pose[2]
        car_position = [car_x, car_y, 0.0]

        target_x, target_y = self.get_next_target_point(
            car_position,
            min_required_distance=0.25,
        )
        if target_x is None:
            target_x = return_x
            target_y = return_y

        target_yaw = math.degrees(math.atan2(target_y - car_y, target_x - car_x)) % 360.0
        yaw_error = self.angle_diff_deg(target_yaw, car_yaw)

        if abs(yaw_error) < 20.0:
            return "FORWARD"
        if yaw_error < 0.0:
            return "CLOCKWISE_ROTATION"
        return "COUNTERCLOCKWISE_ROTATION"

    def get_action_to_align_fixed_pose(self):
        current_pose = self.get_current_tf_pose_map()
        if current_pose is None:
            return "STOP"

        target_yaw = self.return_pose[2]
        yaw_error = self.angle_diff_deg(target_yaw, current_pose[2])

        if not self.return_align_announced:
            print(
                "[mission_nav] Aligning fixed pose yaw: "
                f"target_yaw={target_yaw:.1f}, "
                f"current_yaw={current_pose[2]:.1f}, "
                f"error={yaw_error:.1f}"
            )
            self.return_align_announced = True

        if abs(yaw_error) <= self.return_yaw_threshold:
            self.return_align_rotation_action = None
            self.set_mission_state("DROP_BEAR")
            return "STOP"

        if self.return_align_rotation_action is None:
            if yaw_error > 0:
                self.return_align_rotation_action = "COUNTERCLOCKWISE_ROTATION"
            else:
                self.return_align_rotation_action = "CLOCKWISE_ROTATION"
            print(
                "[mission_nav] Fixed-pose yaw rotation locked: "
                f"action={self.return_align_rotation_action}"
            )

        return self.return_align_rotation_action

    def get_action_to_lock_center_bear(self):
        yolo_target_info = self.data_processor.get_yolo_target_info()
        if yolo_target_info is None or len(yolo_target_info) < 3:
            return "CLOCKWISE_ROTATION_SLOW"

        found = int(yolo_target_info[0])
        distance = float(yolo_target_info[1])
        delta_x = float(yolo_target_info[2])
        if found != 1:
            return "CLOCKWISE_ROTATION_SLOW"

        valid_depth = distance > 0.0 and distance != -1.0
        approach_handoff_threshold = 260.0
        if abs(delta_x) <= approach_handoff_threshold and valid_depth:
            print(
                "[mission_nav] Bear selected for approach: "
                f"distance={distance:.2f}, delta_x={delta_x:.1f}"
            )
            self.reset_camera_nav_state()
            self.camera_target_locked = True
            self.set_mission_state("APPROACH_BEAR")
            return self.camera_nav()

        if delta_x > self.mission_bear_lock_center_threshold:
            return "CLOCKWISE_ROTATION_SLOW"

        return "COUNTERCLOCKWISE_ROTATION_SLOW"
    def prepare_task3_door_nav(self):
        self.global_plan_msg = None
        self.index = 0
        self.goal_published_flag = False
        self.recordFlag = 0
        self.door_goal_published = False
        self.door_distance_announced = False
        self.door_direct_control_announced = False
        self.door_align_announced = False
        self.door_align_rotation_action = None
        self.task3_ready_announced = False
        self.doorknob_rotation_action = None
        self.reset_fixed_goal_progress()

    def get_action_to_task3_fixed_pose(
        self,
        target_pose,
        goal_name,
        reached_state,
        align_state=None,
        position_threshold=None,
        direct_fallback=False,
        prefer_direct=False,
    ):
        current_pose = self.get_current_tf_pose_map()
        if current_pose is None:
            return "STOP"

        target_x, target_y, target_yaw = target_pose
        if position_threshold is None:
            position_threshold = self.door_position_threshold
        distance = math.sqrt(
            (current_pose[0] - target_x) ** 2
            + (current_pose[1] - target_y) ** 2
        )

        if not self.door_distance_announced:
            print(f"[task3] Distance to {goal_name}: {distance:.2f}")
            self.door_distance_announced = True

        if distance <= position_threshold:
            self.ros_communicator.reset_nav2()
            self.reset_fixed_goal_progress()
            print(
                f"[task3] Reached {goal_name} position: "
                f"x={target_x:.3f}, y={target_y:.3f}, yaw={target_yaw:.3f}"
            )
            self.door_align_announced = False
            self.door_align_rotation_action = None
            self.set_mission_state(align_state or reached_state)
            return "STOP"

        if distance <= self.fixed_goal_direct_control_distance:
            if not self.door_direct_control_announced:
                print(
                    f"[task3] {goal_name} is too close for reliable Nav2; "
                    "using direct fixed-pose control"
                )
                self.door_direct_control_announced = True
                self.door_goal_published = False
                self.global_plan_msg = None
                self.index = 0
                self.recordFlag = 0
                self.ros_communicator.reset_nav2()
                if hasattr(self.ros_communicator, "clear_computed_path"):
                    self.ros_communicator.clear_computed_path()

            return self.use_direct_fixed_pose_control(
                current_pose,
                target_pose,
                distance,
                position_threshold,
                goal_name,
            )

        self.door_direct_control_announced = False

        if prefer_direct:
            if self.check_fixed_goal_stuck_and_replan(
                current_pose,
                target_pose,
                goal_name,
                "door_goal_published",
            ):
                return "STOP"
            return self.get_direct_action_to_pose(
                current_pose,
                target_pose,
                distance,
                position_threshold,
            )

        if not self.door_goal_published:
            print(
                f"[task3] Publishing Nav2 {goal_name} goal: "
                f"x={target_x:.3f}, y={target_y:.3f}, yaw={target_yaw:.3f}"
            )
            self.goal_published_flag = False
            self.recordFlag = 0
            self.global_plan_msg = None
            self.index = 0
            self.door_goal_published = True
            self.reset_fixed_goal_progress()
            if hasattr(self.ros_communicator, "clear_computed_path"):
                self.ros_communicator.clear_computed_path()
            if hasattr(self.ros_communicator, "request_compute_path_to_pose"):
                path_requested = self.ros_communicator.request_compute_path_to_pose(
                    target_pose
                )
                if not path_requested:
                    self.door_goal_published = False
            else:
                self.ros_communicator.publish_goal_pose(target_pose)
            return "STOP"

        if self.global_plan_msg is None:
            if not hasattr(self.ros_communicator, "get_latest_computed_path"):
                return "STOP"

            computed_path = self.ros_communicator.get_latest_computed_path()
            if computed_path is None:
                self.check_fixed_goal_periodic_replan(
                    target_pose,
                    goal_name,
                    "door_goal_published",
                )
                if direct_fallback:
                    print(
                        f"[task3] No Nav2 path for {goal_name}; "
                        "using direct fixed-pose fallback"
                    )
                    return self.get_direct_action_to_pose(
                        current_pose,
                        target_pose,
                        distance,
                        position_threshold,
                    )
                return "STOP"

            if not computed_path.poses:
                if direct_fallback:
                    print(
                        f"[task3] Empty Nav2 path for {goal_name}; "
                        "using direct fixed-pose fallback"
                    )
                    if hasattr(self.ros_communicator, "clear_computed_path"):
                        self.ros_communicator.clear_computed_path()
                    return self.get_direct_action_to_pose(
                        current_pose,
                        target_pose,
                        distance,
                        position_threshold,
                    )
                return self.handle_fixed_goal_empty_path(
                    target_pose,
                    goal_name,
                    "door_goal_published",
                )

            self.global_plan_msg = computed_path
            self.recordFlag = 1
            self.index = 0
            print(
                f"[task3] Following Nav2 {goal_name} path: "
                f"{len(computed_path.poses)} poses"
            )
            self.reset_fixed_goal_progress()

        if self.maybe_replace_fixed_goal_plan(
            goal_name,
            target_pose,
            "door_goal_published",
        ):
            return "STOP"

        if self.check_fixed_goal_stuck_and_replan(
            current_pose,
            target_pose,
            goal_name,
            "door_goal_published",
        ):
            return "STOP"

        if self.check_fixed_goal_periodic_replan(
            target_pose,
            goal_name,
            "door_goal_published",
        ):
            return "STOP"

        car_x = current_pose[0]
        car_y = current_pose[1]
        car_yaw = current_pose[2]
        car_position = [car_x, car_y, 0.0]

        target_x, target_y = self.get_next_target_point(
            car_position,
            min_required_distance=0.25,
        )
        if target_x is None:
            target_x = target_pose[0]
            target_y = target_pose[1]

        target_yaw = math.degrees(math.atan2(target_y - car_y, target_x - car_x)) % 360.0
        yaw_error = self.angle_diff_deg(target_yaw, car_yaw)

        if abs(yaw_error) < 20.0:
            if distance <= self.door_slow_approach_distance:
                return "FORWARD_SLOW"
            return "FORWARD"
        if yaw_error < 0.0:
            return "CLOCKWISE_ROTATION"
        return "COUNTERCLOCKWISE_ROTATION"

    def get_action_to_door_pose(self):
        return self.get_action_to_task3_fixed_pose(
            self.door_front_pose,
            "door-front",
            "TASK3_READY_FOR_VISUAL_SERVO",
            align_state="TASK3_ALIGN_DOOR_POSE",
        )

    def get_action_to_post_task1_pose(self):
        return self.get_action_to_task3_fixed_pose(
            self.post_task1_pose,
            "post-Task-1 pose",
            "DONE",
            align_state="POST_TASK1_ALIGN_POSE",
        )

    def prepare_task3_after_arm_nav(self):
        self.prepare_task3_door_nav()

    def get_action_to_task3_after_arm_pose(self):
        return self.get_action_to_task3_fixed_pose(
            self.task3_after_arm_pose,
            "post-arm pose",
            "DONE",
            align_state="TASK3_ALIGN_AFTER_ARM_POSE",
            direct_fallback=True,
            prefer_direct=True,
        )

    def get_direct_action_to_pose(self, current_pose, target_pose, distance, position_threshold):
        if distance <= position_threshold:
            return "STOP"

        car_x = current_pose[0]
        car_y = current_pose[1]
        car_yaw = current_pose[2]
        target_yaw = math.degrees(
            math.atan2(target_pose[1] - car_y, target_pose[0] - car_x)
        ) % 360.0
        yaw_error = self.angle_diff_deg(target_yaw, car_yaw)

        if abs(yaw_error) <= self.direct_pose_heading_threshold:
            if distance <= self.door_slow_approach_distance:
                return "FORWARD_SLOW"
            return "FORWARD"
        if yaw_error < 0.0:
            return "CLOCKWISE_ROTATION"
        return "COUNTERCLOCKWISE_ROTATION"

    def get_action_to_align_door_pose(self):
        current_pose = self.get_current_tf_pose_map()
        if current_pose is None:
            return "STOP"

        target_yaw = self.door_front_pose[2]
        yaw_error = self.angle_diff_deg(target_yaw, current_pose[2])

        if not self.door_align_announced:
            print(
                "[task3] Aligning door-front yaw: "
                f"target_yaw={target_yaw:.1f}, "
                f"current_yaw={current_pose[2]:.1f}, "
                f"error={yaw_error:.1f}"
            )
            self.door_align_announced = True

        if abs(yaw_error) <= self.door_yaw_threshold:
            self.door_align_rotation_action = None
            print(
                "[task3] Reached door-front pose with yaw: "
                f"x={self.door_front_pose[0]:.3f}, "
                f"y={self.door_front_pose[1]:.3f}, "
                f"yaw={target_yaw:.3f}"
            )
            self.doorknob_target_label_published = False
            self.doorknob_servo_reached_count = 0
            self.set_mission_state("TASK3_READY_FOR_VISUAL_SERVO")
            return "STOP"

        if self.door_align_rotation_action is None:
            if yaw_error > 0:
                self.door_align_rotation_action = "COUNTERCLOCKWISE_ROTATION"
            else:
                self.door_align_rotation_action = "CLOCKWISE_ROTATION"
            print(
                "[task3] Door-front yaw rotation locked: "
                f"action={self.door_align_rotation_action}"
            )

        return self.door_align_rotation_action

    def get_action_to_align_post_task1_pose(self):
        current_pose = self.get_current_tf_pose_map()
        if current_pose is None:
            return "STOP"

        target_yaw = self.post_task1_pose[2]
        yaw_error = self.angle_diff_deg(target_yaw, current_pose[2])

        if not self.door_align_announced:
            print(
                "[mission_nav] Aligning post-Task-1 pose yaw: "
                f"target_yaw={target_yaw:.1f}, "
                f"current_yaw={current_pose[2]:.1f}, "
                f"error={yaw_error:.1f}"
            )
            self.door_align_announced = True

        if abs(yaw_error) <= self.door_yaw_threshold:
            self.door_align_rotation_action = None
            print(
                "[mission_nav] Reached post-Task-1 pose with yaw: "
                f"x={self.post_task1_pose[0]:.3f}, "
                f"y={self.post_task1_pose[1]:.3f}, "
                f"yaw={target_yaw:.3f}"
            )
            self.task2_bridge_started = False
            self.set_mission_state("TASK2_BRIDGE_CROSSING")
            return "STOP"

        if self.door_align_rotation_action is None:
            if yaw_error > 0:
                self.door_align_rotation_action = "COUNTERCLOCKWISE_ROTATION"
            else:
                self.door_align_rotation_action = "CLOCKWISE_ROTATION"
            print(
                "[mission_nav] Post-Task-1 yaw rotation locked: "
                f"action={self.door_align_rotation_action}"
            )

        return self.door_align_rotation_action

    def get_action_to_task2_bridge_crossing(self):
        if not self.task2_bridge_started:
            print("[mission_nav] Starting Task 2 bridge visual-servo pickup")
            self.reset_bridge_crossing_nav()
            self.task2_bridge_started = True

        action = self.bridge_crossing_nav()
        if self.bridge_crossing_phase == "DONE":
            print("[mission_nav] Task 2 bridge pickup/down-bridge motion complete; stopping")
            self.set_mission_state("DONE")
            return "STOP"

        return action

    def get_action_to_align_after_arm_pose(self):
        current_pose = self.get_current_tf_pose_map()
        if current_pose is None:
            return "STOP"

        target_yaw = self.task3_after_arm_pose[2]
        yaw_error = self.angle_diff_deg(target_yaw, current_pose[2])

        if not self.door_align_announced:
            print(
                "[task3] Aligning post-arm pose yaw: "
                f"target_yaw={target_yaw:.1f}, "
                f"current_yaw={current_pose[2]:.1f}, "
                f"error={yaw_error:.1f}"
            )
            self.door_align_announced = True

        if abs(yaw_error) <= self.door_yaw_threshold:
            self.door_align_rotation_action = None
            print(
                "[task3] Reached post-arm pose with yaw: "
                f"x={self.task3_after_arm_pose[0]:.3f}, "
                f"y={self.task3_after_arm_pose[1]:.3f}, "
                f"yaw={target_yaw:.3f}"
            )
            self.set_mission_state("DONE")
            return "STOP"

        if self.door_align_rotation_action is None:
            if yaw_error > 0:
                self.door_align_rotation_action = "COUNTERCLOCKWISE_ROTATION"
            else:
                self.door_align_rotation_action = "CLOCKWISE_ROTATION"
            print(
                "[task3] Post-arm yaw rotation locked: "
                f"action={self.door_align_rotation_action}"
            )

        return self.door_align_rotation_action

    def get_action_to_visual_servo_doorknob(self):
        if not self.doorknob_target_label_published:
            # Common aliases for the same physical target. The YOLO node
            # normalizes spaces/hyphens to underscores before matching.
            self.publish_yolo_target_label(
                "doorknob,door_knob,door knob,door_handle,handle,knob"
            )
            print("[task3] Starting doorknob visual servoing")
            self.doorknob_target_label_published = True

        yolo_target_info = self.data_processor.get_yolo_target_info()
        if yolo_target_info is None or len(yolo_target_info) < 3:
            self.doorknob_servo_reached_count = 0
            return "STOP"

        found = int(yolo_target_info[0])
        distance = float(yolo_target_info[1])
        delta_x = float(yolo_target_info[2])

        if found != 1:
            self.doorknob_servo_reached_count = 0
            self.doorknob_rotation_action = None
            return "CLOCKWISE_ROTATION_SLOW"

        valid_depth = distance > 0.0 and distance != -1.0
        centered = abs(delta_x) <= self.doorknob_center_threshold
        near_center = abs(delta_x) <= self.doorknob_center_slow_threshold

        if not near_center:
            self.doorknob_servo_reached_count = 0
            if self.doorknob_rotation_action is None:
                if delta_x > 0.0:
                    self.doorknob_rotation_action = "CLOCKWISE_ROTATION_SLOW"
                else:
                    self.doorknob_rotation_action = "COUNTERCLOCKWISE_ROTATION_SLOW"
            return self.doorknob_rotation_action

        if centered:
            self.doorknob_rotation_action = None

        if valid_depth:
            depth_error = distance - self.doorknob_target_depth
            depth_reached = abs(depth_error) <= self.doorknob_depth_tolerance
        else:
            depth_error = 0.0
            depth_reached = False

        if centered and depth_reached:
            self.doorknob_servo_reached_count += 1
            print(
                "[task3] Doorknob servo confirmation: "
                f"{self.doorknob_servo_reached_count}/"
                f"{self.doorknob_required_reached_frames}, "
                f"distance={distance:.2f}, delta_x={delta_x:.1f}"
            )
            if self.doorknob_servo_reached_count >= self.doorknob_required_reached_frames:
                print("[task3] Doorknob centered at target depth; visual servo complete")
                self.task3_arm_sequence_triggered = False
                self.set_mission_state("TASK3_ARM_SEQUENCE")
            return "STOP"

        self.doorknob_servo_reached_count = 0

        if not centered:
            if self.doorknob_rotation_action is None:
                if delta_x > 0.0:
                    self.doorknob_rotation_action = "CLOCKWISE_ROTATION_SLOW"
                else:
                    self.doorknob_rotation_action = "COUNTERCLOCKWISE_ROTATION_SLOW"
            return self.doorknob_rotation_action

        self.doorknob_rotation_action = None

        if not valid_depth:
            return "STOP"

        if depth_error > 0.0:
            return "FORWARD_SLOW"

        return "BACKWARD_SLOW"

    def get_action_to_task3_arm_sequence(self):
        if self.arm_controller is None:
            if not self.arm_missing_warned:
                print("[task3_arm] Cannot run door knob arm sequence: arm_controller is None")
                self.arm_missing_warned = True
            return "STOP"

        if not self.task3_arm_sequence_triggered:
            if hasattr(self.arm_controller, "trigger_task3_knob_sequence"):
                print("[task3_arm] Triggering door knob arm sequence")
                self.arm_controller.trigger_task3_knob_sequence()
                self.task3_arm_sequence_triggered = True
            else:
                print("[task3_arm] Arm controller does not support door knob sequence")
                self.set_mission_state("DONE")
            return "STOP"

        if getattr(self.arm_controller, "task3_knob_sequence_done", False):
            print("[task3_arm] Door knob arm sequence finished")
            self.prepare_task3_after_arm_nav()
            self.set_mission_state("TASK3_NAV_AFTER_ARM")

        return "STOP"

    def prepare_return_to_fixed_pose(self):
        self.global_plan_msg = None
        self.index = 0
        self.goal_published_flag = False
        self.recordFlag = 0
        self.return_goal_published = False
        self.return_distance_announced = False
        self.return_direct_control_announced = False
        self.return_align_announced = False
        self.return_align_rotation_action = None
        self.reset_fixed_goal_progress()
        self.doorknob_target_label_published = False
        self.doorknob_servo_reached_count = 0

    def current_yolo_matches_grasped_bear(self):
        yolo_target_info = self.data_processor.get_yolo_target_info()
        if yolo_target_info is None or len(yolo_target_info) < 3:
            return False

        found = int(yolo_target_info[0])
        distance = float(yolo_target_info[1])
        delta_x = float(yolo_target_info[2])
        distance_match = 0.0 < distance <= self.grasp_verify_max_distance
        delta_match = (
            abs(delta_x - self.grasp_verify_expected_delta_x)
            <= self.grasp_verify_delta_x_tolerance
        )
        print(
            "[mission_nav] Verify grasp YOLO: "
            f"found={found}, distance={distance:.3f}, delta_x={delta_x:.1f}, "
            f"distance_match={distance_match}, "
            f"max_distance={self.grasp_verify_max_distance:.3f}, "
            f"delta_match={delta_match}"
        )
        return found == 1 and distance_match and delta_match

    def get_action_to_verify_grasp(self):
        # If the grasp has already been accepted once,
        # never retry/release again because of unstable YOLO.
        if self.grasp_success_latched:
            print("[mission_nav] Grasp already latched as success; returning to fixed pose")
            self.prepare_return_to_fixed_pose()
            self.set_mission_state("RETURN_TO_FIXED_POSE")
            return "STOP"

        now = time.time()

        # Start a verification window after grasp finishes. The bear must match
        # the grasp-pose detection for about 3 accumulated seconds before the
        # grasp is accepted, so a short missed YOLO frame will not force a retry.
        if self.grasp_verify_start_time is None:
            self.grasp_verify_start_time = now
            self.grasp_match_accumulated_time = 0.0
            self.grasp_last_check_time = now
            print(
                "[mission_nav] Verifying grasp until bear is seen near grasp pose for "
                f"{self.grasp_confirm_required_time:.1f}s"
            )
            return "STOP"

        elapsed = now - self.grasp_verify_start_time
        dt = now - self.grasp_last_check_time if self.grasp_last_check_time else 0.0
        self.grasp_last_check_time = now

        if self.current_yolo_matches_grasped_bear():
            self.grasp_match_accumulated_time += max(0.0, dt)
            print(
                "[mission_nav] Bear still near grasp pose: "
                f"{self.grasp_match_accumulated_time:.1f}/"
                f"{self.grasp_confirm_required_time:.1f}s"
            )

        if self.grasp_match_accumulated_time >= self.grasp_confirm_required_time:
            print("[mission_nav] Grasp accepted and latched; no more retry checks")
            self.grasp_success_latched = True
            self.grasp_verify_start_time = None
            self.grasp_match_accumulated_time = 0.0
            self.grasp_last_check_time = None
            self.prepare_return_to_fixed_pose()
            self.set_mission_state("RETURN_TO_FIXED_POSE")
            return "STOP"

        if elapsed < self.grasp_verify_timeout:
            return "STOP"

        print("[mission_nav] Grasp verification timed out; retrying bear approach")
        self.grasp_verify_start_time = None
        self.grasp_match_accumulated_time = 0.0
        self.grasp_last_check_time = None

        if self.arm_controller is None:
            if not self.arm_missing_warned:
                print("[mission_nav] Cannot open gripper before retry: arm_controller is None")
                self.arm_missing_warned = True
        elif hasattr(self.arm_controller, "release_bear"):
            self.arm_controller.release_bear()
        else:
            self.arm_controller.manual_control(0, "b")

        self.reset_camera_nav_state()
        self.grasp_retry_requested = True
        self.grasp_retry_backup_start_time = None
        self.set_mission_state("BACK_UP_AFTER_GRASP_FAIL")
        return "STOP"

    def get_action_to_back_up_after_grasp_fail(self):
        now = time.time()
        if self.grasp_retry_backup_start_time is None:
            self.grasp_retry_backup_start_time = now
            print(
                "[mission_nav] Grasp failed; backing up before retrying bear approach"
            )
            return "BACKWARD_SLOW"

        if now - self.grasp_retry_backup_start_time < self.grasp_retry_backup_duration:
            return "BACKWARD_SLOW"

        self.grasp_retry_backup_start_time = None
        self.set_mission_state("LOCK_CENTER_BEAR")
        return "STOP"

    def mission_nav(self):
        """
        Mission-level navigation state machine.

        Currently implemented:
            INIT -> LOCK_CENTER_BEAR -> APPROACH_BEAR -> BEAR_REACHED -> GRASP_BEAR
                 -> VERIFY_GRASP -> BACK_UP_AFTER_GRASP_FAIL
                 -> RETURN_TO_FIXED_POSE -> ALIGN_FIXED_POSE
                 -> DROP_BEAR -> POST_TASK1_NAV_TO_POSE
                 -> POST_TASK1_ALIGN_POSE -> TASK2_BRIDGE_CROSSING -> DONE
        """
        if self.mission_state == "INIT":
            self.reset_arm_for_mission_start()
            self.mission_bear_reached_announced = False
            self.reset_camera_nav_state()
            self.publish_yolo_target_label("bear")
            self.set_mission_state("LOCK_CENTER_BEAR")
            return "STOP"

        if self.mission_state == "LOCK_CENTER_BEAR":
            return self.get_action_to_lock_center_bear()

        if self.mission_state == "APPROACH_BEAR":
            action = self.camera_nav()
            if self.camera_target_reached:
                self.set_mission_state("BEAR_REACHED")
                return "STOP"
            return action

        if self.mission_state == "BEAR_REACHED":
            return "STOP"

        if self.mission_state == "GRASP_BEAR":
            if self.arm_controller is None:
                if not self.arm_missing_warned:
                    print("[mission_nav] Cannot monitor grasp: arm_controller is None")
                    self.arm_missing_warned = True
                return "STOP"

            if self.arm_controller.grasp_done:
                self.grasp_verify_start_time = None
                self.set_mission_state("VERIFY_GRASP")
                return "STOP"

            return "STOP"

        if self.mission_state == "VERIFY_GRASP":
            return self.get_action_to_verify_grasp()

        if self.mission_state == "BACK_UP_AFTER_GRASP_FAIL":
            return self.get_action_to_back_up_after_grasp_fail()

        if self.mission_state == "RETURN_TO_FIXED_POSE":
            return self.get_action_to_return_fixed_pose()

        if self.mission_state == "ALIGN_FIXED_POSE":
            return self.get_action_to_align_fixed_pose()

        if self.mission_state == "DROP_BEAR":
            if not self.drop_bear_triggered:
                print("[mission_nav] Dropping bear")
                if self.arm_controller is None:
                    if not self.arm_missing_warned:
                        print("[mission_nav] Cannot drop bear: arm_controller is None")
                        self.arm_missing_warned = True
                elif hasattr(self.arm_controller, "release_bear"):
                    self.arm_controller.release_bear()
                else:
                    self.arm_controller.manual_control(0, "b")
                self.drop_bear_triggered = True
                print("[mission_nav] Bear dropped; navigating to post-Task-1 pose")
                self.prepare_task3_door_nav()
                self.set_mission_state("POST_TASK1_NAV_TO_POSE")
            return "STOP"

        if self.mission_state == "POST_TASK1_NAV_TO_POSE":
            return self.get_action_to_post_task1_pose()

        if self.mission_state == "POST_TASK1_ALIGN_POSE":
            return self.get_action_to_align_post_task1_pose()

        if self.mission_state == "TASK2_BRIDGE_CROSSING":
            return self.get_action_to_task2_bridge_crossing()

        if self.mission_state == "TASK3_NAV_TO_DOOR":
            return self.get_action_to_door_pose()

        if self.mission_state == "TASK3_ALIGN_DOOR_POSE":
            return self.get_action_to_align_door_pose()

        if self.mission_state == "TASK3_READY_FOR_VISUAL_SERVO":
            if not self.task3_ready_announced:
                print("[task3] Ready for doorknob visual servoing")
                self.task3_ready_announced = True
            return self.get_action_to_visual_servo_doorknob()

        if self.mission_state == "TASK3_ARM_SEQUENCE":
            return self.get_action_to_task3_arm_sequence()

        if self.mission_state == "TASK3_NAV_AFTER_ARM":
            return self.get_action_to_task3_after_arm_pose()

        if self.mission_state == "TASK3_ALIGN_AFTER_ARM_POSE":
            return self.get_action_to_align_after_arm_pose()

        if self.mission_state == "DONE":
            return "STOP"

        print(f"[mission_nav] Unknown state '{self.mission_state}'. Resetting mission.")
        self.reset_mission_state()
        return "STOP"

    def camera_nav_unity(self):
        """
        YOLO 目標資訊 (yolo_target_info) 說明：

        - 索引 0 (index 0)：
            - 表示是否成功偵測到目標
            - 0：未偵測到目標
            - 1：成功偵測到目標

        - 索引 1 (index 1)：
            - 目標的深度距離 (與相機的距離，單位為公尺)，如果沒偵測到目標就回傳 0
            - 與目標過近時(大約 40 公分以內)會回傳 -1

        - 索引 2 (index 2)：
            - 目標相對於畫面正中心的像素偏移量
            - 若目標位於畫面中心右側，數值為正
            - 若目標位於畫面中心左側，數值為負
            - 若沒有目標則回傳 0

        畫面 n 個等分點深度 (camera_multi_depth) 說明 :

        - 儲存相機畫面中央高度上 n 個等距水平點的深度值。
        - 若距離過遠、過近（小於 40 公分）或是實體相機有時候深度會出一些問題，則該點的深度值將設定為 -1。
        """
        yolo_target_info = self.data_processor.get_yolo_target_info()
        camera_multi_depth = self.data_processor.get_camera_x_multi_depth()
        yolo_target_info[1] *= 1
        camera_multi_depth = list(
            map(lambda x: x * 1.0, self.data_processor.get_camera_x_multi_depth())
        )

        if camera_multi_depth == None or yolo_target_info == None:
            return "STOP"

        camera_forward_depth = self.filter_negative_one(camera_multi_depth[7:13])
        camera_left_depth = self.filter_negative_one(camera_multi_depth[0:7])
        camera_right_depth = self.filter_negative_one(camera_multi_depth[13:20])
        action = "STOP"
        limit_distance = 10.0
        print(yolo_target_info[1])
        if all(depth > limit_distance for depth in camera_forward_depth):
            if yolo_target_info[0] == 1:
                if yolo_target_info[2] > 200.0:
                    action = "CLOCKWISE_ROTATION_SLOW"
                elif yolo_target_info[2] < -200.0:
                    action = "COUNTERCLOCKWISE_ROTATION_SLOW"
                else:
                    if yolo_target_info[1] < 2.0:
                        action = "STOP"
                    else:
                        action = "FORWARD_SLOW"
            else:
                action = "FORWARD"
        elif any(depth < limit_distance for depth in camera_left_depth):
            action = "CLOCKWISE_ROTATION"
        elif any(depth < limit_distance for depth in camera_right_depth):
            action = "COUNTERCLOCKWISE_ROTATION"
        return action

    def stop_nav(self):
        return "STOP"
