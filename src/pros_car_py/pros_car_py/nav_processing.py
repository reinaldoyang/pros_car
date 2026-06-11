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

        # Visual servoing state
        self.camera_target_reached = False
        self.camera_target_locked = False
        self.camera_lost_count = 0
        self.camera_reached_count = 0
        self.camera_required_reached_frames = 3

        # Mission-level state machine
        self.mission_state = "INIT"
        self.mission_bear_reached_announced = False
        self.mission_bear_lock_center_threshold = 50.0
        self.return_pose = [0.084, 0.028, 178.438]
        self.return_goal_published = False
        self.return_distance_announced = False
        self.return_align_announced = False
        self.return_position_threshold = 0.08
        self.return_yaw_threshold = 10.0
        self.grasp_verify_start_time = None
        self.grasp_verify_wait_time = 1.0
        self.grasp_verify_expected_distance = 0.288
        self.grasp_verify_distance_tolerance = 0.12
        self.grasp_verify_expected_delta_x = 26.0
        self.grasp_verify_delta_x_tolerance = 80.0
        self.drop_bear_triggered = False
        self.arm_missing_warned = False
        

    def reset_nav_process(self):
        self.finishFlag = False
        self.recordFlag = 0
        self.goal_published_flag = False

        self.reset_camera_nav_state()

    def reset_camera_nav_state(self):
        """Reset visual-servoing state used by camera_nav()."""
        self.camera_target_reached = False
        self.camera_target_locked = False
        self.camera_lost_count = 0
        self.camera_reached_count = 0

    def reset_mission_state(self):
        """Reset the mission state machine to its initial state."""
        self.mission_state = "INIT"
        self.mission_bear_reached_announced = False
        self.return_goal_published = False
        self.return_distance_announced = False
        self.return_align_announced = False
        self.grasp_verify_start_time = None
        self.drop_bear_triggered = False
        self.arm_missing_warned = False
        self.global_plan_msg = None
        self.index = 0

    def set_arm_controller(self, arm_controller):
        self.arm_controller = arm_controller

    def set_mission_state(self, next_state):
        if self.mission_state != next_state:
            print(f"[mission_nav] {self.mission_state} -> {next_state}")
            self.mission_state = next_state

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
        stop_distance = 0.4

        # If target is not found, keep searching.
        # Later, in the full mission state machine, this should switch back to exploration.
        # if found != 1:
        #     self.camera_lost_count += 1
        #     return "CLOCKWISE_ROTATION"
        max_lost_frames = 10

        if found != 1:
            self.camera_reached_count = 0
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

        # If target is off-center, keep centering first. Never move forward while
        # the bear is off-center, even when the reported depth is close/invalid.
        if not target_centered:
            self.camera_reached_count = 0
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
            self.return_align_announced = False
            self.set_mission_state("ALIGN_FIXED_POSE")
            return "STOP"

        if not self.return_goal_published:
            print(
                "[mission_nav] Publishing fixed return goal: "
                f"x={return_x:.3f}, y={return_y:.3f}, yaw={return_yaw:.3f}"
            )
            self.goal_published_flag = False
            self.recordFlag = 0
            self.global_plan_msg = None
            self.index = 0
            self.return_goal_published = True

        return self.get_action_from_nav2_plan_tf_p_2_p(
            goal_coordinates=self.return_pose,
            finish_distance=self.return_position_threshold,
            mark_finished=False,
        )

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
            self.set_mission_state("DROP_BEAR")
            return "STOP"

        if yaw_error > 0:
            return "COUNTERCLOCKWISE_ROTATION_SLOW"

        return "CLOCKWISE_ROTATION_SLOW"

    def get_action_to_lock_center_bear(self):
        yolo_target_info = self.data_processor.get_yolo_target_info()
        if yolo_target_info is None or len(yolo_target_info) < 3:
            return "CLOCKWISE_ROTATION_SLOW"

        found = int(yolo_target_info[0])
        distance = float(yolo_target_info[1])
        delta_x = float(yolo_target_info[2])
        if found != 1:
            return "CLOCKWISE_ROTATION_SLOW"

        if abs(delta_x) <= self.mission_bear_lock_center_threshold:
            print(
                "[mission_nav] Center bear locked: "
                f"distance={distance:.2f}, delta_x={delta_x:.1f}"
            )
            self.reset_camera_nav_state()
            self.camera_target_locked = True
            self.set_mission_state("APPROACH_BEAR")
            return self.camera_nav()

        if delta_x > self.mission_bear_lock_center_threshold:
            return "CLOCKWISE_ROTATION_SLOW"

        return "COUNTERCLOCKWISE_ROTATION_SLOW"

    def prepare_return_to_fixed_pose(self):
        self.global_plan_msg = None
        self.index = 0
        self.goal_published_flag = False
        self.recordFlag = 0
        self.return_goal_published = False
        self.return_distance_announced = False
        self.return_align_announced = False

    def current_yolo_matches_grasped_bear(self):
        yolo_target_info = self.data_processor.get_yolo_target_info()
        if yolo_target_info is None or len(yolo_target_info) < 3:
            return False

        found = int(yolo_target_info[0])
        distance = float(yolo_target_info[1])
        delta_x = float(yolo_target_info[2])
        distance_match = (
            abs(distance - self.grasp_verify_expected_distance)
            <= self.grasp_verify_distance_tolerance
        )
        delta_match = (
            abs(delta_x - self.grasp_verify_expected_delta_x)
            <= self.grasp_verify_delta_x_tolerance
        )
        print(
            "[mission_nav] Verify grasp YOLO: "
            f"found={found}, distance={distance:.3f}, delta_x={delta_x:.1f}, "
            f"distance_match={distance_match}, delta_match={delta_match}"
        )
        return found == 1 and distance_match and delta_match

    def get_action_to_verify_grasp(self):
        if self.grasp_verify_start_time is None:
            self.grasp_verify_start_time = time.time()
            print("[mission_nav] Waiting 1.0s before grasp verification")
            return "STOP"

        if time.time() - self.grasp_verify_start_time < self.grasp_verify_wait_time:
            return "STOP"

        self.grasp_verify_start_time = None
        if self.current_yolo_matches_grasped_bear():
            print("[mission_nav] Grasp verified; returning to fixed pose")
            self.prepare_return_to_fixed_pose()
            self.set_mission_state("RETURN_TO_FIXED_POSE")
            return "STOP"

        print("[mission_nav] Grasp verification failed; retrying bear approach")
        if self.arm_controller is None:
            if not self.arm_missing_warned:
                print("[mission_nav] Cannot open gripper before retry: arm_controller is None")
                self.arm_missing_warned = True
        elif hasattr(self.arm_controller, "release_bear"):
            self.arm_controller.release_bear()
        else:
            self.arm_controller.manual_control(0, "b")

        self.reset_camera_nav_state()
        self.set_mission_state("APPROACH_BEAR")
        return "STOP"

    def mission_nav(self):
        """
        Mission-level navigation state machine.

        Currently implemented:
            INIT -> LOCK_CENTER_BEAR -> APPROACH_BEAR -> BEAR_REACHED -> GRASP_BEAR
                 -> VERIFY_GRASP -> RETURN_TO_FIXED_POSE -> ALIGN_FIXED_POSE
                 -> DROP_BEAR -> DONE
        """
        if self.mission_state == "INIT":
            self.mission_bear_reached_announced = False
            self.reset_camera_nav_state()
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
                self.set_mission_state("DONE")
            return "STOP"

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
