from pros_car_py.nav2_utils import (
    get_yaw_from_quaternion,
    get_direction_vector,
    get_angle_to_target,
    calculate_angle_point,
    cal_distance,
)
import math


class Nav2Processing:
    def __init__(self, ros_communicator, data_processor):
        self.ros_communicator = ros_communicator
        self.data_processor = data_processor
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

        # Mission-level state machine
        self.mission_state = "INIT"
        self.mission_bear_reached_announced = False
        

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

    def reset_mission_state(self):
        """Reset the mission state machine to its initial state."""
        self.mission_state = "INIT"
        self.mission_bear_reached_announced = False

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
        stop_distance = 0.45

        # If target is not found, keep searching.
        # Later, in the full mission state machine, this should switch back to exploration.
        # if found != 1:
        #     self.camera_lost_count += 1
        #     return "CLOCKWISE_ROTATION"
        max_lost_frames = 10

        if found != 1:
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

        # If depth is invalid because target is too close, stop and latch.
        if distance == -1.0:
            self.camera_target_reached = True
            print("[camera_nav] Target reached: invalid/too-close depth. Stop.")
            return "STOP"

        # Rotate until target is aligned.
        if delta_x > x_threshold:
            return "CLOCKWISE_ROTATION_SLOW"
        elif delta_x < -x_threshold:
            return "COUNTERCLOCKWISE_ROTATION_SLOW"

        # Target is aligned. Stop if close enough.
        if distance <= stop_distance:
            self.camera_target_reached = True
            print(f"[camera_nav] Target reached: distance={distance:.2f} m. Stop.")
            return "STOP"

        # Target is aligned but still far.
        return "FORWARD_SLOW"

    def mission_nav(self):
        """
        Mission-level navigation state machine.

        Currently implemented:
            INIT -> SEARCH_BEAR -> APPROACH_BEAR -> BEAR_REACHED

        TODO placeholders:
            GRASP_BEAR -> RETURN_HOME -> DROP_BEAR -> DONE
        """
        if self.mission_state == "INIT":
            self.mission_bear_reached_announced = False
            self.reset_camera_nav_state()
            self.set_mission_state("SEARCH_BEAR")
            return "STOP"

        if self.mission_state == "SEARCH_BEAR":
            yolo_target_info = self.data_processor.get_yolo_target_info()
            if yolo_target_info is None:
                return "CLOCKWISE_ROTATION"

            found = int(yolo_target_info[0])
            if found != 1:
                return "CLOCKWISE_ROTATION"

            self.set_mission_state("APPROACH_BEAR")
            return self.camera_nav()

        if self.mission_state == "APPROACH_BEAR":
            action = self.camera_nav()
            if self.camera_target_reached:
                self.set_mission_state("BEAR_REACHED")
                return "STOP"
            return action

        if self.mission_state == "BEAR_REACHED":
            return "STOP"

        if self.mission_state == "GRASP_BEAR":
            # TODO: Trigger arm grasp sequence.
            return "STOP"

        if self.mission_state == "RETURN_HOME":
            # TODO: Navigate back to the home/drop location.
            return "STOP"

        if self.mission_state == "DROP_BEAR":
            # TODO: Trigger arm release/drop sequence.
            return "STOP"

        if self.mission_state == "DONE":
            # TODO: Final mission-complete behavior.
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
