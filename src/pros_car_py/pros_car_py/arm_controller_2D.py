# return 角度一律 radians
import math
import tf2_ros
import tf2_geometry_msgs
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, PointStamped
from pros_car_py.car_models import DeviceDataTypeEnum
import numpy as np
import time
import sys
from scipy.spatial.transform import Rotation as R
import pybullet as p
import numpy as np
import threading
import rclpy


class ArmController:
    def __init__(self, ros_communicator, data_processor):
        self.ros_communicator = ros_communicator
        self.data_processor = data_processor
        self.target_marker = None
        # self.default_grasp_x = 0.174
        # self.default_grasp_z = -0.075
        self.default_grasp_x = 0.14
        self.default_grasp_z = -0.090
        self.bridge_grasp_x = 0.170
        self.bridge_grasp_z = -0.090
        self.grasp_in_progress = False
        self.grasp_done = False
        self.release_done = False
        self.task3_knob_sequence_in_progress = False
        self.task3_knob_sequence_done = False
        self.task3_elbow_push_in_progress = False
        self.task3_elbow_push_done = False
        # User-servo degree order follows manual arm control:
        # [elbow/submode0, wrist/submode1, finger/submode2].
        self.task3_knob_ready_degrees = [90.0, 62.0, 1.0]
        self.task3_knob_push_degrees = [123.0, 62.0, 1.0]
        
        # 建立 TF2 監聽器 (使用 ros_communicator 作為 Node)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.ros_communicator)
        
        # ==========================================
        # 1. 手臂基礎與機構設定 (統一管理區)
        # ==========================================
        self.base_link_name = 'arm_ik_base' # 第一個馬達的基準座標系
        
        # 動態定義所有關節 (加入 length 臂長 與 angle_offset 角度補償)
        # angle_offset: IK 算出來的數學 0 度可能不是 Unity 的 0 度，可透過這個補償
        self.joint_limits = [
            {"length": 0.08089007, "min_angle": -180, "max_angle": 0, "init": -180, "offset": 270, "dir": -1.0},  # Joint 0 (Shoulder)
            {"length": 0.11, "min_angle": -240, "max_angle": 0, "init": -0,   "offset": -120, "dir": -1.0},  # Joint 1 (Elbow)
            {"length": 0.00, "min_angle": 20, "max_angle": 90,  "init": 90,  "offset": 0.0, "dir": 1.0},  # Joint 2 (Gripper)
        ]
        # Fixed joint-angle grasp target for tuning without IK.
        # Uses internal arm angles: [Joint 0, Joint 1].
        self.joint_grasp_target_angles = [-78.0, -179.0]
        self.joint_grasp_return_angles = [
            self.joint_limits[0]["init"],
            self.joint_limits[1]["init"],
        ]
        self.joint_grasp_close_angle = 22
        # self.joint_grasp_close_angle = self.joint_limits[2]["min_angle"]
        
        self.joint_angles = [joint["init"] for joint in self.joint_limits]
        self.current_user_servo_degrees = [
            (float(self.joint_angles[i]) * self.joint_limits[i].get("dir", 1.0)) % 360.0
            for i in range(len(self.joint_angles))
        ]
        self.manual_step = 3.0   
        
        print(f"🦾 Arm Controller Initialized: {len(self.joint_limits)} Joints Managed.")

    # ==========================================
    # 2. 手動控制邏輯 (Manual Control)
    # ==========================================
    def reset_to_initial_pose(self):
        self.joint_angles = [joint["init"] for joint in self.joint_limits]
        self._clamp_and_publish()
        self._visualize_arm_lines()
        print("手臂已重置為初始角度。")

    def release_bear(self):
        self.release_done = False
        self.reset_to_initial_pose()
        self.release_done = True
        print("[mission_nav] Bear released / gripper opened.")

    def manual_control(self, index, key):
        """處理手動按鍵輸入，並根據 index 控制特定關節"""
        
        # 處理不依賴 index 的全域指令 ('b', 'q')
        if key == "b":  
            self.reset_to_initial_pose()
            return False
            
        elif key == "q":  
            # 結束控制
            print("結束手臂手動控制。")
            return True

        # 處理針對特定 index 的控制 ('i', 'k')
        if 0 <= index < len(self.joint_limits):
            if key == "i":
                self.joint_angles[index] += self.manual_step
            elif key == "k":
                self.joint_angles[index] -= self.manual_step
            else:
                print(f"按鍵 '{key}' 無效，請使用 'i'(增加), 'k'(減少), 'b'(重置), 或 'q'(取消)。")
                return False
                
            # 計算完畢後，進行安全檢查並發布
            self._clamp_and_publish()
            self._visualize_arm_lines()
        else:
            print(f"索引 {index} 無效，請確保其在範圍內（0-{len(self.joint_limits) - 1}）。")
            
        return False

    def auto_control(self, key=None, mode="auto_arm_control"):
        """自動抓取 /yolo/target_marker 的目標"""
        
        # 1. 取得最新目標位置
        target_marker = self.ros_communicator.latest_yolo_marker
        if not target_marker:
            print("尚未收到 YOLO 目標，等待中...")
            return
        
        if key == "g":
            self.target_marker = target_marker
        elif key == "q":
            self.target_marker = None
            return
        elif key == "b":  
            self.reset_to_initial_pose()
            return
        else :
            print(f"按鍵 '{key}' 無效，請使用 'g'(抓取), 'b'(重置), 或 'q'(取消)。")
            return

        # 2. 建立目標的 PointStamped (原本在 map 座標系)
        target_map = PointStamped()
        target_map.header.frame_id = target_marker.header.frame_id # 通常是 'map'
        target_map.header.stamp = self.ros_communicator.get_clock().now().to_msg()
        target_map.point = target_marker.pose.position

        try:
            # 3. 將 map 上的網球，轉換到手臂基準座標系
            transform = self.tf_buffer.lookup_transform(
                self.base_link_name,
                target_map.header.frame_id,
                rclpy.time.Time()
            )
            target_base = tf2_geometry_msgs.do_transform_point(target_map, transform)
            
            x_target = target_base.point.x
            z_target = target_base.point.z
            
            print(f"🎯 目標相對基座座標: X={x_target:.3f}, Z={z_target:.3f}")

            # 🌟 4. 開啟背景執行緒，執行「抓取與緩慢歸位」的完整排程
            # 使用 daemon=True 確保程式關閉時執行緒會自動結束
            self.start_grab_sequence(x_target, z_target)

        except Exception as e:
            print(f"⚠️ 座標轉換或 TF 失敗: {e}")

    def trigger_auto_grasp_at_base(self, x_target=None, z_target=None):
        """Trigger the existing grasp sequence using arm_ik_base coordinates."""
        if x_target is None:
            x_target = self.default_grasp_x
        if z_target is None:
            z_target = self.default_grasp_z

        print(
            "[auto_grasp] Triggering grasp at arm_ik_base "
            f"X={x_target:.3f}, Z={z_target:.3f}"
        )
        self.start_grab_sequence(x_target, z_target)

    def trigger_joint_grasp_sequence(self, target_angles=None):
        """Trigger a grasp using fixed joint angles instead of IK x/z."""
        if target_angles is None:
            target_angles = self.joint_grasp_target_angles

        if len(target_angles) < 2:
            print("[joint_grasp] Need target angles [joint0, joint1]")
            return

        print(
            "[joint_grasp] Triggering fixed joint grasp "
            f"target=[{target_angles[0]:.1f}, {target_angles[1]:.1f}]"
        )
        self.start_joint_grasp_sequence(target_angles[:2])

    def trigger_task3_knob_sequence(self):
        if self.task3_knob_sequence_in_progress:
            return

        self.task3_knob_sequence_in_progress = True
        self.task3_knob_sequence_done = False
        threading.Thread(
            target=self._run_task3_knob_sequence_thread,
            daemon=True,
        ).start()

    def _run_task3_knob_sequence_thread(self):
        try:
            self._execute_task3_knob_sequence()
        except Exception as e:
            self.task3_knob_sequence_in_progress = False
            self.task3_knob_sequence_done = False
            print(f"[task3_arm] Door knob arm sequence failed: {e}")
            return

        self.task3_knob_sequence_in_progress = False
        self.task3_knob_sequence_done = True

    def _execute_task3_knob_sequence(self):
        target = self.task3_knob_ready_degrees
        print(
            "[task3_arm] Moving to knob ready pose: "
            f"elbow={target[0]:.1f}, wrist={target[1]:.1f}, finger={target[2]:.1f}"
        )
        self._smooth_move_user_servo_degrees(target, step=3.0, delay=0.05)
        self._wait_for_user_servo_degrees(target, timeout=3.0)
        print("[task3_arm] Door knob ready pose reached")

    def trigger_task3_elbow_push_sequence(self):
        if self.task3_elbow_push_in_progress:
            return

        self.task3_elbow_push_in_progress = True
        self.task3_elbow_push_done = False
        threading.Thread(
            target=self._run_task3_elbow_push_sequence_thread,
            daemon=True,
        ).start()

    def _run_task3_elbow_push_sequence_thread(self):
        try:
            self._execute_task3_elbow_push_sequence()
        except Exception as e:
            self.task3_elbow_push_in_progress = False
            self.task3_elbow_push_done = False
            print(f"[task3_arm] Door knob elbow push failed: {e}")
            return

        self.task3_elbow_push_in_progress = False
        self.task3_elbow_push_done = True

    def _execute_task3_elbow_push_sequence(self):
        target = self.task3_knob_push_degrees
        print(
            "[task3_arm] Moving elbow for knob push: "
            f"elbow={target[0]:.1f}, wrist={target[1]:.1f}, finger={target[2]:.1f}"
        )
        self._smooth_move_user_servo_degrees(target, step=2.0, delay=0.05)
        self._wait_for_user_servo_degrees(target, timeout=3.0)
        print("[task3_arm] Door knob elbow push pose reached")

    def _smooth_move_user_servo_degrees(self, target_degrees, step=2.0, delay=0.05):
        if not hasattr(self, "task3_user_servo_degrees"):
            self.task3_user_servo_degrees = list(self.current_user_servo_degrees)

        while True:
            all_reached = True
            for i, target in enumerate(target_degrees):
                diff = target - self.task3_user_servo_degrees[i]
                if abs(diff) <= step:
                    self.task3_user_servo_degrees[i] = target
                else:
                    self.task3_user_servo_degrees[i] += step if diff > 0 else -step
                    all_reached = False

            self._publish_user_servo_degrees(self.task3_user_servo_degrees)
            if all_reached:
                break
            time.sleep(delay)

    def _publish_user_servo_degrees(self, degrees):
        self.current_user_servo_degrees = list(degrees)
        joint_pos_radians = [math.radians(float(angle)) for angle in degrees]
        self.ros_communicator.publish_robot_arm_angle(joint_pos_radians)

    def _wait_for_user_servo_degrees(self, target_degrees, timeout=3.0, tolerance=5.0):
        start_time = time.time()
        saw_feedback = False
        while time.time() - start_time < timeout:
            feedback = self._get_user_servo_feedback_degrees(len(target_degrees))
            if feedback is None:
                time.sleep(0.1)
                continue

            saw_feedback = True
            if all(
                self._angle_error_deg(actual, target) <= tolerance
                for actual, target in zip(feedback, target_degrees)
            ):
                return True
            time.sleep(0.1)

        if not saw_feedback:
            print("[task3_arm] No /joint_states feedback; using timed arm wait")
            time.sleep(1.0)
            return True

        print("[task3_arm] Arm feedback did not reach target before timeout; continuing")
        return False

    def _get_user_servo_feedback_degrees(self, count):
        if not hasattr(self.ros_communicator, "get_latest_arm_joint_state"):
            return None

        joint_state = self.ros_communicator.get_latest_arm_joint_state()
        if joint_state is None or len(joint_state.position) < count:
            return None

        return [math.degrees(rad) % 360.0 for rad in joint_state.position[:count]]

    def _angle_error_deg(self, actual, target):
        return abs((actual - target + 180.0) % 360.0 - 180.0)

    def start_grab_sequence(self, x_target, z_target):
        self.grasp_in_progress = True
        self.grasp_done = False
        self.release_done = False
        threading.Thread(
            target=self._run_grab_sequence_thread,
            args=(x_target, z_target),
            daemon=True,
        ).start()

    def _run_grab_sequence_thread(self, x_target, z_target):
        try:
            self._execute_grab_sequence(x_target, z_target)
        except Exception as e:
            self.grasp_in_progress = False
            self.grasp_done = False
            print(f"[auto_grasp] Grasp sequence failed: {e}")
            return

        self.grasp_in_progress = False
        self.grasp_done = True

    def start_joint_grasp_sequence(self, target_angles):
        self.grasp_in_progress = True
        self.grasp_done = False
        self.release_done = False
        threading.Thread(
            target=self._run_joint_grasp_sequence_thread,
            args=(target_angles,),
            daemon=True,
        ).start()

    def _run_joint_grasp_sequence_thread(self, target_angles):
        try:
            self._execute_joint_grasp_sequence(target_angles)
        except Exception as e:
            self.grasp_in_progress = False
            self.grasp_done = False
            print(f"[joint_grasp] Fixed joint grasp sequence failed: {e}")
            return

        self.grasp_in_progress = False
        self.grasp_done = True

    def _execute_joint_grasp_sequence(self, target_angles):
        target_joint_0 = float(target_angles[0])
        target_joint_1 = float(target_angles[1])

        print("🔧 [joint 1/4] 打開夾爪...")
        target_open = [None, None, self.joint_limits[2]["max_angle"]]
        self._smooth_move_to(target_open, step=5.0, delay=0.1)
        time.sleep(0.5)

        print(
            "🤖 [joint 2/4] 平滑移動至指定關節角度: "
            f"J0={target_joint_0:.1f}, J1={target_joint_1:.1f}"
        )
        self._smooth_move_to([target_joint_0, target_joint_1, None], step=5.0, delay=0.1)
        time.sleep(0.5)

        print("✊ [joint 3/4] 夾取目標...")
        target_close = [None, None, self.joint_grasp_close_angle]
        self._smooth_move_to(target_close, step=5.0, delay=0.1)
        time.sleep(1.0)

        print("🏠 [joint 4/4] 緩慢回歸初始位置...")
        return_joint_0 = float(self.joint_grasp_return_angles[0])
        return_joint_1 = float(self.joint_grasp_return_angles[1])
        self._smooth_move_to([None, return_joint_1, None], step=5.0, delay=0.1)
        self._smooth_move_to([return_joint_0, None, None], step=5.0, delay=0.1)

        print("✅ 固定關節抓取任務完成！")
    
    def _execute_grab_sequence(self, x_target, z_target):
        """背景執行的完整抓取流程 (結合軌跡規劃)"""
        
        # 步驟 1：打開夾爪 (只有 Joint 2 移動，速度較快 step=5.0)
        print("🔧 [1/4] 打開夾爪...")
        target_open = [None, None, self.joint_limits[2]["max_angle"]]
        self._smooth_move_to(target_open, step=5.0, delay=0.1)
        time.sleep(0.5)

        # 步驟 2：手臂平滑移動到網球位置
        print("🤖 [2/4] 平滑移動至目標...")
        deg1, deg2 = self._calculate_2d_ik(x_target, z_target)
        # 手臂比較重，用較小的步長 (step=2.0) 慢慢移過去
        self._smooth_move_to([deg1, deg2, None], step=5.0, delay=0.1)
        time.sleep(0.5) # 等待手臂穩定

        # 步驟 3：夾住網球
        print("✊ [3/4] 夾取目標...")
        target_close = [None, None, self.joint_limits[2]["min_angle"]]
        self._smooth_move_to(target_close, step=5.0, delay=0.1)
        time.sleep(1.0) # 等待 Unity 的 FixedJoint 觸發黏合

        # 步驟 4：緩慢退回初始位置
        print("🏠 [4/4] 緩慢回歸初始位置...")
        init_angles = [None, self.joint_limits[1]["init"], None]
        self._smooth_move_to(init_angles, step=5.0, delay=0.1)
        init_angles = [self.joint_limits[0]["init"], None, None]
        self._smooth_move_to(init_angles, step=5.0, delay=0.1)
        
        print("✅ 抓取任務完美完成！")

    def _calculate_2d_ik(self, x, z):
        """計算 2D 逆向運動學，並回傳目標角度 (不直接移動手臂)"""
        L1 = self.joint_limits[0]["length"]
        L2 = self.joint_limits[1]["length"]
        
        D = math.sqrt(x**2 + z**2)
        if D > (L1 + L2):
            print("⚠️ 目標超出最遠抓取距離，以最大伸展姿態計算。")
            D = L1 + L2 - 0.001 
        
        cos_theta2 = (D**2 - L1**2 - L2**2) / (2 * L1 * L2)
        cos_theta2 = max(-1.0, min(1.0, cos_theta2)) 
        
        # 由上往下夾 (Elbow Up)
        theta2_rad = -math.acos(cos_theta2) 
        
        alpha = math.atan2(z, x)
        beta = math.acos((L1**2 + D**2 - L2**2) / (2 * L1 * D))
        theta1_rad = alpha + beta 
        
        # 轉換為 Degrees 並加上偏移量
        deg1 = math.degrees(theta1_rad) + self.joint_limits[0]["offset"]
        deg2 = math.degrees(theta2_rad) + self.joint_limits[1]["offset"]
        
        deg1 = self._normalize_angle(deg1, self.joint_limits[0]["min_angle"], self.joint_limits[0]["max_angle"])
        deg2 = self._normalize_angle(deg2, self.joint_limits[1]["min_angle"], self.joint_limits[1]["max_angle"])
        
        print(f"🧮 IK 計算完成: Shoulder={deg1:.1f}°, Elbow={deg2:.1f}°")
        return deg1, deg2
    
    def _smooth_move_to(self, target_angles, step=2.0, delay=0.05):
        """
        將手臂平滑地移動到目標角度 (Trajectory Interpolation)
        - target_angles: [j0_target, j1_target, j2_target] (若填 None 則該軸不動)
        - step: 每次更新的最大度數 (越小越滑順)
        - delay: 每次更新間隔的時間 (越大越慢)
        """
        while True:
            all_reached = True
            
            for i in range(len(self.joint_angles)):
                if target_angles[i] is None:
                    continue # None 代表該關節不移動
                    
                diff = target_angles[i] - self.joint_angles[i]
                
                if abs(diff) <= step:
                    self.joint_angles[i] = target_angles[i]
                else:
                    self.joint_angles[i] += step if diff > 0 else -step
                    all_reached = False # 只要有一個關節還沒到，就繼續迴圈
            
            # 發布這一小步的姿態
            self._clamp_and_publish()
            self._visualize_arm_lines()
            
            if all_reached:
                break
                
            time.sleep(delay)

    def _normalize_angle(self, angle, min_limit, max_limit):
        """
        嘗試加減 360 度，尋找是否有多轉或少轉一圈後，
        剛好能落入 [min_limit, max_limit] 物理極限內的同界角。
        """
        # 1. 將角度正規化到 0 ~ 360 的基準
        base_angle = angle % 360.0
        
        # 2. 準備三個候選角度：少一圈、當前(0~360)、多一圈
        candidates = [base_angle - 360.0, base_angle, base_angle + 360.0]
        
        # 3. 檢查哪一個落在合法範圍內
        for cand in candidates:
            if min_limit <= cand <= max_limit:
                return cand # 找到合法的同界角，直接回傳！
                
        # 如果加減 360 度後都不在範圍內，代表這個姿勢真的超出了手臂極限。
        # 我們先回傳原始角度，後續交給 _clamp_and_publish 去強制卡在邊界防呆。
        return angle
    # ==========================================
    # 5. 視覺化手臂 (Foxglove Lines)
    # ==========================================
    def _visualize_arm_lines(self):
        """根據當前的角度和長度，算出 3 個點的座標並發布 3D 視覺化線條"""
        L1 = self.joint_limits[0]["length"]
        L2 = self.joint_limits[1]["length"]
        
        # 扣除補償值，轉回純數學弧度，方便做正向運動學(FK)
        th1 = math.radians(self.joint_angles[0] - self.joint_limits[0]["offset"])
        th2 = math.radians(self.joint_angles[1] - self.joint_limits[1]["offset"])
        
        # 基座位置 P0
        p0 = Point(x=0.0, y=0.0, z=0.0)
        # 關節1位置 P1
        p1 = Point(x=L1 * math.cos(th1), y=0.0, z=L1 * math.sin(th1))
        # 夾爪末端位置 P2
        p2 = Point(x=p1.x + L2 * math.cos(th1 + th2), y=0.0, z=p1.z + L2 * math.sin(th1 + th2))
        
        # 建立 Marker
        marker = Marker()
        marker.header.frame_id = self.base_link_name
        marker.header.stamp = self.ros_communicator.get_clock().now().to_msg()
        marker.ns = "arm_kinematics"
        marker.id = 1
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        marker.scale.x = 0.02 # 線條粗細
        marker.color.a = 1.0  # 不透明度
        marker.color.r = 0.0  # 青藍色
        marker.color.g = 1.0
        marker.color.b = 1.0
        
        marker.points = [p0, p1, p2]
        
        self.ros_communicator.publish_arm_visual_lines(marker)

    def _clamp_and_publish(self):
        """確保所有數值在安全範圍內，並轉換為「弧度」後發布"""
        for i in range(len(self.joint_limits)):
            min_a = self.joint_limits[i]["min_angle"]
            max_a = self.joint_limits[i]["max_angle"]
            self.joint_angles[i] = max(min_a, min(max_a, self.joint_angles[i]))
            
        joint_pos_radians = [
            math.radians(float(self.joint_angles[i]) * self.joint_limits[i].get("dir", 1.0)) 
            for i in range(len(self.joint_angles))
        ]
        self.current_user_servo_degrees = [
            math.degrees(angle) % 360.0 for angle in joint_pos_radians
        ]
        self.ros_communicator.publish_robot_arm_angle(joint_pos_radians)
