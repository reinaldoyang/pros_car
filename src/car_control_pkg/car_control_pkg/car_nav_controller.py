import copy
import time

from action_interface.action import NavGoal
from car_control_pkg.nav2_utils import cal_distance, calculate_diff_angle
from geometry_msgs.msg import PoseStamped


class NavigationController:
    # ------------------------------------------------------------------ #
    # Fixed poses for Post_Bridge_Nav — fill in actual map coordinates.   #
    # ------------------------------------------------------------------ #
    # Intermediate waypoint reached right after descending the bridge.
    INTERMEDIATE_POSE = {"x": 0.152, "y": 2.891, "qz": 1.000, "qw": -0.011}
    # Final home pose — fill in actual map coordinates.
    HOME_POSE = {"x": 0.0, "y": 0.0, "qz": 0.0, "qw": 1.0}

    # Recovery tuning
    NO_PATH_TIMEOUT_S = 2.0
    BACKUP_DURATION_S = 2.0
    GOAL_REPUBLISH_COOLDOWN_S = 3.0
    STUCK_MIN_MOVE_M = 0.08
    STUCK_TIMEOUT_S = 3.5

    def __init__(self, car_control_node):
        self.car_control_node = car_control_node
        self.nav_end_flag = 0
        self._reset_recovery_state()

    # ------------------------------------------------------------------
    # State-reset helpers
    # ------------------------------------------------------------------

    def _reset_path_state(self):
        """Reset path-following index and recovery timers (shared between legs)."""
        self.index = 0
        self.nav_end_flag = 0
        self._no_path_start = None
        self._backup_end_time = None
        self._goal_republish_at = 0.0
        self._last_known_position = None
        self._position_last_changed = None

    def _reset_recovery_state(self):
        """Full reset: path state + post-bridge phase. Called by reset_index()."""
        self._reset_path_state()
        self._post_bridge_phase = "intermediate"
        self._post_bridge_goal_set = False

    # ------------------------------------------------------------------
    # Recovery helpers (used by manual_nav)
    # ------------------------------------------------------------------

    def _do_backup(self, now):
        """
        Drive backward for BACKUP_DURATION_S.
        Returns True while still backing up, False when done.
        """
        if self._backup_end_time is None:
            self._backup_end_time = now + self.BACKUP_DURATION_S
            self.car_control_node.get_logger().warn(
                "Navigation recovery: driving backward"
            )
        if now < self._backup_end_time:
            self.car_control_node.publish_control("BACKWARD_SLOW")
            return True
        self._backup_end_time = None
        return False

    def _republish_goal(self, now):
        """Re-publish the stored goal to force Nav2 to replan from new position."""
        if now - self._goal_republish_at < self.GOAL_REPUBLISH_COOLDOWN_S:
            return
        latest = self.car_control_node.latest_goal_pose
        if latest is None:
            return
        goal_msg = copy.deepcopy(latest)
        goal_msg.header.stamp = self.car_control_node.get_clock().now().to_msg()
        self.car_control_node.goal_clear_pub.publish(goal_msg)
        self._goal_republish_at = now
        self.car_control_node.get_logger().warn(
            "Navigation recovery: republished goal to trigger Nav2 replan"
        )

    def _is_stuck(self, car_position, now):
        """True if position has not changed by STUCK_MIN_MOVE_M in STUCK_TIMEOUT_S."""
        if self._last_known_position is None:
            self._last_known_position = list(car_position)
            self._position_last_changed = now
            return False
        if cal_distance(car_position, self._last_known_position) > self.STUCK_MIN_MOVE_M:
            self._last_known_position = list(car_position)
            self._position_last_changed = now
            return False
        return (now - self._position_last_changed) > self.STUCK_TIMEOUT_S

    # ------------------------------------------------------------------
    # Fixed-goal publisher (used by post_bridge_nav)
    # ------------------------------------------------------------------

    def _publish_fixed_goal(self, pose_dict):
        """
        Publish a PoseStamped to /goal_pose and update latest_goal_pose so
        manual_nav can use it immediately on the same tick.
        Also clears the stale plan so the no-path timer starts fresh.
        """
        msg = PoseStamped()
        msg.header.stamp = self.car_control_node.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x = float(pose_dict["x"])
        msg.pose.position.y = float(pose_dict["y"])
        msg.pose.position.z = 0.0
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = float(pose_dict.get("qz", 0.0))
        msg.pose.orientation.w = float(pose_dict.get("qw", 1.0))

        self.car_control_node.goal_clear_pub.publish(msg)
        self.car_control_node.latest_goal_pose = msg
        self.car_control_node.clear_plan()   # Nav2 will publish a fresh plan shortly
        self.car_control_node.get_logger().info(
            f"Published fixed goal: x={pose_dict['x']}, y={pose_dict['y']}"
        )

    # ------------------------------------------------------------------
    # Public interface (called by NavigationActionServer)
    # ------------------------------------------------------------------

    def check_prerequisites(self):
        car_position, car_orientation = (
            self.car_control_node.get_car_position_and_orientation()
        )
        path_points = self.car_control_node.get_path_points()
        goal_pose = self.car_control_node.get_goal_pose()

        if not car_position or not path_points or not goal_pose:
            message = (
                "Cannot obtain car position data"
                if not car_position
                else (
                    "No path points available for navigation"
                    if not path_points
                    else "No goal pose defined for navigation"
                )
            )
            return NavGoal.Result(success=False, message=message)

        return car_position, car_orientation, path_points, goal_pose

    def data_init(self, car_position, car_orientation, goal_pose):
        return (
            [car_position.x, car_position.y],
            [car_orientation.z, car_orientation.w],
            [goal_pose.x, goal_pose.y],
        )

    def reset_index(self):
        self._reset_recovery_state()

    # ------------------------------------------------------------------
    # Navigation modes
    # ------------------------------------------------------------------

    def post_bridge_nav(self):
        """
        Two-leg Nav2 navigation executed after descending the bridge.
          Leg 1 — navigate to INTERMEDIATE_POSE (clears the bridge area).
          Leg 2 — navigate to HOME_POSE.
        Edit INTERMEDIATE_POSE and HOME_POSE at the top of this class.
        """
        if self._post_bridge_phase == "intermediate":
            if not self._post_bridge_goal_set:
                self._publish_fixed_goal(self.INTERMEDIATE_POSE)
                self._post_bridge_goal_set = True
                self.car_control_node.get_logger().info(
                    "Post-bridge leg 1: navigating to intermediate pose"
                )
                return

            result = self.manual_nav()
            if not isinstance(result, NavGoal.Result):
                return  # still navigating

            if not result.success:
                return result  # propagate failure (e.g. fatal missing data)

            # Leg 1 complete — prepare leg 2
            self.car_control_node.get_logger().info(
                "Post-bridge leg 1 complete: heading to home pose"
            )
            self._post_bridge_phase = "home"
            self._post_bridge_goal_set = False
            self._reset_path_state()   # fresh index + timers; phase kept intact

        elif self._post_bridge_phase == "home":
            if not self._post_bridge_goal_set:
                self._publish_fixed_goal(self.HOME_POSE)
                self._post_bridge_goal_set = True
                self.car_control_node.get_logger().info(
                    "Post-bridge leg 2: navigating to home pose"
                )
                return

            return self.manual_nav()

    def customize_nav(self):
        result = self.check_prerequisites()
        coordinate = self.car_control_node.get_latest_object_coordinates()
        if coordinate == {} or not coordinate:
            if self.nav_end_flag == 0:
                self.signal = self.manual_nav()
            else:
                if self.nav_end_flag == 1:
                    self.car_control_node.clear_plan()
                    self.car_control_node.clear_goal_pose()
                    self.car_control_node.publish_control("COUNTERCLOCKWISE_ROTATION_SLOW")
        else:
            self.nav_end_flag = 0
            y_offset = coordinate["ball"][1]
            object_depth = coordinate["ball"][0]
            if object_depth < 0.3:
                for i in range(10):
                    self.car_control_node.publish_control("STOP")
                    time.sleep(0.1)
                self.car_control_node.clear_plan()
                self.car_control_node.clear_goal_pose()
                return NavGoal.Result(
                    success=True,
                    message="Navigation goal reached successfully. Final distance",
                )
            action = self.choose_action_y_offset(y_offset, object_depth)
            self.car_control_node.publish_control(action)

    def choose_action_y_offset(self, y_offset, object_depth):
        if object_depth >= 0.5:
            limit = 0.5
        elif object_depth <= 0.5:
            limit = 0.1
        if y_offset > -limit and y_offset < limit:
            return "FORWARD_SLOW"
        elif y_offset >= limit:
            return "COUNTERCLOCKWISE_ROTATION_SLOW"
        elif y_offset <= -limit:
            return "CLOCKWISE_ROTATION_SLOW"

    def manual_nav(self):
        now = time.monotonic()
        result = self.check_prerequisites()

        if isinstance(result, NavGoal.Result):
            if "No path" not in result.message:
                return result  # fatal: no position or no goal

            # --- No-path recovery ---
            if self._no_path_start is None:
                self._no_path_start = now
                self.car_control_node.get_logger().warn(
                    "No path available; will start backward recovery if it persists"
                )

            if now - self._no_path_start < self.NO_PATH_TIMEOUT_S:
                self.car_control_node.publish_control("STOP")
                return

            if self._do_backup(now):
                return

            # Backup done: request replan and reset timer so we wait again
            self._no_path_start = None
            self._republish_goal(now)
            return

        # --- Normal navigation ---
        car_position_msg, car_orientation_msg, path_points, goal_pose = result
        self._no_path_start = None

        car_position, car_orientation, goal_position = self.data_init(
            car_position_msg, car_orientation_msg, goal_pose
        )

        target_distance = cal_distance(car_position, goal_position)
        if target_distance < 0.5:
            self.nav_end_flag = 1
            self.car_control_node.publish_control("STOP")
            return NavGoal.Result(
                success=True,
                message="Navigation goal reached successfully. Final distance",
            )

        # --- Stuck detection ---
        if self._is_stuck(car_position, now):
            self.car_control_node.get_logger().warn(
                "Robot stuck; initiating backward recovery"
            )
            if self._do_backup(now):
                return
            self._last_known_position = None
            self._position_last_changed = None
            self.car_control_node.clear_plan()
            self._republish_goal(now)
            return

        # --- Path following ---
        target_points, orientation_points = self.get_next_target_point(
            car_position=car_position, path_points=path_points
        )
        diff_angle = calculate_diff_angle(car_position, car_orientation, target_points)
        action_key = self.choose_action(diff_angle)
        self.car_control_node.publish_control(action_key)

    def choose_action(self, diff_angle):
        if diff_angle < 20 and diff_angle > -20:
            action_key = "FORWARD"
        elif diff_angle < -20 and diff_angle > -180:
            action_key = "CLOCKWISE_ROTATION"
        elif diff_angle > 20 and diff_angle < 180:
            action_key = "COUNTERCLOCKWISE_ROTATION"
        return action_key

    def get_next_target_point(
        self, car_position, path_points, min_required_distance=0.5
    ):
        logger = self.car_control_node.get_logger()

        if not path_points:
            logger.error("Error: No path points available!")
            return None, None

        if not hasattr(self, "index"):
            self.index = 0

        for idx in range(self.index, len(path_points)):
            point = path_points[idx]
            try:
                pos = point["position"]
                orient = point["orientation"]
                target_x, target_y = pos[0], pos[1]
                orientation_x, orientation_y = orient[0], orient[1]
            except (KeyError, IndexError, TypeError) as e:
                logger.error(f"Invalid path point format at index {idx}: {e}")
                continue

            distance_to_target = cal_distance(car_position, (target_x, target_y))
            if distance_to_target >= min_required_distance:
                self.index = idx
                logger.debug(
                    f"Found valid target point at index {idx} with distance {distance_to_target:.2f}"
                )
                return [target_x, target_y], [orientation_x, orientation_y]
            else:
                logger.debug(
                    f"Skipping point at index {idx}: distance {distance_to_target:.2f} is less than required {min_required_distance}"
                )

        try:
            last_point = path_points[-1]
            pos = last_point["position"]
            orient = last_point["orientation"]
            last_x, last_y = pos[0], pos[1]
            last_ox, last_oy = orient[0], orient[1]
            logger.info(
                "No point met the minimum distance requirement; using the last point as target."
            )
            self.index = len(path_points) - 1
            return [last_x, last_y], [last_ox, last_oy]
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Invalid format for last path point: {e}")

        logger.warning("No valid target point found.")
        return None, None
