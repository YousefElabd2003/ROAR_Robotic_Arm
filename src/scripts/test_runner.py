#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import statistics
import time
from datetime import datetime

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, PositionConstraint
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from collision_guard import CollisionGuard

LINK_NAME = "link_6"
GROUP_NAME = "arm_controller"
FRAME_ID = "world"
PITCH_JOINT = "joint_4"
TWIST_JOINT = "joint_5"
LOCK_TOL = 0.05

HOME_JOINTS = {
    "joint_0": math.radians(0.0),
    "joint_1": math.radians(0.0),
    "joint_2": math.radians(0.0),
    "joint_3": math.radians(0.0),
    "joint_4": math.radians(0.0),
    "joint_5": math.radians(0.0),
}

DIRECTION_MAP = {
    "w": (+1, 0, 0),
    "s": (-1, 0, 0),
    "a": (0, +1, 0),
    "d": (0, -1, 0),
    "q": (0, 0, +1),
    "e": (0, 0, -1),
}

AXIS_LABEL = {
    "w": "+X forward",
    "s": "-X back",
    "a": "+Y left",
    "d": "-Y right",
    "q": "+Z up",
    "e": "-Z down",
}


class ArmTestRunner(Node):
    def __init__(self, planning_time: float, planning_attempts: int, vel_scale: float, acc_scale: float):
        super().__init__("arm_test_runner")
        self.move_client = ActionClient(self, MoveGroup, "move_action")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.current_joints: dict[str, float] = {}
        self.js_received = False
        self.create_subscription(JointState, "joint_states", self._js_cb, 10)
        self.collision_guard = CollisionGuard(self, GROUP_NAME)

        self.planning_time = planning_time
        self.planning_attempts = planning_attempts
        self.vel_scale = vel_scale
        self.acc_scale = acc_scale

        self.locked = False
        self.locked_pitch = 0.0
        self.locked_twist = 0.0

    def _js_cb(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self.current_joints[name] = pos
        self.js_received = True

    def wait_for_joint_states(self, timeout: float = 5.0) -> bool:
        t0 = time.time()
        while not self.js_received:
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - t0 > timeout:
                return False
        return True

    def wait_for_move_server(self, timeout: float = 5.0) -> bool:
        return self.move_client.wait_for_server(timeout_sec=timeout)

    def _tf_ee_world(self):
        try:
            t = self.tf_buffer.lookup_transform(
                FRAME_ID,
                LINK_NAME,
                rclpy.time.Time(),
                timeout=Duration(seconds=1.0),
            )
            p = t.transform.translation
            return p.x, p.y, p.z
        except Exception:
            return None

    def _joint_constraint(self, joint_name: str, position: float, tol: float = 0.03):
        jc = JointConstraint()
        jc.joint_name = joint_name
        jc.position = position
        jc.tolerance_above = tol
        jc.tolerance_below = tol
        jc.weight = 1.0
        return jc

    def _pos_constraint(self, x: float, y: float, z: float, tol: float = 0.01):
        pc = PositionConstraint()
        pc.header.frame_id = FRAME_ID
        pc.link_name = LINK_NAME
        pc.weight = 1.0
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [tol, tol, tol]
        pc.constraint_region.primitives.append(box)
        tp = PoseStamped()
        tp.header.frame_id = FRAME_ID
        tp.pose.position.x = x
        tp.pose.position.y = y
        tp.pose.position.z = z
        tp.pose.orientation.w = 1.0
        pc.constraint_region.primitive_poses.append(tp.pose)
        return pc

    def _lock_constraints(self):
        if not self.locked:
            return []
        out = []
        for jn, jv in [(PITCH_JOINT, self.locked_pitch), (TWIST_JOINT, self.locked_twist)]:
            jc = JointConstraint()
            jc.joint_name = jn
            jc.position = jv
            jc.tolerance_above = LOCK_TOL
            jc.tolerance_below = LOCK_TOL
            jc.weight = 1.0
            out.append(jc)
        return out

    def _execute_constraints(self, constraints: Constraints):
        ok, reason = self.collision_guard.check_current_state(self.current_joints)
        if not ok:
            return False, -400, 0.0

        goal = MoveGroup.Goal()
        goal.request.group_name = GROUP_NAME
        goal.request.allowed_planning_time = self.planning_time
        goal.request.num_planning_attempts = self.planning_attempts
        goal.request.max_velocity_scaling_factor = self.vel_scale
        goal.request.max_acceleration_scaling_factor = self.acc_scale
        goal.request.goal_constraints.append(constraints)

        t0 = time.time()
        goal_future = self.move_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, goal_future, timeout_sec=self.planning_time + 2.0)
        if not goal_future.done() or goal_future.result() is None:
            return False, -100, time.time() - t0

        goal_handle = goal_future.result()
        if not goal_handle.accepted:
            return False, -101, time.time() - t0

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=self.planning_time + 5.0)
        if not result_future.done() or result_future.result() is None:
            return False, -102, time.time() - t0

        code = int(result_future.result().result.error_code.val)
        return code == 1, code, time.time() - t0

    def move_home(self):
        c = Constraints()
        for jn in ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5"]:
            c.joint_constraints.append(self._joint_constraint(jn, HOME_JOINTS[jn], tol=0.03))
        return self._execute_constraints(c)

    def set_lock_from_current(self):
        if PITCH_JOINT not in self.current_joints or TWIST_JOINT not in self.current_joints:
            return False
        self.locked_pitch = self.current_joints[PITCH_JOINT]
        self.locked_twist = self.current_joints[TWIST_JOINT]
        self.locked = True
        return True

    def clear_lock(self):
        self.locked = False

    def move_xyz(self, direction_key: str, step_cm: float):
        ee = self._tf_ee_world()
        if ee is None:
            return False, -200, 0.0
        cx, cy, cz = ee
        sx, sy, sz = DIRECTION_MAP[direction_key]
        step_m = step_cm / 100.0

        tx = cx + sx * step_m
        ty = cy + sy * step_m
        tz = cz + sz * step_m

        c = Constraints()
        c.position_constraints.append(self._pos_constraint(tx, ty, tz, tol=0.01))
        c.joint_constraints.extend(self._lock_constraints())
        return self._execute_constraints(c)


def parse_csv_list(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def safe_mean(values):
    return statistics.mean(values) if values else 0.0


def safe_stdev(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def run():
    parser = argparse.ArgumentParser(description="Automated motion reliability tester")
    parser.add_argument("--trials", type=int, default=10, help="Trials per case")
    parser.add_argument("--steps", type=str, default="1,2,5", help="Step sizes in cm, comma-separated")
    parser.add_argument("--directions", type=str, default="w,s,a,d,q,e", help="Direction keys, comma-separated")
    parser.add_argument("--modes", type=str, default="unlocked,locked", help="Modes: unlocked,locked")
    parser.add_argument("--home-only", action="store_true", help="Only test reaching HOME repeatedly")
    parser.add_argument("--home-trials", type=int, default=20, help="Trials for --home-only mode")
    parser.add_argument("--retries", type=int, default=2, help="Extra retries after initial failed move")
    parser.add_argument("--fallback-splits", type=int, default=2, help="How many step halvings on repeated failure")
    parser.add_argument("--fallback-factor", type=float, default=0.5, help="Step reduction factor per fallback split")
    parser.add_argument("--planning-time", type=float, default=3.0)
    parser.add_argument("--planning-attempts", type=int, default=10)
    parser.add_argument("--vel-scale", type=float, default=0.3)
    parser.add_argument("--acc-scale", type=float, default=0.3)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    steps = [float(x) for x in parse_csv_list(args.steps)]
    directions = parse_csv_list(args.directions)
    modes = parse_csv_list(args.modes)

    for d in directions:
        if d not in DIRECTION_MAP:
            raise ValueError(f"Unsupported direction key: {d}")
    for m in modes:
        if m not in ["locked", "unlocked"]:
            raise ValueError(f"Unsupported mode: {m}")

    rclpy.init()
    node = ArmTestRunner(
        planning_time=args.planning_time,
        planning_attempts=args.planning_attempts,
        vel_scale=args.vel_scale,
        acc_scale=args.acc_scale,
    )

    if not node.wait_for_joint_states():
        print("ERROR: No joint states received")
        rclpy.shutdown()
        return
    if not node.wait_for_move_server():
        print("ERROR: move_action server unavailable")
        rclpy.shutdown()
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output.strip()
    if not output_path:
        os.makedirs("test_reports", exist_ok=True)
        output_path = f"test_reports/arm_test_report_{ts}.txt"

    lines = []
    lines.append("ROVER Arm Test Report")
    lines.append(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Trials per case: {args.trials}")
    lines.append(f"Retries per move: {args.retries}")
    lines.append(f"Fallback splits: {args.fallback_splits}")
    lines.append(f"Fallback factor: {args.fallback_factor}")
    lines.append(f"Steps (cm): {steps}")
    lines.append(f"Directions: {directions}")
    lines.append(f"Modes: {modes}")
    lines.append(
        f"Planner: time={args.planning_time}s attempts={args.planning_attempts} vel={args.vel_scale} acc={args.acc_scale}"
    )
    lines.append("")

    trial_rows = []
    case_stats = {}

    def attempt_move_with_retries(direction: str, base_step_cm: float):
        step_cm = base_step_cm
        total_attempt_index = 0

        for split_idx in range(args.fallback_splits + 1):
            for local_try in range(args.retries + 1):
                total_attempt_index += 1
                ok_move, code_move, dt_move = node.move_xyz(direction, step_cm)
                if ok_move:
                    return {
                        "ok": True,
                        "code": code_move,
                        "time_s": dt_move,
                        "attempts": total_attempt_index,
                        "used_step_cm": step_cm,
                        "split_idx": split_idx,
                    }
            step_cm *= args.fallback_factor

        return {
            "ok": False,
            "code": code_move,
            "time_s": dt_move,
            "attempts": total_attempt_index,
            "used_step_cm": step_cm,
            "split_idx": args.fallback_splits,
        }

    if args.home_only:
        home_times = []
        home_ok = 0
        lines.append("")
        lines.append("HOME-only sanity mode")
        lines.append(f"Trials: {args.home_trials}")

        for i in range(1, args.home_trials + 1):
            ok_home, code_home, dt_home = node.move_home()
            if ok_home:
                home_ok += 1
                home_times.append(dt_home)
            trial_rows.append({
                "trial": i,
                "mode": "home-only",
                "dir": "-",
                "step_cm": 0.0,
                "attempt": i,
                "result": "OK" if ok_home else "FAIL",
                "code": code_home,
                "time_s": dt_home,
                "attempts_until_result": 1,
                "effective_step_cm": 0.0,
                "fallback_level": 0,
            })
            print(f"[HOME {i:03d}] {'OK' if ok_home else 'FAIL'} code={code_home} t={dt_home:.2f}s")

        total = args.home_trials
        rate = 100.0 * home_ok / total if total else 0.0
        lines.append(f"HOME success rate: {home_ok}/{total} = {rate:.1f}%")
        lines.append(f"HOME mean time: {safe_mean(home_times):.3f}s")
        lines.append(f"HOME stdev time: {safe_stdev(home_times):.3f}s")

    else:
        trial_id = 0
        for mode in modes:
            for step_cm in steps:
                for direction in directions:
                    case_key = (mode, step_cm, direction)
                    case_stats[case_key] = {
                        "ok": 0,
                        "total": 0,
                        "times": [],
                        "attempts": [],
                        "fallback_used": 0,
                    }

                    for n in range(1, args.trials + 1):
                        trial_id += 1
                        ok_home, code_home, dt_home = node.move_home()
                        if not ok_home:
                            trial_rows.append({
                                "trial": trial_id,
                                "mode": mode,
                                "dir": direction,
                                "step_cm": step_cm,
                                "attempt": n,
                                "result": "HOME_FAIL",
                                "code": code_home,
                                "time_s": dt_home,
                                "attempts_until_result": 1,
                                "effective_step_cm": step_cm,
                                "fallback_level": 0,
                            })
                            case_stats[case_key]["total"] += 1
                            continue

                        if mode == "locked":
                            if not node.set_lock_from_current():
                                trial_rows.append({
                                    "trial": trial_id,
                                    "mode": mode,
                                    "dir": direction,
                                    "step_cm": step_cm,
                                    "attempt": n,
                                    "result": "LOCK_FAIL",
                                    "code": -300,
                                    "time_s": 0.0,
                                    "attempts_until_result": 1,
                                    "effective_step_cm": step_cm,
                                    "fallback_level": 0,
                                })
                                case_stats[case_key]["total"] += 1
                                continue
                        else:
                            node.clear_lock()

                        res = attempt_move_with_retries(direction, step_cm)
                        case_stats[case_key]["total"] += 1
                        case_stats[case_key]["attempts"].append(res["attempts"])
                        if res["split_idx"] > 0:
                            case_stats[case_key]["fallback_used"] += 1
                        if res["ok"]:
                            case_stats[case_key]["ok"] += 1
                            case_stats[case_key]["times"].append(res["time_s"])

                        trial_rows.append({
                            "trial": trial_id,
                            "mode": mode,
                            "dir": direction,
                            "step_cm": step_cm,
                            "attempt": n,
                            "result": "OK" if res["ok"] else "FAIL",
                            "code": res["code"],
                            "time_s": res["time_s"],
                            "attempts_until_result": res["attempts"],
                            "effective_step_cm": res["used_step_cm"],
                            "fallback_level": res["split_idx"],
                        })

                        print(
                            f"[{trial_id:03d}] {mode:8s} {direction} ({AXIS_LABEL[direction]:10s}) {step_cm:>4.1f}cm "
                            f"-> {'OK' if res['ok'] else 'FAIL'} code={res['code']} t={res['time_s']:.2f}s "
                            f"tries={res['attempts']} eff_step={res['used_step_cm']:.2f}cm split={res['split_idx']}"
                        )

    lines.append("Per-trial results")
    lines.append("trial | mode | dir | step_cm | try | result | code | time_s | attempts | effective_step_cm | fallback_level")
    for row in trial_rows:
        lines.append(
            f"{row['trial']:03d} | {row['mode']} | {row['dir']} | {row['step_cm']:.1f} | {row['attempt']} | "
            f"{row['result']} | {row['code']} | {row['time_s']:.3f} | "
            f"{row['attempts_until_result']} | {row['effective_step_cm']:.2f} | {row['fallback_level']}"
        )

    if not args.home_only:
        lines.append("")
        lines.append("Summary by case")
        lines.append("mode | dir | step_cm | success_rate | mean_time_s | stdev_time_s | mean_attempts | fallback_rate")
        for (mode, step_cm, direction), st in case_stats.items():
            total = st["total"]
            ok = st["ok"]
            rate = (100.0 * ok / total) if total else 0.0
            mean_t = safe_mean(st["times"])
            stdev_t = safe_stdev(st["times"])
            mean_attempts = safe_mean(st["attempts"])
            fallback_rate = (100.0 * st["fallback_used"] / total) if total else 0.0
            lines.append(
                f"{mode} | {direction} ({AXIS_LABEL[direction]}) | {step_cm:.1f} | {rate:6.1f}% | "
                f"{mean_t:.3f} | {stdev_t:.3f} | {mean_attempts:.2f} | {fallback_rate:.1f}%"
            )

        lines.append("")
        lines.append("Top stable cases (success>=90%, sorted by largest step then fastest)")
        good_cases = []
        for (mode, step_cm, direction), st in case_stats.items():
            total = st["total"]
            if total == 0:
                continue
            rate = 100.0 * st["ok"] / total
            if rate >= 90.0:
                good_cases.append((mode, direction, step_cm, rate, safe_mean(st["times"])))
        good_cases.sort(key=lambda x: (-x[2], x[4]))
        if good_cases:
            for mode, direction, step_cm, rate, mean_t in good_cases[:12]:
                lines.append(
                    f"{mode} | {direction} ({AXIS_LABEL[direction]}) | {step_cm:.1f} cm | {rate:.1f}% | {mean_t:.3f}s"
                )
        else:
            lines.append("None")

    # txt report
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # csv report
    csv_path = output_path.replace(".txt", ".csv") if output_path.endswith(".txt") else output_path + ".csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trial",
                "mode",
                "dir",
                "step_cm",
                "attempt",
                "result",
                "code",
                "time_s",
                "attempts_until_result",
                "effective_step_cm",
                "fallback_level",
            ],
        )
        writer.writeheader()
        for row in trial_rows:
            writer.writerow(row)

    # json summary
    json_path = output_path.replace(".txt", ".json") if output_path.endswith(".txt") else output_path + ".json"
    json_payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "trials": args.trials,
            "steps": steps,
            "directions": directions,
            "modes": modes,
            "home_only": args.home_only,
            "home_trials": args.home_trials,
            "retries": args.retries,
            "fallback_splits": args.fallback_splits,
            "fallback_factor": args.fallback_factor,
            "planning_time": args.planning_time,
            "planning_attempts": args.planning_attempts,
            "vel_scale": args.vel_scale,
            "acc_scale": args.acc_scale,
        },
        "trial_count": len(trial_rows),
        "case_stats": {},
    }
    for (mode, step_cm, direction), st in case_stats.items():
        key = f"{mode}:{direction}:{step_cm:.1f}"
        total = st["total"]
        ok = st["ok"]
        json_payload["case_stats"][key] = {
            "mode": mode,
            "direction": direction,
            "axis_label": AXIS_LABEL[direction],
            "step_cm": step_cm,
            "total": total,
            "ok": ok,
            "success_rate": (100.0 * ok / total) if total else 0.0,
            "mean_time_s": safe_mean(st["times"]),
            "stdev_time_s": safe_stdev(st["times"]),
            "mean_attempts": safe_mean(st["attempts"]),
            "fallback_rate": (100.0 * st["fallback_used"] / total) if total else 0.0,
        }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)

    print(f"\nSaved report: {output_path}")
    print(f"Saved report: {csv_path}")
    print(f"Saved report: {json_path}")

    rclpy.shutdown()


if __name__ == "__main__":
    run()
