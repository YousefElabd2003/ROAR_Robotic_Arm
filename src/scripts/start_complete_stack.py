#!/usr/bin/env python3

from __future__ import annotations

import os
import shutil
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
ENV_PREFIX = (
    "unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH && "
    f"cd {shlex.quote(str(WORKSPACE_ROOT))} && "
    "source /opt/ros/humble/setup.bash && "
    "source install/setup.bash"
)

KILL_PATTERNS = [
    "ros2 launch sixdof_moveit complete.launch.py",
    "ros2 launch sixdof_pkg gazebo.launch.py",
    "python3 src/scripts/teleop.py",
    "python3 src/scripts/workspace.py",
    "python3 src/scripts/workspace_checker.py",
    "python3 src/scripts/end_effector_service.py",
    "python3 src/scripts/planning_scene_refresher.py",
    "move_group",
    "rviz2",
    "robot_state_publisher",
    "ros_gz_bridge",
    "spawner",
    "ign gazebo",
]

CHILDREN: list[tuple[str, subprocess.Popen]] = []
RVIZ_SOURCE = WORKSPACE_ROOT / "src" / "sixdof_moveit" / "config" / "moveit.rviz"
RVIZ_INSTALL = WORKSPACE_ROOT / "install" / "sixdof_moveit" / "share" / "sixdof_moveit" / "config" / "moveit.rviz"


def run_shell(command: str) -> int:
    return subprocess.run(["bash", "-lc", command], check=False).returncode


def kill_lingering_processes() -> None:
    for pattern in KILL_PATTERNS:
        run_shell(f"pkill -f {shlex.quote(pattern)} || true")
    time.sleep(1.0)


def sync_rviz_config() -> None:
    if not RVIZ_SOURCE.exists():
        print(f"RViz source config not found: {RVIZ_SOURCE}")
        return
    RVIZ_INSTALL.parent.mkdir(parents=True, exist_ok=True)
    try:
        if RVIZ_INSTALL.exists() and RVIZ_SOURCE.samefile(RVIZ_INSTALL):
            print(f"[sync] rviz config already current: {RVIZ_INSTALL}")
            return
    except FileNotFoundError:
        pass
    shutil.copy2(RVIZ_SOURCE, RVIZ_INSTALL)
    print(f"[sync] rviz config: {RVIZ_SOURCE} -> {RVIZ_INSTALL}")


def spawn_process(name: str, command: str) -> subprocess.Popen:
    proc = subprocess.Popen(
        ["bash", "-lc", f"{ENV_PREFIX} && {command}"],
        preexec_fn=os.setsid,
    )
    CHILDREN.append((name, proc))
    print(f"[start] {name}: pid={proc.pid}")
    return proc


def stop_children() -> None:
    for _, proc in reversed(CHILDREN):
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
    time.sleep(1.5)
    for _, proc in reversed(CHILDREN):
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass


def main() -> int:
    print(f"Workspace: {WORKSPACE_ROOT}")
    print("Killing lingering ROS, Gazebo, RViz, MoveIt, and script processes...")
    kill_lingering_processes()
    sync_rviz_config()

    try:
        complete = spawn_process("complete_launch", "ros2 launch sixdof_moveit complete.launch.py")
        time.sleep(12.0)
        if complete.poll() is not None:
            print("complete.launch.py exited early")
            return complete.returncode or 1

        spawn_process("teleop", "python3 src/scripts/teleop.py")
        time.sleep(1.0)

        spawn_process("static_tf", "python3 src/scripts/static_world_to_base.py")
        time.sleep(0.5)

        spawn_process("ref_broadcaster", "python3 src/scripts/reference_frame_broadcaster.py")
        time.sleep(2.0)

        spawn_process("workspace", "python3 src/scripts/workspace.py")

        # NEW: start the end effector service
        spawn_process("ee_service", "python3 src/scripts/end_effector_service.py")
        time.sleep(1.0)

        # NEW: optional planning scene refresher to keep RViz synced
        spawn_process("scene_refresh", "python3 src/scripts/planning_scene_refresher.py")

        print("All processes started. Press Ctrl+C to stop everything.")
        while True:
            time.sleep(1.0)
            if complete.poll() is not None:
                print("complete.launch.py exited; stopping child processes.")
                return complete.returncode or 1
    except KeyboardInterrupt:
        print("Stopping all launched processes...")
        return 0
    finally:
        stop_children()


if __name__ == "__main__":
    sys.exit(main())
