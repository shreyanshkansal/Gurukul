"""Run a 45-D Go2 locomotion policy onboard and relay actions to a PC.

This process reads the physical Go2's lowstate and wireless-controller input,
runs a 45-D ONNX locomotion policy, and sends the resulting 12 raw policy
actions over UDP.

It deliberately never publishes rt/lowcmd and therefore does not command the
physical Go2. It is intended for hardware-in-the-loop action replay with a
Unitree MuJoCo bridge running on another computer.
"""

from __future__ import annotations

import argparse
import socket
import struct
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


# ---------------------------------------------------------------------------
# UDP packet format
# Must match go2_depth_action_receiver.py /
# go2_depth_student_action_relay.py.
# ---------------------------------------------------------------------------

PACKET_MAGIC = 0x47324441  # G2DA
PACKET_VERSION = 1
ACTION_PACKET = struct.Struct("!IHHQQ12f")


def pack_action(seq: int, timestamp_ns: int, actions: np.ndarray) -> bytes:
    actions = np.asarray(actions, dtype=np.float32).reshape(-1)

    if actions.shape != (12,) or not np.isfinite(actions).all():
        raise ValueError("Relay actions must be 12 finite float32 values")

    if seq < 0 or timestamp_ns < 0:
        raise ValueError("Sequence and timestamp must be nonnegative")

    return ACTION_PACKET.pack(
        PACKET_MAGIC,
        PACKET_VERSION,
        12,
        int(seq),
        int(timestamp_ns),
        *actions,
    )


# ---------------------------------------------------------------------------
# Unitree imports
# Subscriber only: there is intentionally no ChannelPublisher / LowCmd.
# ---------------------------------------------------------------------------

def _import_sdk():
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelSubscriber,
    )
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import (
        LowState_,
        WirelessController_,
    )

    return (
        ChannelFactoryInitialize,
        ChannelSubscriber,
        LowState_,
        WirelessController_,
    )


def quat_to_rot(quat):
    w, x, y, z = quat

    return np.asarray(
        [
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - w * z),
                2 * (x * z + w * y),
            ],
            [
                2 * (x * y + w * z),
                1 - 2 * (x * x + z * z),
                2 * (y * z - w * x),
            ],
            [
                2 * (x * z - w * y),
                2 * (y * z + w * x),
                1 - 2 * (x * x + y * y),
            ],
        ],
        dtype=np.float32,
    )


class ActionRelay:
    def __init__(self, args, actor, udp_socket):
        self.args = args
        self.actor = actor
        self.udp_socket = udp_socket

        self.actor_input_name = actor.get_inputs()[0].name
        self.actor_output_name = actor.get_outputs()[0].name

        self.low_state = None
        self.state_at = None
        self.tick = None

        # Same initial remote state as the working hardware runner:
        # zero command until remote events are received.
        self.remote = np.zeros(4, dtype=np.float32)
        self.keys = 0

        self.previous_action = np.zeros((1, 12), dtype=np.float32)

        self.gravity = np.array(
            [0.0, 0.0, -1.0],
            dtype=np.float32,
        )

        # Same FR, FL, RR, RL order and defaults as
        # go2_hardware_pos_obs45.py.
        self.default_angles = np.array(
            [
                -0.1, 0.8, -1.5,   # FR
                 0.1, 0.8, -1.5,   # FL
                -0.1, 1.0, -1.5,   # RR
                 0.1, 1.0, -1.5,   # RL
            ],
            dtype=np.float32,
        )

        self.action_scales = np.full(12, 0.25, dtype=np.float32)
        self.action_scales[[0, 3, 6, 9]] = 0.125

        self.sequence = 0

        self.last_report_at = time.monotonic()
        self.steps_since_report = 0

    # ------------------------------------------------------------------
    # DDS callbacks
    # ------------------------------------------------------------------

    def state_callback(self, msg):
        # Ignore duplicate robot ticks.
        if self.tick == msg.tick:
            return

        self.tick = msg.tick
        self.low_state = msg
        self.state_at = time.monotonic()

    def remote_callback(self, msg):
        values = np.asarray(
            [msg.lx, msg.ly, msg.rx, msg.ry],
            dtype=np.float32,
        )

        if not np.isfinite(values).all():
            return

        self.remote = values
        self.keys = int(msg.keys)

    def stopped_by_remote(self):
        # R1 + L1.
        return bool(self.keys & 1) and bool(self.keys & 2)

    # ------------------------------------------------------------------
    # Same 45-D proprioception as go2_hardware_pos_obs45.py
    # ------------------------------------------------------------------

    def observation(self):
        if self.low_state is None or self.state_at is None:
            raise RuntimeError("No Go2 lowstate received")

        state_age = time.monotonic() - self.state_at

        if state_age > self.args.max_state_age:
            raise RuntimeError(
                f"Go2 lowstate is stale: {state_age:.3f}s"
            )

        if self.stopped_by_remote():
            raise RuntimeError("R1+L1 relay stop requested")

        state = self.low_state

        # IMU quaternion.
        quat = np.asarray(
            state.imu_state.quaternion,
            dtype=np.float32,
        )

        if (
            not np.isfinite(quat).all()
            or not 0.95 <= np.linalg.norm(quat) <= 1.05
        ):
            raise RuntimeError("Invalid Go2 IMU quaternion")

        # 3: angular velocity, same 0.25 scaling as baseline.
        ang_vel = np.asarray(
            state.imu_state.gyroscope[:3],
            dtype=np.float32,
        ) * 0.25

        # 3: projected gravity.
        projected_gravity = quat_to_rot(quat).T @ self.gravity

        # 3: velocity command.
        #
        # Baseline:
        #   lin_vel_x = j_ly
        #   lin_vel_y = j_lx
        #   ang_vel_z = -j_rx
        command = np.array(
            [
                self.remote[1],
                self.remote[0],
                -self.remote[2],
            ],
            dtype=np.float32,
        )

        command = np.clip(command, -1.0, 1.0)

        # 12 joint positions and velocities.
        positions = np.asarray(
            [motor.q for motor in state.motor_state[:12]],
            dtype=np.float32,
        )

        velocities = np.asarray(
            [motor.dq for motor in state.motor_state[:12]],
            dtype=np.float32,
        )

        if (
            not np.isfinite(positions).all()
            or not np.isfinite(velocities).all()
        ):
            raise RuntimeError("Non-finite Go2 motor state")

        # Same ordering as go2_hardware_pos_obs45.py:
        #
        #   3 angular velocity
        #   3 projected gravity
        #   3 command
        #   12 joint-position offsets
        #   12 scaled joint velocities
        #   12 previous action
        #
        # Total = 45.
        obs = np.concatenate(
            (
                ang_vel,
                projected_gravity,
                command,
                positions - self.default_angles,
                velocities * 0.05,
                self.previous_action[0],
            )
        ).reshape(1, 45)

        obs = obs.astype(np.float32, copy=False)

        if not np.isfinite(obs).all():
            raise RuntimeError("Non-finite policy observation")

        return obs

    # ------------------------------------------------------------------
    # Policy + UDP
    # ------------------------------------------------------------------

    def step(self):
        obs = self.observation()

        actions = self.actor.run(
            [self.actor_output_name],
            {self.actor_input_name: obs},
        )[0]

        actions = np.asarray(actions, dtype=np.float32)

        if actions.shape != (1, 12):
            raise RuntimeError(
                f"Expected policy action shape (1, 12), got {actions.shape}"
            )

        if not np.isfinite(actions).all():
            raise RuntimeError("Policy produced non-finite actions")

        raw_actions = actions[0]

        packet = pack_action(
            self.sequence,
            time.time_ns(),
            raw_actions,
        )

        if self.sequence % 50 == 0:
            print(
                f"TX seq={self.sequence} "
                f"actions={np.array2string(raw_actions, precision=9, max_line_width=1000)}",
                flush=True,
            )

        self.udp_socket.sendto(
            packet,
            self.args.destination,
        )

        self.previous_action[0] = raw_actions
        self.sequence += 1
        self.steps_since_report += 1

        # Human-readable report approximately once per second.
        now = time.monotonic()
        elapsed = now - self.last_report_at

        if elapsed >= 1.0:
            command = obs[0, 6:9]

            # Actual joint positions of the physical Go2.
            physical_q = np.asarray(
                [motor.q for motor in self.low_state.motor_state[:12]],
                dtype=np.float32,
            )

            # Exactly what the known-working hardware controller would command.
            q_target = (
                self.default_angles
                + raw_actions * self.action_scales
            )

            print(
                f"\n[{self.sequence:6d}] "
                f"rate={self.steps_since_report / elapsed:5.1f} Hz"
            )

            print(
                "command [vx vy wz]: "
                f"[{command[0]: .3f}, "
                f"{command[1]: .3f}, "
                f"{command[2]: .3f}]"
            )

            print(
                "physical q: "
                + np.array2string(
                    physical_q,
                    precision=3,
                    suppress_small=True,
                    floatmode="fixed",
                )
            )

            # print(
            #     "raw actions: "
            #     + np.array2string(
            #         raw_actions,
            #         precision=3,
            #         suppress_small=True,
            #         floatmode="fixed",
            #     )
            # )

            # print(
            #     "q_target:   "
            #     + np.array2string(
            #         q_target,
            #         precision=3,
            #         suppress_small=True,
            #         floatmode="fixed",
            #     )
            # )

            # print(
            #     f"action range: "
            #     f"[{raw_actions.min(): .3f}, {raw_actions.max(): .3f}], "
            #     f"mean |a|={np.mean(np.abs(raw_actions)):.3f}"
            # )

            self.last_report_at = now
            self.steps_since_report = 0

        return obs, raw_actions


def load_policy(path: Path, threads: int):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1

    actor = ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    inputs = actor.get_inputs()
    outputs = actor.get_outputs()

    if (
        len(inputs) != 1
        or inputs[0].shape != [1, 45]
        or inputs[0].type != "tensor(float)"
    ):
        raise ValueError(
            "45-D locomotion policy requires one float32 input [1, 45]"
        )

    if (
        len(outputs) != 1
        or outputs[0].shape != [1, 12]
        or outputs[0].type != "tensor(float)"
    ):
        raise ValueError(
            "Locomotion policy requires one float32 output [1, 12]"
        )

    print(
        f"Loaded policy: {path}\n"
        f"  input:  {inputs[0].name} {inputs[0].shape}\n"
        f"  output: {outputs[0].name} {outputs[0].shape}"
    )

    return actor


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--policy",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--interface",
        required=True,
        help="Go2 DDS interface; this program never publishes lowcmd",
    )

    parser.add_argument(
        "--pc-ip",
        required=True,
    )

    parser.add_argument(
        "--pc-port",
        type=int,
        default=15150,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--max-state-age",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="0 means run until Ctrl+C or R1+L1",
    )

    args = parser.parse_args(argv)

    if not args.policy.is_file():
        parser.error(f"Policy not found: {args.policy}")

    if not 1 <= args.pc_port <= 65535:
        parser.error("Invalid PC port")

    if args.duration < 0:
        parser.error("--duration cannot be negative")

    if args.threads <= 0:
        parser.error("--threads must be positive")

    if (
        not np.isfinite(args.max_state_age)
        or args.max_state_age <= 0
    ):
        parser.error("--max-state-age must be finite and positive")

    args.destination = (
        args.pc_ip,
        args.pc_port,
    )

    return args


def main(argv=None):
    args = parse_args(argv)

    (
        ChannelFactoryInitialize,
        ChannelSubscriber,
        LowState_,
        WirelessController_,
    ) = _import_sdk()

    # Physical Go2 DDS.
    ChannelFactoryInitialize(
        0,
        args.interface,
    )

    actor = load_policy(
        args.policy,
        args.threads,
    )

    udp_socket = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM,
    )

    relay = ActionRelay(
        args,
        actor,
        udp_socket,
    )

    state_sub = ChannelSubscriber(
        "rt/lowstate",
        LowState_,
    )

    remote_sub = ChannelSubscriber(
        "rt/wirelesscontroller",
        WirelessController_,
    )

    state_sub.Init(
        relay.state_callback,
        10,
    )

    remote_sub.Init(
        relay.remote_callback,
        10,
    )

    # Avoid the startup race we encountered with the depth relay.
    print("Waiting for Go2 lowstate...")

    deadline_state = time.monotonic() + 5.0

    while relay.low_state is None:
        if time.monotonic() > deadline_state:
            raise RuntimeError(
                "Timed out waiting for Go2 lowstate"
            )

        time.sleep(0.01)

    print("Go2 lowstate received.")
    print(
        f"Relaying raw 12-D actions to "
        f"{args.pc_ip}:{args.pc_port} at 50 Hz."
    )
    print(
        "No rt/lowcmd publisher exists in this process; "
        "the physical Go2 is not commanded."
    )

    deadline = (
        time.monotonic() + args.duration
        if args.duration
        else None
    )

    try:
        while (
            deadline is None
            or time.monotonic() < deadline
        ):
            started = time.monotonic()

            relay.step()

            # Same policy rate as the baseline:
            # 200-Hz motor loop / decimation 4 = 50 Hz.
            time.sleep(
                max(
                    0.0,
                    0.02 - (time.monotonic() - started),
                )
            )

    finally:
        udp_socket.close()


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print(
            "Relay stopped; no motor command "
            "was published to the physical Go2."
        )

    except RuntimeError as exc:
        print(f"Relay stopped: {exc}")