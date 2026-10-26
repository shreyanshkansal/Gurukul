"""Receive onboard policy actions and publish them to Unitree MuJoCo.

[Update] NOTE: The stand-up procedure has been commented out, start the simulated robot in a standing position instead.
Startup/control sequence mirrors go2_hardware_pos_obs45.py:

1. Wait for MuJoCo lowstate.
2. Execute the same 5-second stand-up trajectory used on hardware:
       stand_down_joint_pos -> stand_up_joint_pos
       500 Hz
       kp = phase * 50 + (1 - phase) * 20
       kd = 3.5
3. Switch to relayed policy actions:
       q_target = default_angles + action * action_scale
       200 Hz LowCmd
       kp = 20
       kd = 0.5
4. Incoming policy actions arrive independently over UDP at ~50 Hz.

This process controls ONLY the MuJoCo robot through the configured DDS domain.
"""

from __future__ import annotations

import argparse
import socket
import threading
import time

import numpy as np

from go2_depth_policy_action_relay import ACTION_PACKET, PACKET_MAGIC, PACKET_VERSION


def unpack_action(packet: bytes):
    """Decode relay actions without importing onboard camera dependencies."""
    if len(packet) != ACTION_PACKET.size:
        raise ValueError(f"Invalid action packet size {len(packet)} != {ACTION_PACKET.size}")
    magic, version, count, seq, timestamp_ns, *values = ACTION_PACKET.unpack(packet)
    if magic != PACKET_MAGIC or version != PACKET_VERSION or count != 12:
        raise ValueError("Invalid action packet header")
    actions = np.asarray(values, dtype=np.float32)
    if not np.isfinite(actions).all():
        raise ValueError("Action packet contains non-finite values")
    return int(seq), int(timestamp_ns), actions


def _import_sdk():
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelPublisher,
        ChannelSubscriber,
    )
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC

    return (
        ChannelFactoryInitialize,
        ChannelPublisher,
        ChannelSubscriber,
        LowCmd_,
        LowState_,
        unitree_go_msg_dds__LowCmd_,
        CRC,
    )


class ActionReceiver:
    def __init__(
        self,
        publisher,
        default_command,
        crc,
        max_age=0.15,
    ):
        self.publisher = publisher
        self.default_command = default_command
        self.crc = crc
        self.max_age = max_age

        # --------------------------------------------------------------
        # Policy action semantics -- same as working hardware controller.
        # --------------------------------------------------------------

        self.default_angles = np.array(
            [
                -0.1, 0.8, -1.5,   # FR
                 0.1, 0.8, -1.5,   # FL
                -0.1, 1.0, -1.5,   # RR
                 0.1, 1.0, -1.5,   # RL
            ],
            dtype=np.float32,
        )

        self.action_scales = np.full(
            12,
            0.25,
            dtype=np.float32,
        )

        self.action_scales[[0, 3, 6, 9]] = 0.125

        # --------------------------------------------------------------
        # Exact stand-up poses from go2_hardware_pos_obs45.py.
        # --------------------------------------------------------------

        self.stand_up_joint_pos = np.array(
            # [
            #     -0.00571868, 0.608813, -1.21763,
            #      0.00571868, 0.608813, -1.21763,
            #     -0.00571868, 0.608813, -1.21763,
            #      0.00571868, 0.608813, -1.21763,
            # ],

            [-0.019,  0.668, -1.369,  
             0.024,  0.666, -1.366, 
             -0.086,  0.659, -1.346,  
             0.082,  0.661, -1.352],
            dtype=np.float32,
        )

        self.stand_down_joint_pos = np.array(
            # [
            #     -0.0473455, 1.22187, -2.44375,
            #      0.0473455, 1.22187, -2.44375,
            #     -0.0473455, 1.22187, -2.44375,
            #      0.0473455, 1.22187, -2.44375,
            # ], 
            [
                -0.087,  1.232, -2.757,  
                0.058,  1.237, -2.797, 
                -0.394,  1.247, -2.780,  
                0.408,  1.237, -2.775
            ],
            dtype=np.float32,
        )

        # --------------------------------------------------------------
        # Latest incoming policy action.
        # --------------------------------------------------------------

        self.latest_actions = None
        self.latest_seq = None
        self.latest_at = None

        self.action_lock = threading.Lock()

        # --------------------------------------------------------------
        # MuJoCo feedback.
        # --------------------------------------------------------------

        self.mujoco_q = None
        self.mujoco_state_at = None

        # --------------------------------------------------------------
        # Control-thread state.
        # --------------------------------------------------------------

        self.running = True
        self.policy_enabled = False

        # --------------------------------------------------------------
        # Diagnostics.
        # --------------------------------------------------------------

        self.report_interval = 0.1
        self.last_report_at = None
        self.policy_start_at = None

    # ==================================================================
    # MuJoCo feedback
    # ==================================================================

    def lowstate_callback(self, msg):
        q = np.asarray(
            [motor.q for motor in msg.motor_state[:12]],
            dtype=np.float32,
        )

        if np.isfinite(q).all():
            self.mujoco_q = q
            self.mujoco_state_at = time.monotonic()

    # ==================================================================
    # LowCmd helpers
    # ==================================================================

    def new_command(self):
        command = self.default_command()

        command.head[0] = 0xFE
        command.head[1] = 0xEF
        command.level_flag = 0xFF
        command.gpio = 0

        return command

    def write_position_command(
        self,
        q_target,
        kp,
        kd,
    ):
        command = self.new_command()

        for i in range(12):
            motor = command.motor_cmd[i]

            motor.mode = 0x01

            motor.q = float(q_target[i])
            motor.kp = float(
                kp[i] if np.ndim(kp) else kp
            )

            motor.dq = 0.0
            motor.kd = float(
                kd[i] if np.ndim(kd) else kd
            )

            motor.tau = 0.0

        command.crc = self.crc.Crc(command)

        self.publisher.Write(command)

    def publish_passive(self):
        command = self.new_command()

        for motor in command.motor_cmd:
            motor.mode = 0x01
            motor.q = 0.0
            motor.kp = 0.0
            motor.dq = 0.0
            motor.kd = 0.0
            motor.tau = 0.0

        command.crc = self.crc.Crc(command)

        self.publisher.Write(command)

    # ==================================================================
    # Hardware-matching stand-up
    # ==================================================================

    def stand_up(self):
        """Reproduce go2_hardware_pos_obs45.py stand_up()."""

        print()
        print("Starting hardware-matching stand-up sequence.")
        print("  duration: 5.0 s")
        print("  LowCmd rate: 500 Hz")
        print("  kp: 20 -> 50")
        print("  kd: 3.5")
        print()

        running_time = 0.0

        step_start = time.perf_counter()

        next_report = 0.0

        while running_time < 5.0 and self.running:
            running_time += 0.002

            # EXACT phase equation used by working hardware runner.
            phase = np.tanh(
                running_time / 1.2
            )

            q_target = (
                phase * self.stand_up_joint_pos
                + (1.0 - phase)
                * self.stand_down_joint_pos
            )

            kp = (
                phase * 50.0
                + (1.0 - phase) * 20.0
            )

            self.write_position_command(
                q_target=q_target,
                kp=kp,
                kd=3.5,
            )

            # Diagnostic output every 0.5 s during stand-up.
            # if running_time >= next_report:
            #     print(
            #         f"[stand-up t={running_time:5.2f}s] "
            #         f"phase={phase:.3f}, "
            #         f"kp={kp:.2f}"
            #     )

            #     print(
            #         "  q_target: "
            #         + np.array2string(
            #             q_target,
            #             precision=3,
            #             suppress_small=True,
            #             floatmode="fixed",
            #         )
            #     )

            #     if self.mujoco_q is not None:
            #         error = q_target - self.mujoco_q

            #         print(
            #             "  actual q: "
            #             + np.array2string(
            #                 self.mujoco_q,
            #                 precision=3,
            #                 suppress_small=True,
            #                 floatmode="fixed",
            #             )
            #         )

            #         print(
            #             f"  max |error|="
            #             f"{np.max(np.abs(error)):.3f} rad, "
            #             f"mean |error|="
            #             f"{np.mean(np.abs(error)):.3f} rad"
            #         )

            #     next_report += 0.5

            # EXACT 500-Hz timing used by hardware stand-up.
            sleep_time = (
                0.002
                - (time.perf_counter() - step_start)
            )

            if sleep_time > 0:
                time.sleep(sleep_time)

            step_start = time.perf_counter()

        if not self.running:
            return

        print()
        print("Stand-up sequence complete.")

        if self.mujoco_q is not None:
            print(
                "Final stand-up actual q: "
                + np.array2string(
                    self.mujoco_q,
                    precision=3,
                    suppress_small=True,
                    floatmode="fixed",
                )
            )

        print()

    # ==================================================================
    # Incoming UDP policy actions
    # ==================================================================

    def receive_action(self, packet, received_at=None):
        seq, _, actions = unpack_action(packet)

        if seq % 50 == 0:
            print(
                f"RX seq={seq} "
                f"actions={np.array2string(actions, precision=9, max_line_width=1000)}",
                flush=True,
            )

        received_at = (
            time.monotonic()
            if received_at is None
            else received_at
        )

        if (
            self.latest_seq is not None
            and seq <= self.latest_seq
        ):
            raise ValueError(
                "Out-of-order action packet"
            )

        actions = np.asarray(
            actions,
            dtype=np.float32,
        )

        if not np.isfinite(actions).all():
            raise ValueError(
                "Non-finite action packet"
            )

        with self.action_lock:
            self.latest_actions = actions.copy()

        self.latest_seq = seq
        self.latest_at = received_at

    # ==================================================================
    # Policy LowCmd loop -- same gains/rate as working hardware
    # ==================================================================

    def policy_publish_loop(self):
        """Publish latest policy target at 200 Hz."""

        interval = 1.0 / 200.0
        next_tick = time.perf_counter()

        self.policy_enabled = True
        self.policy_start_at = time.monotonic()
        self.last_report_at = self.policy_start_at

        print("Policy control enabled.")
        print("  LowCmd rate: 200 Hz")
        print("  kp: 20.0")
        print("  kd: 0.5")
        print()

        while self.running:
            with self.action_lock:
                actions = (
                    None
                    if self.latest_actions is None
                    else self.latest_actions.copy()
                )

            if actions is not None:
                q_target = (
                    self.default_angles
                    + actions * self.action_scales
                )

                self.write_position_command(
                    q_target=q_target,
                    kp=20.0,
                    kd=0.5,
                )

                # self.maybe_report_policy(
                #     actions,
                #     q_target,
                # )

            next_tick += interval

            sleep_time = (
                next_tick
                - time.perf_counter()
            )

            if sleep_time > 0:
                time.sleep(sleep_time)

            else:
                next_tick = time.perf_counter()

    # ==================================================================
    # Policy diagnostics
    # ==================================================================

    def maybe_report_policy(
        self,
        actions,
        q_target,
    ):
        now = time.monotonic()

        if (
            self.last_report_at is None
            or now - self.last_report_at
            < self.report_interval
        ):
            return

        elapsed = now - self.policy_start_at

        print(
            f"\n[policy t={elapsed:6.3f}s, "
            f"seq={self.latest_seq}]"
        )

        print(
            "actions:  "
            + np.array2string(
                actions,
                precision=3,
                suppress_small=True,
                floatmode="fixed",
            )
        )

        print(
            "q_target: "
            + np.array2string(
                q_target,
                precision=3,
                suppress_small=True,
                floatmode="fixed",
            )
        )

        if (
            self.mujoco_q is not None
            and self.mujoco_state_at is not None
        ):
            error = (
                q_target
                - self.mujoco_q
            )

            error_deg = np.rad2deg(error)

            state_age = (
                now
                - self.mujoco_state_at
            )

            print(
                "actual q: "
                + np.array2string(
                    self.mujoco_q,
                    precision=3,
                    suppress_small=True,
                    floatmode="fixed",
                )
            )

            print(
                f"state age: "
                f"{state_age * 1000:.1f} ms"
            )

            print(
                "error [rad]: "
                + np.array2string(
                    error,
                    precision=3,
                    suppress_small=True,
                    floatmode="fixed",
                )
            )

            print(
                f"max |error|="
                f"{np.max(np.abs(error)):.3f} rad "
                f"({np.max(np.abs(error_deg)):.1f} deg), "
                f"mean |error|="
                f"{np.mean(np.abs(error)):.3f} rad "
                f"({np.mean(np.abs(error_deg)):.1f} deg)"
            )

        self.last_report_at = now

    # ==================================================================
    # Safety / freshness
    # ==================================================================

    def action_is_stale(self):
        if self.latest_at is None:
            return True

        return (
            time.monotonic()
            - self.latest_at
            > self.max_age
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--listen",
        default="0.0.0.0:15150",
    )

    parser.add_argument(
        "--interface",
        default="lo",
        help="DDS interface used by MuJoCo",
    )

    parser.add_argument(
        "--domain-id",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--max-age",
        type=float,
        default=0.15,
    )

    args = parser.parse_args(argv)

    host, separator, port = (
        args.listen.rpartition(":")
    )

    if (
        not separator
        or not host
        or not 1 <= int(port) <= 65535
    ):
        parser.error(
            "--listen must be HOST:PORT"
        )

    args.host = host
    args.port = int(port)

    return args


def main(argv=None):
    args = parse_args(argv)

    (
        ChannelFactoryInitialize,
        ChannelPublisher,
        ChannelSubscriber,
        LowCmd_,
        LowState_,
        default_command,
        CRC,
    ) = _import_sdk()

    # --------------------------------------------------------------
    # Connect ONLY to MuJoCo DDS.
    # --------------------------------------------------------------

    ChannelFactoryInitialize(
        args.domain_id,
        args.interface,
    )

    publisher = ChannelPublisher(
        "rt/lowcmd",
        LowCmd_,
    )

    publisher.Init()

    receiver = ActionReceiver(
        publisher=publisher,
        default_command=default_command,
        crc=CRC(),
        max_age=args.max_age,
    )

    # --------------------------------------------------------------
    # MuJoCo feedback.
    # --------------------------------------------------------------

    state_sub = ChannelSubscriber(
        "rt/lowstate",
        LowState_,
    )

    state_sub.Init(
        receiver.lowstate_callback,
        10,
    )

    print("Waiting for MuJoCo lowstate...")

    deadline = (
        time.monotonic() + 5.0
    )

    while receiver.mujoco_q is None:
        if time.monotonic() > deadline:
            raise RuntimeError(
                "Timed out waiting for "
                "MuJoCo rt/lowstate"
            )

        time.sleep(0.01)

    print("MuJoCo lowstate received.")

    print(
        "Initial MuJoCo q: "
        + np.array2string(
            receiver.mujoco_q,
            precision=3,
            suppress_small=True,
            floatmode="fixed",
        )
    )

    # --------------------------------------------------------------
    # UDP socket.
    #
    # Bind BEFORE stand-up so onboard packets can already arrive.
    # --------------------------------------------------------------

    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM,
    )

    sock.bind(
        (args.host, args.port)
    )

    sock.settimeout(0.02)

    print(
        f"Listening for "
        f"{ACTION_PACKET.size}-byte "
        f"action packets on "
        f"{args.host}:{args.port}"
    )

    # --------------------------------------------------------------
    # UDP receiver thread.
    #
    # Important: actions may arrive during stand-up, but they are only
    # STORED. They are not sent to MuJoCo until stand-up completes.
    # --------------------------------------------------------------

    udp_error = []

    def udp_loop():
        try:
            while receiver.running:
                try:
                    packet, _ = sock.recvfrom(
                        ACTION_PACKET.size + 1
                    )

                    receiver.receive_action(
                        packet
                    )

                except TimeoutError:
                    continue

        except Exception as exc:
            udp_error.append(exc)
            receiver.running = False

    udp_thread = threading.Thread(
        target=udp_loop,
        daemon=True,
    )

    udp_thread.start()

    policy_thread = None

    try:
        # ----------------------------------------------------------
        # Phase 1:
        # Exact hardware stand-up behavior.
        # ----------------------------------------------------------

        # receiver.stand_up()

        if not receiver.running:
            raise RuntimeError(
                "Receiver stopped during stand-up"
            )

        # ----------------------------------------------------------
        # Require a fresh policy action before handing control over.
        # ----------------------------------------------------------

        print(
            "Waiting for fresh onboard "
            "policy actions..."
        )

        action_deadline = (
            time.monotonic() + 5.0
        )

        while receiver.action_is_stale():
            if udp_error:
                raise RuntimeError(
                    f"UDP receiver failed: "
                    f"{udp_error[0]}"
                )

            if (
                time.monotonic()
                > action_deadline
            ):
                raise RuntimeError(
                    "Timed out waiting for fresh "
                    "onboard action stream"
                )

            time.sleep(0.01)

        print(
            "Fresh onboard action received."
        )

        # ----------------------------------------------------------
        # Phase 2:
        # Hardware-matching policy control.
        # ----------------------------------------------------------

        policy_thread = threading.Thread(
            target=receiver.policy_publish_loop,
            daemon=True,
        )

        policy_thread.start()

        # ----------------------------------------------------------
        # Supervisory loop.
        # ----------------------------------------------------------

        while receiver.running:
            if udp_error:
                raise RuntimeError(
                    f"UDP receiver failed: "
                    f"{udp_error[0]}"
                )

            if receiver.action_is_stale():
                raise RuntimeError(
                    "Onboard action stream is stale"
                )

            time.sleep(0.01)

    except RuntimeError as exc:
        print(f"\nSTOP: {exc}")

    finally:
        # ----------------------------------------------------------
        # Stop all active publishers BEFORE sending passive command.
        # ----------------------------------------------------------

        receiver.running = False

        if policy_thread is not None:
            policy_thread.join(
                timeout=1.0
            )

        udp_thread.join(
            timeout=1.0
        )

        # ----------------------------------------------------------
        # Passive simulated motors.
        # ----------------------------------------------------------

        receiver.publish_passive()

        sock.close()

        print(
            "Receiver stopped; "
            "MuJoCo motors commanded passive."
        )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\nInterrupted.")
