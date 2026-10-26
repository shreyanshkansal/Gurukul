# Go2 real2sim pipeline

Stream depth from the Go2, run the depth policy on the PC using the real robot's
state and controller input, and replay the resulting actions in MuJoCo. The PC
relay reads the physical robot's state; the action receiver commands MuJoCo on
DDS domain 1 through loopback.

Run each command in a separate terminal, in the order below. On the PC, activate
the `gurukul` environment first. The Go2 needs its camera helper,
`shared_depth.py`, and RealSense dependencies; the PC needs the ONNX policy,
ONNX Runtime, Unitree SDK, and MuJoCo dependencies. Adjust the policy path,
network interface, and Go2 IP for your setup.

Ensure the Go2 is standing up using L2+A from the remote before running this.

## 1. Go2: start the depth camera server

```bash
cd ~/go2-depth/unitree-sim2real/RL_policy_runner/sim2real
python3 go2_depth_policy_action_relay.py --camera-server
```

## 2. PC: start MuJoCo

```bash
cd ~/Shreyansh/Gurukul
GURUKUL_DOMAIN_ID=1 GURUKUL_INTERFACE=lo GURUKUL_USE_JOYSTICK=0 \
    python unitree-sim2real/unitree_mujoco/simulate_python/unitree_mujoco.py \
    --robot go2 --scene flat
```

## 3. PC: start policy inference and action relay

```bash
cd ~/Shreyansh/Gurukul/unitree-sim2real/RL_policy_runner/real2sim
python3 go2_depth_policy_action_relay.py \
    --policy /home/psyduck/Shreyansh/Gurukul/policies/policy_derek_60.onnx \
    --interface enp11s0 \
    --depth-host 192.168.123.18 \
    --pc-ip 127.0.0.1 \
    --pc-port 15150
```

### Alternative: relay with a live depth preview

To display the normalized 58×87 depth image used by the policy, run this command
instead of the relay command above. It requires OpenCV with GUI support and a
desktop display. Run only one relay at a time; the other pipeline commands stay
the same.

```bash
cd ~/Shreyansh/Gurukul/unitree-sim2real/RL_policy_runner/real2sim
python3 go2_depth_policy_action_relay_depth_viz.py \
    --policy /home/psyduck/Shreyansh/Gurukul/policies/policy_derek_60.onnx \
    --interface enp11s0 \
    --depth-host 192.168.123.18 \
    --pc-ip 127.0.0.1 \
    --pc-port 15150
```

Optional display flags: `--depth-window-scale 6` and
`--depth-colormap turbo` (choices: `gray`, `turbo`, `jet`, `inferno`, `magma`).
Press Q or Esc in the preview window to stop the relay.

## 4. PC: start the action receiver

```bash
cd ~/Shreyansh/Gurukul/unitree-sim2real/RL_policy_runner/real2sim
python3 go2_action_receiver.py \
    --listen 127.0.0.1:15150 \
    --interface lo \
    --domain-id 1
```

The receiver waits for MuJoCo state and fresh policy actions, then applies the
relayed actions to the simulated robot.

## 5. Remote controller: give velocity commands to the policy
Press the left joystick forward to see the simulated robot move a bit. This serves as a sanity check to see if RL actions are not dangerous. Unlock the robot motors and let it tele-operate slowly to see the simulated robot behave similarly. The robot becomes unstable as the real and simulated observation spaces diverge.

No recording artifact is produced by
these commands. Stop the receiver with Ctrl+C before closing the other processes;
its shutdown commands the simulated motors passive.
