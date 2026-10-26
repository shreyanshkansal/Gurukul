# Go2 blind real2sim pipeline

Run the blind ONNX policy using the physical Go2's state and controller input,
then replay its actions in MuJoCo. Choose inference on the PC or onboard the Go2.
The relay reads the physical robot's state; the receiver commands MuJoCo on DDS
domain 1 through loopback. No depth camera server is needed.

Activate the `gurukul` environment on the PC. The inference machine needs the
blind ONNX policy, NumPy, ONNX Runtime, and Unitree SDK; the PC also needs the
MuJoCo dependencies and Unitree SDK. Adjust policy paths, interfaces, and the PC
IP address for your setup.

Ensure the Go2 is standing up using L2+A from the remote before running this.
Run each process in a separate terminal: start MuJoCo, start one relay option,
then start the matching receiver command.

## 1. PC: start MuJoCo (both options)

```bash
cd ~/Shreyansh/Gurukul
GURUKUL_DOMAIN_ID=1 GURUKUL_INTERFACE=lo GURUKUL_USE_JOYSTICK=0 \
    python unitree-sim2real/unitree_mujoco/simulate_python/unitree_mujoco.py \
    --robot go2 --scene standing
```

## 2A. Inference on the PC

Use the bundled blind relay on the PC, with actions sent to the local receiver:

```bash
cd ~/Shreyansh/Gurukul/unitree-sim2real/RL_policy_runner/real2sim
python3 go2_policy_action_relay.py \
    --policy /home/psyduck/Shreyansh/Gurukul/policies/policy_sood.onnx \
    --interface enp11s0 \
    --pc-ip 127.0.0.1 \
    --pc-port 15150
```

Start the receiver in another PC terminal:

```bash
cd ~/Shreyansh/Gurukul/unitree-sim2real/RL_policy_runner/real2sim
python3 go2_action_receiver.py \
    --listen 127.0.0.1:15150 \
    --interface lo \
    --domain-id 1
```

## 2B. Inference onboard the Go2

Use the deployed relay on the Go2, with actions sent to the PC's network address:

```bash
cd ~/go2-depth/unitree-sim2real/RL_policy_runner/sim2real
python3 go2_policy_action_relay.py \
    --policy /home/unitree/go2-depth/policies/policy_sood.onnx \
    --interface eth10 \
    --pc-ip 192.168.123.51 \
    --pc-port 15150
```

Start the receiver on the PC, listening for UDP actions from the Go2:

```bash
cd ~/Shreyansh/Gurukul/unitree-sim2real/RL_policy_runner/real2sim
python3 go2_action_receiver.py \
    --listen 0.0.0.0:15150 \
    --interface lo \
    --domain-id 1
```

Run only one relay and one receiver at a time. The receiver waits for MuJoCo
state and fresh policy actions, then applies the actions to the simulated robot.

## 3. Remote controller: give velocity commands to the policy

Press the left joystick forward to see the simulated robot move a bit. The
simulated trajectory can diverge because policy observations come from the
physical robot rather than the simulated robot. This replay is a visual sanity
check, not a guarantee of hardware safety.

No recording artifact is produced by these commands. Stop the receiver with
Ctrl+C before closing the other processes; its shutdown commands the simulated
motors passive.
