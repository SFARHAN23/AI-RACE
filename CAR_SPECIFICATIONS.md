# AIRACE RC Car Specifications and Mathematical Model

This file records the car/simulator specifications used by the included AIRACE TD3 model.
The values are taken from `src/rc_car_model.py` and the training/rollout environment.

## Physical car / simulator constants

| Quantity | Value | Notes |
|---|---:|---|
| Scale/model type | 1/10 RC car | Kinematic bicycle model |
| Mass | 3.5 kg | `CAR_MASS` |
| Length | 0.45 m | `CAR_LENGTH` |
| Width | 0.20 m | `CAR_WIDTH` |
| Wheelbase | 0.3302 m | `CAR_WHEELBASE` |
| Front length from CG | 0.1651 m | wheelbase / 2 |
| Rear length from CG | 0.1651 m | wheelbase / 2 |
| Maximum speed | 8.0 m/s | 28.8 km/h |
| Maximum steering angle | 0.4189 rad | about 24.0 degrees |
| Maximum steering rate | 0.4189 rad/s | full lock in about 1 second |
| Control timestep | 0.02 s | 50 Hz control loop |
| Gravity | 9.81 m/s² | used for rolling friction |
| Rolling friction coefficient | 0.02 | simplified rolling resistance |
| Track/lane width | 0.60 m | simulation track width |

## Acceleration, braking, force, and torque

| Quantity | Value / Equation |
|---|---:|
| Max throttle acceleration | 8.0 m/s² |
| Max braking deceleration | 12.0 m/s² |
| Full-throttle drive force | mass × acceleration = 3.5 × 8.0 = 28.0 N |
| Full-brake force | mass × deceleration = 3.5 × 12.0 = 42.0 N |
| Rolling-friction force while moving | coefficient × mass × gravity = 0.02 × 3.5 × 9.81 = 0.687 N |
| Net full-throttle acceleration while moving | (28.0 - 0.687) / 3.5 = 7.804 m/s² |
| Net coasting deceleration from rolling friction | 0.687 / 3.5 = 0.196 m/s² |
| Wheel torque from drive force | torque = drive_force × wheel_radius |
| Example torque if wheel radius is 0.05 m | 28.0 × 0.05 = 1.40 N·m total at ground contact |
| Example brake torque if wheel radius is 0.05 m | 42.0 × 0.05 = 2.10 N·m total at ground contact |

Important: the current simulator stores acceleration/force directly, not a measured motor torque curve.
If the real wheel radius is measured, replace the example radius above and recompute:

```text
drive_torque_Nm = 28.0 * wheel_radius_m
brake_torque_Nm = 42.0 * wheel_radius_m
```

## LiDAR / observation specs

| Quantity | Value |
|---|---:|
| LiDAR beams | 20 |
| Field of view | -135° to +135° |
| Minimum range | 0.05 m |
| Maximum range | 12.0 m |
| Base observation size | 23 values | speed, yaw rate, steering, 20 LiDAR beams |
| Extended observation size | 36 values | base 23 + IMU/memory features |
| Action size | 2 values | throttle/brake and steering, both normalized to [-1, 1] |

## Control mapping

The TD3 policy outputs two normalized values:

```text
action[0] = throttle/brake command in [-1, 1]
action[1] = steering command in [-1, 1]
```

Longitudinal force:

```text
if throttle >= 0:
    force = throttle * 8.0 * 3.5
else:
    force = throttle * 12.0 * 3.5
```

Steering:

```text
target_steer = steering_command * 0.4189 rad
max_steer_change_per_step = 0.4189 * 0.02 = 0.008378 rad/step
```

## Kinematic bicycle equations

State:

```text
x, y       = car position
psi        = heading angle
v          = speed
delta      = steering angle
dt         = 0.02 s
L          = 0.3302 m wheelbase
Lr         = 0.1651 m rear length from center of gravity
```

Slip angle:

```text
beta = atan((Lr / L) * tan(delta))
```

Yaw rate:

```text
psi_dot = v * cos(beta) / L * tan(delta)
```

Position update:

```text
x_next   = x + v * cos(psi + beta) * dt
y_next   = y + v * sin(psi + beta) * dt
psi_next = psi + psi_dot * dt
```

Longitudinal acceleration update:

```text
friction_force = 0.02 * 3.5 * 9.81  # while moving
net_force = drive_or_brake_force - sign(speed) * friction_force
longitudinal_acceleration = net_force / 3.5
speed_next = clip(speed + longitudinal_acceleration * 0.02, 0.0, 8.0)
```

Lateral acceleration approximation:

```text
radius = Lr / sin(beta)
lateral_acceleration = speed² / radius
```

## Reward/profile values used in the AIRACE training setup

| Reward term | Value |
|---|---:|
| Target speed | 0.90 m/s |
| Progress weight | 1.15 |
| Progress clip | 1.50 |
| Reverse weight | 2.50 |
| Reverse bias | 0.25 |
| Forward alive bonus | 0.06 |
| Stable speed bonus | 0.03 |
| Low-speed penalty | 0.55 |
| Brake penalty | 0.12 |
| High-speed threshold | 1.60 m/s |
| High-speed penalty | 0.06 |
| Wall penalty | 0.45 |
| Wall penalty starts at | 55% of half lane width = 0.165 m from center |
| Steering penalty | 0.025 |
| Steering delta penalty | 0.16 |
| Throttle delta penalty | 0.08 |
| Yaw-rate penalty | 0.030 |
| Collision terminal reward | -18.0 |
| Wrong-direction terminal reward | -18.0 |
| Stopped terminal reward | -28.0 |
| Finished terminal reward | +220.0 |
| Time-limit penalty | -20.0 × remaining progress fraction |
| Sector bonus | 0.25 |
| Intermediate lap bonus | +45.0 |

Core progress calculation:

```text
target_progress_per_step = target_speed * dt = 0.90 * 0.02 = 0.018 m
progress_rate = step_progress_delta / 0.018
progress_reward = 1.15 * clip(progress_rate, 0, 1.5)
maximum_progress_reward_per_step = 1.725
```
