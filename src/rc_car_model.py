"""
rc_car_model.py — 1/10-scale kinematic bicycle model for RC car simulation.

Simplified drivetrain (no aerodynamics at RC scale).  State is updated at
50 Hz with rate-limited steering and clipped speed.
"""

import numpy as np

# ── Shared constants ─────────────────────────────────────────────────────────
CAR_MASS = 3.5          # kg
CAR_WHEELBASE = 0.3302  # m
CAR_LENGTH = 0.45       # m
CAR_WIDTH = 0.20        # m
CAR_MAX_SPEED = 8.0     # m/s
CAR_MAX_STEER = 0.4189  # rad (~24 deg)
CAR_MAX_STEER_RATE = CAR_MAX_STEER  # rad/s (full lock in 1 s)
DT = 0.02               # s (50 Hz control loop)

TRACK_WIDTH = 0.60      # m

N_BEAMS = 20
LIDAR_ANGLE_MIN = -2.3562   # -135 deg in rad
LIDAR_ANGLE_MAX = 2.3562    # +135 deg in rad
LIDAR_MAX_RANGE = 12.0      # m
LIDAR_MIN_RANGE = 0.05      # m

# ── Car model parameters ─────────────────────────────────────────────────────
GravityAcc = 9.81

CarWheelBase = CAR_WHEELBASE
CarLenF = CarWheelBase / 2.0   # 0.1651 m (50/50 front/rear split)
CarLenR = CarWheelBase / 2.0   # 0.1651 m
CarMass = CAR_MASS
CarLength = CAR_LENGTH
CarWidth = CAR_WIDTH
MaxSteer = CAR_MAX_STEER
MaxSteerRate = CAR_MAX_STEER_RATE
SPD_MAX = CAR_MAX_SPEED
SPD_MIN = 0.0

# Simplified drivetrain (no aero at RC scale)
K_THROTTLE = 8.0          # m/s² max acceleration at full throttle
K_BRAKE = 12.0            # m/s² max deceleration at full brake
ROLLING_FRICTION = 0.02   # coefficient


class RCCarModelClass:
    """1/10-scale kinematic bicycle model."""

    def __init__(self, pose0, spd0):
        """
        Parameters
        ----------
        pose0 : list/array  [x, y, psi]
        spd0  : float       initial speed (m/s)
        """
        self.pose = np.array(pose0, dtype=np.float64)   # [x, y, psi]
        self.spd = float(spd0)
        self.steer = 0.0      # current steering angle (rad)
        self.psi_dot = 0.0    # yaw rate (rad/s)

        self.long_acc = 0.0   # longitudinal acceleration (m/s²)
        self.lat_acc = 0.0    # lateral acceleration (m/s²)

        # Internal
        self._force = 0.0     # longitudinal force (N)

    # ── reset ─────────────────────────────────────────────────────────────
    def reset(self, pose0, spd0):
        """Reset car to a given state."""
        self.pose = np.array(pose0, dtype=np.float64)
        self.spd = float(spd0)
        self.steer = 0.0
        self.psi_dot = 0.0
        self.long_acc = 0.0
        self.lat_acc = 0.0
        self._force = 0.0

    # ── step ──────────────────────────────────────────────────────────────
    def step(self, action):
        """
        Advance by one time-step (DT).

        Parameters
        ----------
        action : array-like  [ux, uy] each in [-1, 1]
            ux — throttle (+) / brake (-)
            uy — steering command (+ = left, - = right)
        """
        ux = float(np.clip(action[0], -1.0, 1.0))
        uy = float(np.clip(action[1], -1.0, 1.0))

        self._convert_control(ux, uy)
        self._longitudinal_dynamic()
        self._lateral_kinematic()
        self._update_pose()

    # ── convert control ───────────────────────────────────────────────────
    def _convert_control(self, ux, uy):
        """Map normalised inputs to force and steering."""
        # Longitudinal
        if ux >= 0:
            self._force = ux * K_THROTTLE * CarMass
        else:
            self._force = ux * K_BRAKE * CarMass

        # Steering (rate-limited)
        target_steer = uy * MaxSteer
        steer_diff = target_steer - self.steer
        max_delta = MaxSteerRate * DT
        steer_diff = np.clip(steer_diff, -max_delta, max_delta)
        self.steer = np.clip(self.steer + steer_diff, -MaxSteer, MaxSteer)

    # ── longitudinal dynamics ─────────────────────────────────────────────
    def _longitudinal_dynamic(self):
        """Update speed from force and rolling friction."""
        # Rolling friction opposes motion
        if self.spd > 1e-4:
            friction_force = ROLLING_FRICTION * CarMass * GravityAcc
        else:
            friction_force = 0.0

        net_force = self._force - np.sign(self.spd) * friction_force
        self.long_acc = net_force / CarMass
        self.spd += self.long_acc * DT
        self.spd = float(np.clip(self.spd, SPD_MIN, SPD_MAX))

    # ── lateral kinematics (bicycle model) ────────────────────────────────
    def _lateral_kinematic(self):
        """Compute slip angle, yaw rate, lateral acceleration."""
        if abs(self.steer) < 1e-6:
            self.psi_dot = 0.0
            self.lat_acc = 0.0
            self._beta = 0.0
            return

        # Side-slip angle at CG
        beta = np.arctan(CarLenR / CarWheelBase * np.tan(self.steer))
        self._beta = beta

        # Turn radius
        if abs(np.tan(self.steer)) > 1e-8:
            radius = CarLenR / np.sin(beta) if abs(np.sin(beta)) > 1e-8 else 1e6
        else:
            radius = 1e6

        # Yaw rate
        self.psi_dot = self.spd * np.cos(beta) / CarWheelBase * np.tan(self.steer)

        # Lateral acceleration
        if abs(radius) > 1e-3:
            self.lat_acc = self.spd ** 2 / radius
        else:
            self.lat_acc = 0.0

    # ── pose update ───────────────────────────────────────────────────────
    def _update_pose(self):
        """Integrate position and heading."""
        beta = getattr(self, "_beta", 0.0)
        psi = self.pose[2]

        self.pose[0] += self.spd * np.cos(psi + beta) * DT
        self.pose[1] += self.spd * np.sin(psi + beta) * DT
        self.pose[2] += self.psi_dot * DT

        # Normalise heading to [-pi, pi]
        self.pose[2] = (self.pose[2] + np.pi) % (2 * np.pi) - np.pi

    # ── body corners ──────────────────────────────────────────────────────
    def get_body_corners(self) -> np.ndarray:
        """
        Return 4x2 array of car body corners in world frame.

        The rectangle (CarLength x CarWidth) is centred at self.pose[:2]
        and rotated by self.pose[2].
        """
        x, y, psi = self.pose
        c, s = np.cos(psi), np.sin(psi)
        hl = CarLength / 2.0
        hw = CarWidth / 2.0

        corners = np.array([
            [x + c * hl - s * hw, y + s * hl + c * hw],  # front-left
            [x + c * hl + s * hw, y + s * hl - c * hw],  # front-right
            [x - c * hl + s * hw, y - s * hl - c * hw],  # rear-right
            [x - c * hl - s * hw, y - s * hl + c * hw],  # rear-left
        ])
        return corners


# ── self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== RCCarModelClass Self-Test ===")

    car = RCCarModelClass(pose0=[0.0, 0.0, np.pi / 2], spd0=0.0)

    # Accelerate straight for 1 second (50 steps)
    for _ in range(50):
        car.step([1.0, 0.0])  # full throttle, no steer

    print(f"  After 1 s full throttle:")
    print(f"    pose = [{car.pose[0]:.4f}, {car.pose[1]:.4f}, {car.pose[2]:.4f}]")
    print(f"    spd  = {car.spd:.4f} m/s")
    assert car.spd > 0, "Speed should increase"
    assert car.pose[1] > 0, "Car should move forward (up, since heading=pi/2)"

    # Steer right for 1 second
    for _ in range(50):
        car.step([0.5, -1.0])  # half throttle, full right steer

    print(f"  After 1 s half-throttle + right steer:")
    print(f"    pose = [{car.pose[0]:.4f}, {car.pose[1]:.4f}, {car.pose[2]:.4f}]")
    print(f"    spd  = {car.spd:.4f} m/s")
    print(f"    steer = {car.steer:.4f} rad")
    print(f"    psi_dot = {car.psi_dot:.4f} rad/s")

    # Body corners
    corners = car.get_body_corners()
    print(f"  Body corners:\n{corners}")
    assert corners.shape == (4, 2)

    # Reset
    car.reset([0, 0, 0], 0.0)
    assert np.allclose(car.pose, [0, 0, 0])
    assert car.spd == 0.0

    # Brake test
    car.reset([0, 0, 0], 5.0)
    for _ in range(100):
        car.step([-1.0, 0.0])  # full brake
    print(f"  After 2 s full brake from 5 m/s:")
    print(f"    spd = {car.spd:.4f} m/s")
    assert car.spd < 0.01, "Car should have stopped"

    # Max speed test
    car.reset([0, 0, 0], 0.0)
    for _ in range(500):
        car.step([1.0, 0.0])
    print(f"  After 10 s full throttle:")
    print(f"    spd = {car.spd:.4f} m/s  (max = {SPD_MAX})")
    assert car.spd <= SPD_MAX + 1e-6, "Speed should be clipped to SPD_MAX"

    print("=== All RCCarModel tests passed! ===")
