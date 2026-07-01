"""
lidar_sim.py — Vectorised raycasting LiDAR simulator for synthetic RPLIDAR C1 scans.

Provides two implementations:
  • scan()      — fully vectorised (all beams × all segments via broadcasting)
  • scan_loop() — per-beam loop (easier to read, used as reference)
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


class LidarSimulator:
    """Vectorised raycasting LiDAR for 2-D wall segments."""

    def __init__(
        self,
        n_beams: int = N_BEAMS,
        angle_min: float = LIDAR_ANGLE_MIN,
        angle_max: float = LIDAR_ANGLE_MAX,
        max_range: float = LIDAR_MAX_RANGE,
        min_range: float = LIDAR_MIN_RANGE,
    ):
        self.n_beams = n_beams
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.max_range = max_range
        self.min_range = min_range

        # Pre-compute beam angles relative to car heading
        self.beam_angles = np.linspace(angle_min, angle_max, n_beams)

        # Wall segments (set via set_walls)
        self.walls = None   # Mx4 [x1, y1, x2, y2]

    # ── wall setup ────────────────────────────────────────────────────────
    def set_walls(self, wall_segments: np.ndarray):
        """
        Store wall segments.  Called once after track construction.

        Parameters
        ----------
        wall_segments : np.ndarray, shape (M, 4)
            Each row is [x1, y1, x2, y2].
        """
        self.walls = np.asarray(wall_segments, dtype=np.float64)

    # ── fully vectorised scan (default) ───────────────────────────────────
    def scan(self, pos_x: float, pos_y: float, heading: float) -> np.ndarray:
        """
        Generate a LiDAR scan — fully vectorised (no Python loops).

        Parameters
        ----------
        pos_x, pos_y : float   — sensor origin in world frame
        heading      : float   — sensor heading (rad)

        Returns
        -------
        distances : np.ndarray, shape (n_beams,)
            Range for each beam, clipped to [min_range, max_range].
        """
        assert self.walls is not None, "Call set_walls() before scanning."

        # Absolute beam angles
        abs_angles = heading + self.beam_angles          # (B,)
        ray_dx = np.cos(abs_angles)                      # (B,)
        ray_dy = np.sin(abs_angles)                      # (B,)

        # Wall segment endpoints
        ax = self.walls[:, 0]                            # (M,)
        ay = self.walls[:, 1]
        dx_seg = self.walls[:, 2] - ax                   # (M,)
        dy_seg = self.walls[:, 3] - ay

        # Broadcasting: (B,1) op (1,M) -> (B,M)
        # Cross product  D × (B-A)
        denom = (ray_dx[:, None] * dy_seg[None, :]
                 - ray_dy[:, None] * dx_seg[None, :])    # (B, M)

        # (A - P) vectors  — broadcast to (B, M) automatically
        apx = ax[None, :] - pos_x                       # (1, M)
        apy = ay[None, :] - pos_y                        # (1, M)

        # Numerators
        t_num = apx * dy_seg[None, :] - apy * dx_seg[None, :]   # (B, M)
        u_num = apx * ray_dy[:, None] - apy * ray_dx[:, None]   # (B, M)

        # Safe division (avoid /0 for parallel rays)
        safe_denom = np.where(np.abs(denom) > 1e-10, denom, 1.0)
        t = t_num / safe_denom
        u = u_num / safe_denom

        # Valid intersections: non-parallel, t > 0, 0 ≤ u ≤ 1
        valid = (np.abs(denom) > 1e-10) & (t > 1e-6) & (u >= 0.0) & (u <= 1.0)

        t[~valid] = np.inf
        distances = np.min(t, axis=1)                    # (B,)

        distances = np.clip(distances, self.min_range, self.max_range)
        return distances

    # ── per-beam loop scan (reference) ────────────────────────────────────
    def scan_loop(self, pos_x: float, pos_y: float, heading: float) -> np.ndarray:
        """
        Generate a LiDAR scan — per-beam loop version (for clarity).

        Same interface and semantics as scan().
        """
        assert self.walls is not None, "Call set_walls() before scanning."

        abs_angles = heading + self.beam_angles
        ray_dx = np.cos(abs_angles)
        ray_dy = np.sin(abs_angles)

        ax = self.walls[:, 0]
        ay = self.walls[:, 1]
        dx_seg = self.walls[:, 2] - ax
        dy_seg = self.walls[:, 3] - ay

        distances = np.full(self.n_beams, self.max_range)

        for i in range(self.n_beams):
            rdx = ray_dx[i]
            rdy = ray_dy[i]

            # D × seg_dir for all segments
            denom = rdx * dy_seg - rdy * dx_seg          # (M,)
            valid = np.abs(denom) > 1e-10

            apx = ax - pos_x
            apy = ay - pos_y

            t_num = apx * dy_seg - apy * dx_seg
            u_num = apx * rdy - apy * rdx

            t = np.full(len(ax), np.inf)
            u = np.full(len(ax), -1.0)
            t[valid] = t_num[valid] / denom[valid]
            u[valid] = u_num[valid] / denom[valid]

            hit = valid & (t > 0) & (u >= 0) & (u <= 1)
            if np.any(hit):
                distances[i] = np.min(t[hit])

        distances = np.clip(distances, self.min_range, self.max_range)
        return distances


# ── self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== LidarSimulator Self-Test ===")

    # ---- build a track for testing ----
    from rc_track import RCTrackClass

    track = RCTrackClass()
    walls = track.get_wall_segments()
    print(f"  Track wall segments: {walls.shape}")

    # ---- create LiDAR ----
    lidar = LidarSimulator()
    lidar.set_walls(walls)
    print(f"  Beams      : {lidar.n_beams}")
    print(f"  Angle range: [{lidar.angle_min:.4f}, {lidar.angle_max:.4f}] rad")

    # ---- scan from start pose ----
    start = track.test_reset_pose()  # [0, 0, pi/2]
    px, py, heading = start

    scan_vec = lidar.scan(px, py, heading)
    scan_lp = lidar.scan_loop(px, py, heading)

    print(f"\n  Scan from start pose ({px:.1f}, {py:.1f}, hdg={heading:.2f}):")
    print(f"    Vectorised : min={scan_vec.min():.3f}  max={scan_vec.max():.3f}")
    print(f"    Loop       : min={scan_lp.min():.3f}  max={scan_lp.max():.3f}")

    # Both methods should agree closely
    diff = np.abs(scan_vec - scan_lp)
    print(f"    Max diff   : {diff.max():.6f}")
    assert diff.max() < 1e-6, "Vectorised and loop scans disagree!"

    # All distances should be positive
    assert np.all(scan_vec >= lidar.min_range), "Distances below min_range!"
    assert np.all(scan_vec <= lidar.max_range), "Distances above max_range!"

    # ---- pretty-print per-beam ----
    print(f"\n  Per-beam distances (vectorised):")
    for i, (angle, dist) in enumerate(zip(lidar.beam_angles, scan_vec)):
        angle_deg = np.degrees(angle)
        bar = "#" * int(dist / lidar.max_range * 40)
        print(f"    Beam {i:2d}  {angle_deg:+7.1f}°  {dist:6.3f} m  {bar}")

    # ---- scan from a random pose ----
    rp = track.random_car_pose()
    scan2 = lidar.scan(rp[0], rp[1], rp[2])
    print(f"\n  Scan from random pose ({rp[0]:.2f}, {rp[1]:.2f}, {rp[2]:.2f}):")
    print(f"    min={scan2.min():.3f}  max={scan2.max():.3f}")

    print("\n=== All LidarSimulator tests passed! ===")
