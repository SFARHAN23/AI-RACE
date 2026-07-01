"""
rc_track.py — Parametric track for 1/10-scale RC cars with hard border walls.

The track is constructed from segments, each consisting of a straight section
followed by a circular arc (curve). Wall segments are pre-computed as an Mx4
numpy array for fast LiDAR raycasting.

Default layout: small rounded rectangle (~9.14 m centerline).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from track_library import get_track_config

# ── Shared constants ─────────────────────────────────────────────────────────
CAR_MASS = 3.5          # kg
CAR_WHEELBASE = 0.3302  # m
CAR_LENGTH = 0.45       # m
CAR_WIDTH = 0.20        # m
CAR_MAX_SPEED = 8.0     # m/s
CAR_MAX_STEER = 0.4189  # rad (~24 deg)
CAR_MAX_STEER_RATE = CAR_MAX_STEER  # rad/s (full lock in 1 s)
DT = 0.02               # s (50 Hz control loop)

TRACK_WIDTH = 1.20      # m — baseline lane width for the 1/10-scale simulator

N_BEAMS = 20
LIDAR_ANGLE_MIN = -2.3562   # -135 deg in rad
LIDAR_ANGLE_MAX = 2.3562    # +135 deg in rad
LIDAR_MAX_RANGE = 12.0      # m
LIDAR_MIN_RANGE = 0.05      # m

# Sampling resolution along the track (meters)
_SAMPLE_DS = 0.02


# ── Default segment list ─────────────────────────────────────────────────────
# Each tuple: (straight_length_m, curve_angle_rad, curve_radius_m)
# Positive angle = right turn (clockwise heading decrease).
DEFAULT_SEGMENTS = [
    (2.0, np.pi / 2, 1.2),   # Unit 0: straight 2m, 90° right, R=1.2m
    (1.0, np.pi / 2, 1.2),   # Unit 1: straight 1m, 90° right, R=1.2m
    (2.0, np.pi / 2, 1.2),   # Unit 2: straight 2m, 90° right, R=1.2m
    (1.0, np.pi / 2, 1.2),   # Unit 3: straight 1m, 90° right, R=1.2m
]


class RCTrackClass:
    """Parametric closed track for 1/10-scale RC cars."""

    # ── construction ──────────────────────────────────────────────────────
    def __init__(self, segments=None, width=TRACK_WIDTH, ds=_SAMPLE_DS, track_name=None):
        self.track_name = track_name or os.environ.get('SIMTOREAL_TRACK', 'default')
        self._centerline_points = None
        if segments is None:
            if self.track_name in (None, '', 'default'):
                segments = DEFAULT_SEGMENTS
                width = width
            else:
                cfg = get_track_config(self.track_name)
                width = cfg.get('width_m', width)
                ds = cfg.get('ds_m', ds)
                if 'centerline_xy' in cfg:
                    self._centerline_points = np.asarray(cfg['centerline_xy'], dtype=np.float64)
                    segments = None
                else:
                    segments = cfg['segments']
        self.segments = segments
        self.width = width
        self._ds = ds

        # Build raw centerline points + headings.  Legacy tracks use straight/arc
        # segments; high-detail imported circuits can provide centerline points.
        if self._centerline_points is not None:
            raw_xy, raw_hdg, raw_dist = self._build_centerline_from_points(self._centerline_points)
        else:
            raw_xy, raw_hdg, raw_dist = self._build_centerline(segments)

        # Store
        self.centerline_xy = raw_xy          # Nx2
        self.centerline_headings = raw_hdg   # N
        self.centerline_cumdist = raw_dist   # N
        self.total_trip = raw_dist[-1]

        # Offset walls
        hw = self.width / 2.0
        normals = np.column_stack([
            -np.sin(self.centerline_headings),
             np.cos(self.centerline_headings),
        ])  # Nx2  — unit left-normal
        self.left_wall_xy = self.centerline_xy + hw * normals
        self.right_wall_xy = self.centerline_xy - hw * normals

        # Pre-compute wall line-segments (Mx4)
        self.wall_segments = self._make_wall_segments()

        # State set by findcar
        self.centerlinedist = 0.0
        self.track_dir = 0.0
        self.car_trip = 0.0

    # ── internal builders ─────────────────────────────────────────────────
    def _build_centerline(self, segments):
        """Return (xy: Nx2, headings: N, cumdist: N) from segment list."""
        pts = []
        hdgs = []

        x, y, psi = 0.0, 0.0, np.pi / 2  # start pose

        for straight_len, curve_angle, radius in segments:
            # ---- straight ----
            if straight_len > 0:
                n_pts = max(int(round(straight_len / self._ds)), 1)
                for i in range(n_pts):
                    frac = i / n_pts
                    px = x + frac * straight_len * np.cos(psi)
                    py = y + frac * straight_len * np.sin(psi)
                    pts.append([px, py])
                    hdgs.append(psi)
                # advance pose
                x += straight_len * np.cos(psi)
                y += straight_len * np.sin(psi)

            # ---- arc ----
            if abs(curve_angle) > 1e-9 and radius > 0:
                arc_len = abs(curve_angle) * radius
                n_pts = max(int(round(arc_len / self._ds)), 1)
                # Turn direction: positive curve_angle = right turn
                sign = -1.0 if curve_angle > 0 else 1.0  # heading change sign
                # Center of curvature
                cx = x + radius * np.cos(psi + sign * np.pi / 2)
                cy = y + radius * np.sin(psi + sign * np.pi / 2)
                start_angle = np.arctan2(y - cy, x - cx)
                for i in range(n_pts):
                    frac = i / n_pts
                    # Move around the circle in the same direction as the
                    # heading change. For a right turn from heading north,
                    # theta must decrease pi -> pi/2; the old sign here made
                    # theta increase pi -> 3pi/2, creating a cusp/discontinuous
                    # track that sent the car into the first cardboard wall.
                    theta = start_angle + sign * curve_angle * frac
                    px = cx + radius * np.cos(theta)
                    py = cy + radius * np.sin(theta)
                    pts.append([px, py])
                    hdgs.append(psi + sign * curve_angle * frac)
                # advance pose
                psi += sign * curve_angle
                end_theta = start_angle + sign * curve_angle
                x = cx + radius * np.cos(end_theta)
                y = cy + radius * np.sin(end_theta)

        pts = np.array(pts, dtype=np.float64)
        hdgs = np.array(hdgs, dtype=np.float64)

        # Cumulative arc-length distance
        diffs = np.diff(pts, axis=0)
        seg_lens = np.sqrt(diffs[:, 0] ** 2 + diffs[:, 1] ** 2)
        cumdist = np.zeros(len(pts))
        cumdist[1:] = np.cumsum(seg_lens)

        return pts, hdgs, cumdist

    def _build_centerline_from_points(self, points):
        """Return (xy, headings, cumdist) from imported closed centerline points.

        Imported F1 layouts are already uniformly sampled in meters.  We still
        close/resample them to this track's ds so downstream LiDAR, collision,
        progress, and adaptive-spawn logic can use the exact same arrays as the
        legacy segment tracks.
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 4:
            raise ValueError("centerline points must be an Nx2 array with at least 4 points")
        # Drop duplicate consecutive points.
        step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        pts = pts[np.r_[True, step > 1e-9]]
        if np.linalg.norm(pts[0] - pts[-1]) > 1e-9:
            closed = np.vstack([pts, pts[0]])
        else:
            closed = pts
        seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        keep = np.r_[True, seg > 1e-9]
        closed = closed[keep]
        seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        dist_closed = np.r_[0.0, np.cumsum(seg)]
        total = float(dist_closed[-1])
        n_pts = max(4, int(round(total / max(self._ds, 1e-6))))
        sample_s = np.linspace(0.0, total, n_pts, endpoint=False)
        x = np.interp(sample_s, dist_closed, closed[:, 0])
        y = np.interp(sample_s, dist_closed, closed[:, 1])
        xy = np.column_stack([x, y])
        xy_next = np.roll(xy, -1, axis=0)
        delta = xy_next - xy
        hdgs = np.arctan2(delta[:, 1], delta[:, 0])
        diffs = np.diff(xy, axis=0)
        seg_lens = np.sqrt(diffs[:, 0] ** 2 + diffs[:, 1] ** 2)
        cumdist = np.zeros(len(xy))
        cumdist[1:] = np.cumsum(seg_lens)
        return xy, hdgs, cumdist

    def _make_wall_segments(self):
        """Build Mx4 array of [x1,y1,x2,y2] from consecutive wall points."""
        segs = []
        for wall in (self.left_wall_xy, self.right_wall_xy):
            n = len(wall)
            for i in range(n - 1):
                segs.append([
                    wall[i, 0], wall[i, 1],
                    wall[i + 1, 0], wall[i + 1, 1],
                ])
            # Close the loop (last point -> first point)
            segs.append([
                wall[-1, 0], wall[-1, 1],
                wall[0, 0], wall[0, 1],
            ])
        return np.array(segs, dtype=np.float64)

    # ── public API ────────────────────────────────────────────────────────
    def get_wall_segments(self) -> np.ndarray:
        """Return the Mx4 wall-segment array."""
        return self.wall_segments

    def findcar(self, pos) -> bool:
        """
        Locate *pos* ([x, y, ...]) relative to the centerline.

        Sets:
            self.centerlinedist — signed perpendicular distance (+left, -right)
            self.track_dir      — heading at nearest centerline point
            self.car_trip       — cumulative distance along centerline

        Returns True if the car is within track boundaries.
        """
        p = np.asarray(pos[:2], dtype=np.float64)

        # Nearest centerline point (vectorised)
        diffs = self.centerline_xy - p
        dists = np.sqrt(diffs[:, 0] ** 2 + diffs[:, 1] ** 2)
        nearest_idx = int(np.argmin(dists))

        # Track heading at nearest point
        track_heading = self.centerline_headings[nearest_idx]
        forward = np.array([np.cos(track_heading), np.sin(track_heading)])

        # Vector from centerline to car
        to_car = p - self.centerline_xy[nearest_idx]

        # Signed perpendicular distance (cross product; + = left)
        cross = forward[0] * to_car[1] - forward[1] * to_car[0]

        self.centerlinedist = float(cross)
        self.track_dir = float(track_heading)
        self.car_trip = float(self.centerline_cumdist[nearest_idx])

        return abs(cross) <= self.width / 2.0

    def check_body_collision(self, pos, heading, half_length=CAR_LENGTH / 2,
                             half_width=CAR_WIDTH / 2) -> bool:
        """
        Return True if ANY corner of the car body rectangle is outside
        the track boundaries.
        """
        c = np.cos(heading)
        s = np.sin(heading)
        corners = np.array([
            [pos[0] + c * half_length - s * half_width,
             pos[1] + s * half_length + c * half_width],
            [pos[0] + c * half_length + s * half_width,
             pos[1] + s * half_length - c * half_width],
            [pos[0] - c * half_length + s * half_width,
             pos[1] - s * half_length - c * half_width],
            [pos[0] - c * half_length - s * half_width,
             pos[1] - s * half_length + c * half_width],
        ])
        for corner in corners:
            if not self.findcar(corner):
                return True
        return False

    def random_car_pose(self, lateral_frac: float = 0.42, heading_jitter_rad: float = 0.12,
                        allow_curves: bool = True, min_clearance_m: float = 0.18,
                        max_attempts: int = 250):
        """Return an adaptive random valid [x, y, psi] anywhere on the track.

        Older training sampled only straight sections and only near the centerline.
        That taught a center-spawn assumption.  This sampler deliberately covers
        non-center lane offsets, curve entries/exits, and different progress
        positions while rejecting poses whose body corners are too close to the
        cardboard walls.
        """
        n = len(self.centerline_xy)
        dh = np.abs(np.diff(np.unwrap(self.centerline_headings), prepend=self.centerline_headings[0]))
        straight_mask = dh < 1e-3
        curve_mask = ~straight_mask
        valid_idxs = np.arange(n) if allow_curves or not np.any(straight_mask) else np.where(straight_mask)[0]
        safe_half_lane = max(0.02, self.width * 0.5 - min_clearance_m - CAR_WIDTH * 0.5)
        max_lat = min(abs(lateral_frac) * self.width, safe_half_lane)

        for _ in range(max_attempts):
            # Bias slightly toward straights for easier early learning, but keep
            # enough curve/off-center starts to remove the center-start crutch.
            if allow_curves and np.any(curve_mask) and np.random.rand() < 0.35:
                idx = int(np.random.choice(np.where(curve_mask)[0]))
            else:
                idx = int(np.random.choice(valid_idxs))
            hdg = float(self.centerline_headings[idx])
            lat = float(np.random.uniform(-max_lat, max_lat))
            normal = np.array([-np.sin(hdg), np.cos(hdg)])
            xy = self.centerline_xy[idx] + lat * normal
            psi = hdg + float(np.random.uniform(-heading_jitter_rad, heading_jitter_rad))
            if self.findcar(xy) and not self.check_body_collision(xy, psi):
                return [float(xy[0]), float(xy[1]), float(psi)]

        # Conservative fallback: valid centerline sample.
        idx = int(np.random.choice(valid_idxs))
        return [float(self.centerline_xy[idx, 0]), float(self.centerline_xy[idx, 1]), float(self.centerline_headings[idx])]

    def pose_at_trip(self, trip_m: float, lateral_offset_m: float = 0.0, heading_jitter_rad: float = 0.0):
        """Return [x, y, psi] at a chosen progress distance and lateral offset."""
        target = float(trip_m) % self.total_trip
        idx = int(np.argmin(np.abs(self.centerline_cumdist - target)))
        hdg = float(self.centerline_headings[idx])
        max_lat = max(0.0, self.width * 0.5 - CAR_WIDTH * 0.5 - 0.05)
        lat = float(np.clip(lateral_offset_m, -max_lat, max_lat))
        normal = np.array([-np.sin(hdg), np.cos(hdg)])
        xy = self.centerline_xy[idx] + lat * normal
        return [float(xy[0]), float(xy[1]), float(hdg + heading_jitter_rad)]

    def test_reset_pose(self, lateral_offset_m: float = 0.0, trip_m: float = 0.0):
        """Return a deterministic start pose, optionally off-center."""
        return self.pose_at_trip(trip_m=trip_m, lateral_offset_m=lateral_offset_m)

    # ── visualisation ─────────────────────────────────────────────────────
    def show(self):
        """Plot track with walls and centerline. Returns the figure."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.set_aspect("equal")
        ax.set_title("RC Track")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

        # Centerline
        ax.plot(self.centerline_xy[:, 0], self.centerline_xy[:, 1],
                "k--", lw=0.5, label="centerline")

        # Walls
        ax.plot(self.left_wall_xy[:, 0], self.left_wall_xy[:, 1],
                "b-", lw=1.2, label="left wall")
        ax.plot(self.right_wall_xy[:, 0], self.right_wall_xy[:, 1],
                "r-", lw=1.2, label="right wall")

        # Start marker
        ax.plot(0, 0, "go", ms=8, label="start")

        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig


# ── self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== RCTrackClass Self-Test ===")

    track = RCTrackClass()
    print(f"  Track width       : {track.width:.2f} m")
    print(f"  Total trip        : {track.total_trip:.2f} m")
    print(f"  Centerline points : {len(track.centerline_xy)}")
    print(f"  Wall segments     : {track.wall_segments.shape}")

    # findcar at origin (should be on track)
    on = track.findcar([0.0, 0.0])
    print(f"  findcar([0,0])    : on_track={on}, "
          f"dist={track.centerlinedist:.4f}, dir={track.track_dir:.4f}")
    assert on, "Origin should be on track!"

    # findcar well outside track
    off = track.findcar([10.0, 10.0])
    print(f"  findcar([10,10])  : on_track={off}")
    assert not off, "Far point should be off track!"

    # Body collision at origin (should be fine)
    col = track.check_body_collision([0.0, 0.0], np.pi / 2)
    print(f"  Body collision at origin: {col}")

    # Random pose
    rp = track.random_car_pose()
    print(f"  Random pose       : x={rp[0]:.3f}, y={rp[1]:.3f}, psi={rp[2]:.3f}")

    # Test reset
    tp = track.test_reset_pose()
    print(f"  Test reset pose   : {tp}")

    # Show
    fig = track.show()
    plt.savefig("rc_track_test.png", dpi=100)
    print("  Saved rc_track_test.png")
    plt.show()

    print("=== All RCTrack tests passed! ===")
