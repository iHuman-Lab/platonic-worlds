"""Constants shared by the CarRacing collector."""

IMAGE_SIZES = {"low": (96, 96), "middle": (300, 200), "high": (600, 400)}
SIMULATION_FPS = 50
CONTROLLER_DETAIL_NAMES = (
    "target_speed",
    "target_heading_error",
    "preview_curvature_abs",
    "steering_noise",
)
EXTRA_NAMES = (
    "forward_velocity",
    "lateral_velocity_right",
    "steering_angle_mean",
    "wheel_contact_count",
    "progress",
    "center_distance",
    "nearest_segment",
    "beta_valid",
    "road_heading",
)
