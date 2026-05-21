# evrptw_gen/utils/__init__.py
from .geometry import (
    Rect, euclidean, dist_matrix,
    running_bbox_init, running_bbox_update,
    inflate_and_clip, sample_uniform_rect, clip_point_to_rect, clamp
)
from .timing import (
    minutes_from_distance, energy_from_distance, energy_from_time,
    can_travel_direct, clamp_time_windows,
)

from .feasibility import (cs_min_time_to_depot)

from .energy_consumption_model import consumption_model
