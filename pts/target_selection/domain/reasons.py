from enum import Enum


class SelectionReason(str, Enum):
    NO_VALID_CANDIDATES = "no_valid_candidates"
    ACQUIRE_SCORE_BELOW_THRESHOLD = "acquire_score_below_threshold"
    ACQUIRE_PERSISTENCE_NOT_REACHED = "acquire_persistence_not_reached"
    INITIAL_ACQUIRE = "initial_acquire"
    HOLD_CURRENT_TARGET = "hold_current_target"
    HOLD_LOST_GRACE = "hold_lost_grace"
    TARGET_LOST = "target_lost"
    SWITCH_MARGIN_NOT_REACHED = "switch_margin_not_reached"
    SWITCH_PERSISTENCE_NOT_REACHED = "switch_persistence_not_reached"
    SWITCHED_TO_BETTER_CANDIDATE = "switched_to_better_candidate"


class RejectReason(str, Enum):
    LOW_CONF = "candidate_rejected_low_conf"
    LOW_LIFETIME = "candidate_rejected_low_lifetime"
    SMALL_AREA = "candidate_rejected_small_area"
    UNSTABLE = "candidate_rejected_unstable"
