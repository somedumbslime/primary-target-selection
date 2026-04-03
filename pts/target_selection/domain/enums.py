from enum import Enum


class SelectionState(str, Enum):
    NO_TARGET = "no_target"
    TARGET_LOCKED = "target_locked"
    SWITCH_PENDING = "switch_pending"


class EventType(str, Enum):
    TARGET_ACQUIRED = "target_acquired"
    TARGET_LOST = "target_lost"
    TARGET_SWITCHED = "target_switched"
    CANDIDATE_REJECTED = "candidate_rejected"
    SCORE_UPDATED = "score_updated"

