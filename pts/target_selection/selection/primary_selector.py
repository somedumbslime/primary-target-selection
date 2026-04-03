from __future__ import annotations

from dataclasses import dataclass

from ..domain.enums import EventType, SelectionState
from ..domain.models import ScoreBreakdown, SelectionResult
from ..domain.reasons import SelectionReason


@dataclass
class PrimarySelectorConfig:
    switch_margin: float = 0.18
    switch_persistence_frames: int = 5
    target_lost_grace_frames: int = 10
    min_acquire_score: float = 0.50
    acquire_persistence_frames: int = 3


class PrimaryTargetSelector:
    def __init__(self, config: PrimarySelectorConfig):
        self.config = config
        self.current_target_id: int | None = None
        self.current_target_last_score: float | None = None
        self.current_missing_counter: int = 0
        self.switch_candidate_id: int | None = None
        self.switch_counter: int = 0
        self.acquire_candidate_id: int | None = None
        self.acquire_counter: int = 0

    def _reset_switch(self) -> None:
        self.switch_candidate_id = None
        self.switch_counter = 0

    def _reset_acquire(self) -> None:
        self.acquire_candidate_id = None
        self.acquire_counter = 0

    def reset(self) -> None:
        self.current_target_id = None
        self.current_target_last_score = None
        self.current_missing_counter = 0
        self._reset_switch()
        self._reset_acquire()

    def _acquire_if_ready(self, best: ScoreBreakdown | None) -> SelectionResult:
        required_frames = max(1, int(self.config.acquire_persistence_frames))
        min_score = float(self.config.min_acquire_score)

        if best is None:
            self._reset_acquire()
            return SelectionResult(
                primary_target_id=None,
                primary_score=None,
                selection_state=SelectionState.NO_TARGET,
                switch_candidate_id=None,
                selection_reason=SelectionReason.NO_VALID_CANDIDATES.value,
            )

        if best.final_score < min_score:
            self._reset_acquire()
            return SelectionResult(
                primary_target_id=None,
                primary_score=None,
                selection_state=SelectionState.NO_TARGET,
                switch_candidate_id=None,
                selection_reason=SelectionReason.ACQUIRE_SCORE_BELOW_THRESHOLD.value,
            )

        if self.acquire_candidate_id == best.track_id:
            self.acquire_counter += 1
        else:
            self.acquire_candidate_id = best.track_id
            self.acquire_counter = 1

        if self.acquire_counter < required_frames:
            return SelectionResult(
                primary_target_id=None,
                primary_score=None,
                selection_state=SelectionState.NO_TARGET,
                switch_candidate_id=self.acquire_candidate_id,
                selection_reason=SelectionReason.ACQUIRE_PERSISTENCE_NOT_REACHED.value,
            )

        self.current_target_id = best.track_id
        self.current_target_last_score = best.final_score
        self.current_missing_counter = 0
        self._reset_switch()
        self._reset_acquire()
        return SelectionResult(
            primary_target_id=best.track_id,
            primary_score=best.final_score,
            selection_state=SelectionState.TARGET_LOCKED,
            switch_candidate_id=None,
            selection_reason=SelectionReason.INITIAL_ACQUIRE.value,
            event_type=EventType.TARGET_ACQUIRED,
            previous_target_id=None,
        )

    def select(self, scored: dict[int, ScoreBreakdown]) -> SelectionResult:
        ranked = sorted(scored.values(), key=lambda s: s.final_score, reverse=True)
        best = ranked[0] if ranked else None

        if self.current_target_id is None:
            return self._acquire_if_ready(best)

        current = scored.get(self.current_target_id)
        if current is None:
            self.current_missing_counter += 1
            self._reset_switch()
            grace = max(0, int(self.config.target_lost_grace_frames))

            if self.current_missing_counter <= grace:
                return SelectionResult(
                    primary_target_id=self.current_target_id,
                    primary_score=self.current_target_last_score,
                    selection_state=SelectionState.TARGET_LOCKED,
                    switch_candidate_id=None,
                    selection_reason=SelectionReason.HOLD_LOST_GRACE.value,
                )

            prev = self.current_target_id
            self.current_target_id = None
            self.current_target_last_score = None
            self.current_missing_counter = 0
            self._reset_acquire()
            return SelectionResult(
                primary_target_id=None,
                primary_score=None,
                selection_state=SelectionState.NO_TARGET,
                switch_candidate_id=None,
                selection_reason=SelectionReason.TARGET_LOST.value,
                event_type=EventType.TARGET_LOST,
                previous_target_id=prev,
            )

        self.current_missing_counter = 0
        self.current_target_last_score = current.final_score
        self._reset_acquire()

        if best is None or best.track_id == self.current_target_id:
            self._reset_switch()
            return SelectionResult(
                primary_target_id=self.current_target_id,
                primary_score=current.final_score,
                selection_state=SelectionState.TARGET_LOCKED,
                switch_candidate_id=None,
                selection_reason=SelectionReason.HOLD_CURRENT_TARGET.value,
            )

        if best.final_score > current.final_score + self.config.switch_margin:
            if self.switch_candidate_id == best.track_id:
                self.switch_counter += 1
            else:
                self.switch_candidate_id = best.track_id
                self.switch_counter = 1

            required_switch_frames = max(1, int(self.config.switch_persistence_frames))
            if self.switch_counter >= required_switch_frames:
                prev = self.current_target_id
                self.current_target_id = best.track_id
                self.current_target_last_score = best.final_score
                self._reset_switch()
                return SelectionResult(
                    primary_target_id=best.track_id,
                    primary_score=best.final_score,
                    selection_state=SelectionState.TARGET_LOCKED,
                    switch_candidate_id=None,
                    selection_reason=SelectionReason.SWITCHED_TO_BETTER_CANDIDATE.value,
                    event_type=EventType.TARGET_SWITCHED,
                    previous_target_id=prev,
                )

            return SelectionResult(
                primary_target_id=self.current_target_id,
                primary_score=current.final_score,
                selection_state=SelectionState.SWITCH_PENDING,
                switch_candidate_id=self.switch_candidate_id,
                selection_reason=SelectionReason.SWITCH_PERSISTENCE_NOT_REACHED.value,
            )

        self._reset_switch()
        return SelectionResult(
            primary_target_id=self.current_target_id,
            primary_score=current.final_score,
            selection_state=SelectionState.TARGET_LOCKED,
            switch_candidate_id=None,
            selection_reason=SelectionReason.SWITCH_MARGIN_NOT_REACHED.value,
        )
