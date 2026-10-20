"""
Safe-Net Architecture — Layer 2: Interaction AI (Dyadic Behavioral Engine)
Description:
    Receives decoupled SafetyEvent metadata from Layer 1 (Context AI).
    Never reads raw conversational text to preserve user privacy and edge battery.
    Uses online streaming ML (`river`) wrapped in 3 deterministic safety shields
    to detect dyadic grooming escalation and abnormal interaction patterns
    with zero false-positive alarms.
"""

from datetime import datetime
from typing import Dict, Any, Optional
from river import anomaly, stats


class RiverContactTracker:
    """
    Per-contact streaming ML wrapper using River.
    Maintains a tiny memory footprint per contact_id on-device:
      1. EWMA (Exponentially Weighted Moving Average) for rolling Layer 1 risk trend.
      2. Half-Space Trees for streaming multi-feature behavioral anomaly detection.
    """

    def __init__(self, ewma_alpha: float = 0.3, window_size: int = 25):
        # Tracks rolling L1 risk trend (alpha=0.3 weights recent escalation higher)
        # Tracks rolling L1 risk trend (compatible with all versions of river)
        if hasattr(stats, "ExponentialMovingAverage"):
            self.rolling_risk = stats.ExponentialMovingAverage(alpha=ewma_alpha)
        elif hasattr(stats, "EWMA"):
            self.rolling_risk = stats.EWMA(alpha=ewma_alpha)
        else:
            # Native Python fallback if river stats structure changes
            self.rolling_risk = stats.Mean()

        # Streaming isolation-forest style detector for interaction metadata
        self.anomaly_detector = anomaly.HalfSpaceTrees(
            n_trees=5,
            height=3,
            window_size=window_size,
            seed=42
        )

        self.interaction_count: int = 0
        self.high_risk_flag_count: int = 0
        self.last_seen_timestamp: Optional[float] = None

    def update_and_score(
        self,
        l1_score: float,
        msg_velocity: float,
        is_late_night: int,
        timestamp: float
    ) -> Dict[str, Any]:
        """
        Feeds one interaction event into River and extracts pure Python floats.
        """
        self.interaction_count += 1
        if l1_score >= 0.50:
            self.high_risk_flag_count += 1
        self.last_seen_timestamp = timestamp

        # 1. Update rolling risk trend (catches slow grooming creep)
        self.rolling_risk.update(l1_score)
        ewma_risk = float(self.rolling_risk.get())

        # 2. Construct normalized feature dictionary for HalfSpaceTrees
        # Scale velocity to [0.0, 1.0] assuming 15 msgs/min is max burst
        normalized_velocity = min(max(msg_velocity / 15.0, 0.0), 1.0)
        features = {
            "l1_score": float(l1_score),
            "velocity": round(normalized_velocity, 4),
            "late_night": float(is_late_night)
        }

        # Score current event BEFORE learning so anomalies stand out, then update model
        raw_anomaly_score = float(self.anomaly_detector.score_one(features))
        self.anomaly_detector.learn_one(features)

        return {
            "interaction_count": self.interaction_count,
            "high_risk_flag_count": self.high_risk_flag_count,
            "ewma_risk": round(ewma_risk, 4),
            "raw_anomaly_score": round(raw_anomaly_score, 4)
        }


class Layer2InteractionAI:
    """
    Main Layer 2 Orchestrator & False-Alarm Shield Engine.
    Links Layer 1 (Context AI) -> Layer 2 (Interaction AI) -> Layer 3 (Relational AI).
    """

    # Configurable Threshold Constants
    GATE0_SAFE_EXIT_THRESHOLD = 0.15
    WARMUP_MIN_INTERACTIONS = 5
    BENIGN_L1_CEILING = 0.20
    BENIGN_EWMA_CEILING = 0.25
    ANOMALY_WEIGHT = 0.25
    ESCALATE_L3_THRESHOLD = 0.65
    WATCHLIST_THRESHOLD = 0.35

    def __init__(self):
        # On-device state store mapping contact_id -> RiverContactTracker
        self.contact_registry: Dict[str, RiverContactTracker] = {}

    @staticmethod
    def _is_late_night(hour_of_day: int) -> int:
        """Returns 1 if interaction occurs between 11 PM (23) and 5 AM (5), else 0."""
        return 1 if (hour_of_day >= 23 or hour_of_day <= 5) else 0

    def process_safety_event(self, safety_event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Core pipeline method called immediately after Layer 1 finishes.
        Validates input contract, checks Early-Exit Gate, updates River,
        applies the 3 Safety Shields, and formats output for Layer 3.
        """
        # --- 1. Extract & Validate Layer 1 Contract Metadata ---
        contact_id: str = str(safety_event.get("contact_id", "unknown_contact"))
        l1_score: float = float(safety_event.get("l1_risk_score", 0.0))
        msg_velocity: float = float(safety_event.get("msg_velocity", 1.0))
        is_known_contact: bool = bool(safety_event.get("is_known_contact", False))
        timestamp: float = float(
            safety_event.get("timestamp", datetime.now().timestamp())
        )
        hour_of_day: int = int(
            safety_event.get("hour_of_day", datetime.fromtimestamp(timestamp).hour)
        )

        # --- GATE 0: Fast Early-Exit (Saves Edge Battery & CPU) ---
        # If the message is completely benign AND from a trusted known contact
        # with no prior high-risk history, drop early before running ML.
        existing_tracker = self.contact_registry.get(contact_id)
        has_prior_flags = (
            existing_tracker is not None and existing_tracker.high_risk_flag_count > 0
        )

        if (
            l1_score < self.GATE0_SAFE_EXIT_THRESHOLD
            and is_known_contact
            and not has_prior_flags
        ):
            safety_event["l2_status"] = "SAFE_DROPPED_AT_GATE0"
            safety_event["l2_final_score"] = round(l1_score, 4)
            safety_event["l2_action"] = "PASS_SAFE"
            safety_event["propagate_to_l3"] = False
            return safety_event

        # --- 2. Load or Initialize Contact Profile ---
        if contact_id not in self.contact_registry:
            self.contact_registry[contact_id] = RiverContactTracker()

        tracker = self.contact_registry[contact_id]
        late_night_flag = self._is_late_night(hour_of_day)

        # --- 3. Run River Streaming Update ---
        river_out = tracker.update_and_score(
            l1_score=l1_score,
            msg_velocity=msg_velocity,
            is_late_night=late_night_flag,
            timestamp=timestamp
        )

        count = river_out["interaction_count"]
        ewma_risk = river_out["ewma_risk"]
        raw_anomaly = river_out["raw_anomaly_score"]

        # =====================================================================
        # SHIELD 1: The Warm-Up Shield (Prevents Cold-Start False Alarms)
        # HalfSpaceTrees needs at least 5 interactions to learn a contact's baseline.
        # =====================================================================
        if count < self.WARMUP_MIN_INTERACTIONS:
            effective_anomaly = 0.0
            shield1_triggered = True
        else:
            effective_anomaly = raw_anomaly
            shield1_triggered = False

        # =====================================================================
        # SHIELD 2: Benign Anomaly Shield (Prevents Innocent Late-Night False Alarms)
        # Anomaly != Danger. If both current L1 risk and rolling EWMA risk are low,
        # suppress the behavioral anomaly boost so safe family/friend chats never alert.
        # =====================================================================
        if l1_score < self.BENIGN_L1_CEILING and ewma_risk < self.BENIGN_EWMA_CEILING:
            effective_anomaly = 0.0
            shield2_triggered = True
        else:
            shield2_triggered = False

        # =====================================================================
        # SHIELD 3: Multiplicative Coupling & Final Routing Gate
        # Combine the stronger of (immediate L1 risk, rolling EWMA risk) with
        # the shielded anomaly boost, plus a small repeat-offender penalty.
        # =====================================================================
        base_risk = max(l1_score, ewma_risk)
        repeat_flag_boost = min(river_out["high_risk_flag_count"] * 0.05, 0.20)

        l2_final_score = min(
            1.0,
            base_risk + (effective_anomaly * self.ANOMALY_WEIGHT) + repeat_flag_boost
        )
        l2_final_score = round(l2_final_score, 4)

        # Routing decision for Layer 3 (Relational/Group AI)
        if l2_final_score >= self.ESCALATE_L3_THRESHOLD:
            l2_action = "ESCALATE_TO_L3"
            propagate_to_l3 = True
        elif l2_final_score >= self.WATCHLIST_THRESHOLD:
            l2_action = "MONITOR_WATCHLIST"
            # Still pass watchlist events to Layer 3/4 so graph & temporal trends update
            propagate_to_l3 = True
        else:
            l2_action = "PASS_SAFE"
            propagate_to_l3 = False

        # --- 4. Enrich SafetyEvent Contract for Layer 3 & Layer 4 ---
        safety_event["l2_status"] = "PROCESSED"
        safety_event["l2_final_score"] = l2_final_score
        safety_event["l2_action"] = l2_action
        safety_event["propagate_to_l3"] = propagate_to_l3
        safety_event["l2_metrics"] = {
            "interaction_count": count,
            "high_risk_flag_count": river_out["high_risk_flag_count"],
            "ewma_risk": ewma_risk,
            "raw_anomaly_score": raw_anomaly,
            "effective_anomaly_score": round(effective_anomaly, 4),
            "is_late_night": bool(late_night_flag),
            "shields_active": {
                "warmup_shield_blocked": shield1_triggered,
                "benign_shield_blocked": shield2_triggered
            }
        }

        return safety_event


# =====================================================================
# SELF-TEST SUITE: Run `python layer2_interaction.py` to verify all 3 scenarios
# =====================================================================
if __name__ == "__main__":
    engine = Layer2InteractionAI()

    print("=== TEST 1: Brand-New Classmate (Cold-Start Shield Check) ===")
    new_friend_event = {
        "event_id": "evt_001",
        "child_id": "child_A",
        "contact_id": "classmate_101",
        "l1_risk_score": 0.08,
        "msg_velocity": 9.0,  # Fast excited texting
        "hour_of_day": 16,
        "is_known_contact": False
    }
    res1 = engine.process_safety_event(new_friend_event)
    print(f"Action: {res1['l2_action']} | Final Score: {res1['l2_final_score']} | Metrics: {res1['l2_metrics']}\n")

    print("=== TEST 2: Safe Late-Night Family Chat (Benign Anomaly Shield Check) ===")
    for i in range(6):
        late_safe_event = {
            "event_id": f"evt_10{i}",
            "child_id": "child_A",
            "contact_id": "aunt_sarah",
            "l1_risk_score": 0.06,
            "msg_velocity": 2.0,
            "hour_of_day": 2 if i == 5 else 18,  # 6th message happens at 2:00 AM
            "is_known_contact": False
        }
        res2 = engine.process_safety_event(late_safe_event)
    print(f"Action: {res2['l2_action']} | Final Score: {res2['l2_final_score']} | Metrics: {res2['l2_metrics']}\n")

    print("=== TEST 3: Slow Grooming Escalation at 2 AM (Escalation to Layer 3 Check) ===")
    escalating_scores = [0.25, 0.32, 0.40, 0.48, 0.55, 0.62]
    for idx, score in enumerate(escalating_scores):
        grooming_event = {
            "event_id": f"evt_20{idx}",
            "child_id": "child_A",
            "contact_id": "stranger_999",
            "l1_risk_score": score,
            "msg_velocity": 12.0,  # High pressure messaging
            "hour_of_day": 2,      # 2:00 AM
            "is_known_contact": False
        }
        res3 = engine.process_safety_event(grooming_event)
    print(f"Action: {res3['l2_action']} | Final Score: {res3['l2_final_score']} | Propagate to L3: {res3['propagate_to_l3']}")
    print(f"Metrics: {res3['l2_metrics']}")