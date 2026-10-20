"""
Safe-Net Architecture — Layer 4: Temporal AI (Longitudinal Risk & Trend Shift Engine)
Description:
    Tracks longitudinal behavioral trajectories per contact over time.
    Uses a pre-calibrated Gaussian Hidden Markov Model (hmmlearn) for zero-latency
    deterministic regime detection (Normal -> Escalation -> Acute Danger).
    Runs purely in inference mode on-device to avoid model drift and preserve battery.
"""

from datetime import datetime
from typing import Dict, Any, List
import numpy as np
from hmmlearn import hmm


class PrecalibratedSafetyHMM:
    """
    Fixed-parameter Gaussian HMM for edge inference.
    Hidden States:
        0: BENIGN_STABLE (Normal interactions)
        1: BOUNDARY_TESTING (Gradual grooming / subtle creep)
        2: ACUTE_DANGER (Aggressive escalation / coercive control)
    """

    def __init__(self):
        # 3 hidden states
        self.model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=10)
        
        # Initial state probabilities: Vast majority start as Benign
        self.model.startprob_ = np.array([0.90, 0.09, 0.01])

        # Transition Probability Matrix (A):
        # High self-retention, with directional paths toward escalation
        self.model.transmat_ = np.array([
            [0.85, 0.13, 0.02],  # From Benign: mostly stays benign
            [0.10, 0.75, 0.15],  # From Testing: can de-escalate (0.10) or escalate (0.15)
            [0.02, 0.08, 0.90]   # From Acute: stays locked in high risk until audited
        ])

        # Emission Means (B): [l3_risk_score, msg_velocity_factor]
        self.model.means_ = np.array([
            [0.10, 0.20],  # State 0: Low risk, normal pace
            [0.45, 0.60],  # State 1: Moderate risk, urgent/pushy pace
            [0.80, 0.85]   # State 2: Severe risk, high frequency
        ])

       # FIXED: Shape is (n_components=3, n_features=2)
        self.model.covars_ = np.array([
            [0.03, 0.04],  # State 0 variances (risk variance, velocity variance)
            [0.04, 0.05],  # State 1 variances
            [0.02, 0.03]   # State 2 variances
        ])

    def predict_state_sequence(self, observations: np.ndarray) -> np.ndarray:
        """Runs the Viterbi algorithm to decode the most likely path of hidden states."""
        if len(observations) == 0:
            return np.array([0])
        return self.model.predict(observations)

    def compute_state_posteriors(self, observations: np.ndarray) -> np.ndarray:
        """Computes current probability distribution across [Benign, Testing, Acute]."""
        return self.model.predict_proba(observations)[-1]


class Layer4TemporalAI:
    """
    Main Layer 4 Temporal Orchestrator.
    Connects Layer 3 -> Layer 4 -> Layer 5 (Safety State Engine).
    """

    HISTORY_WINDOW_SIZE = 30  # Sliding window of observations per contact on edge

    def __init__(self):
        self.hmm_engine = PrecalibratedSafetyHMM()
        # In-memory history per contact: { contact_id: [ [score, velocity], ... ] }
        self.contact_timelines: Dict[str, List[List[float]]] = {}
        self.state_names = {
            0: "BENIGN_STABLE",
            1: "BOUNDARY_TESTING",
            2: "ACUTE_DANGER"
        }

    def process_safety_event(self, safety_event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Receives enriched SafetyEvent from Layer 3, evaluates longitudinal trajectory,
        and determines temporal escalation.
        """
        contact_id = str(safety_event.get("canonical_contact_id", safety_event.get("contact_id", "unknown")))
        
        # Take Layer 3 composite score, falling back to L2 or L1
        l3_score = float(safety_event.get("l3_final_score", safety_event.get("l2_final_score", 0.0)))
        
        # Normalize velocity (0.0 to 1.0)
        raw_velocity = float(safety_event.get("msg_velocity", 1.0))
        velocity_factor = min(max(raw_velocity / 12.0, 0.0), 1.0)

        # 1. Update contact observation buffer
        if contact_id not in self.contact_timelines:
            self.contact_timelines[contact_id] = []

        timeline = self.contact_timelines[contact_id]
        timeline.append([round(l3_score, 4), round(velocity_factor, 4)])

        # Keep buffer bounded to prevent on-device memory growth
        if len(timeline) > self.HISTORY_WINDOW_SIZE:
            timeline.pop(0)

        obs_array = np.array(timeline)

        # 2. Decode Temporal Behavioral Regime via HMM
        hidden_states = self.hmm_engine.predict_state_sequence(obs_array)
        current_state_idx = int(hidden_states[-1])
        current_state_label = self.state_names[current_state_idx]
        posteriors = self.hmm_engine.compute_state_posteriors(obs_array)

        # 3. Temporal Trajectory / Velocity Slope Calculation
        # Check if risk is accelerating over the last 3 interactions
        if len(timeline) >= 3:
            recent_scores = [pt[0] for pt in timeline[-3:]]
            trajectory_slope = round((recent_scores[-1] - recent_scores[0]) / 2.0, 4)
        else:
            trajectory_slope = 0.0

        # 4. Layer 4 Safety Action Decision Gate
        if current_state_idx == 2 or (current_state_idx == 1 and trajectory_slope > 0.15):
            l4_action = "ESCALATE_TO_VAULT"  # Escalates to encrypted storage & safety state
            propagate_to_l5 = True
        elif current_state_idx == 1 or trajectory_slope > 0.08:
            l4_action = "MONITOR_HIGH_SURVEILLANCE"
            propagate_to_l5 = True
        else:
            l4_action = "TEMPORAL_PASS_SAFE"
            propagate_to_l5 = bool(safety_event.get("propagate_to_l4", False))

        # 5. Enrich Contract for Downstream Layers (Layer 5+)
        safety_event["l4_temporal_state"] = current_state_label
        safety_event["l4_action"] = l4_action
        safety_event["propagate_to_l5"] = propagate_to_l5
        safety_event["l4_temporal_metrics"] = {
            "current_state_index": current_state_idx,
            "state_label": current_state_label,
            "state_probabilities": {
                "benign": round(float(posteriors[0]), 4),
                "testing": round(float(posteriors[1]), 4),
                "acute": round(float(posteriors[2]), 4)
            },
            "trajectory_slope": trajectory_slope,
            "window_observations_evaluated": len(timeline)
        }

        return safety_event


# =====================================================================
# SELF-TEST SUITE: Run `python layer4_temporal.py` to verify transitions
# =====================================================================
if __name__ == "__main__":
    l4_engine = Layer4TemporalAI()
    cid = "suspect_contact_007"

    print("=== TEST 1: Baseline Normal Chat (State 0 Expected) ===")
    evt1 = {
        "canonical_contact_id": cid,
        "l3_final_score": 0.08,
        "msg_velocity": 1.5,
        "propagate_to_l4": False
    }
    res1 = l4_engine.process_safety_event(evt1)
    print(f"State: {res1['l4_temporal_state']} | Action: {res1['l4_action']}")
    print(f"Probabilities: {res1['l4_temporal_metrics']['state_probabilities']}\n")

    print("=== TEST 2: Multi-Day Escalating Grooming Pattern ===")
    escalation_stream = [
        (0.20, 2.0),
        (0.35, 4.0),
        (0.52, 6.0),
        (0.68, 9.0),
        (0.85, 11.0)
    ]

    for idx, (score, vel) in enumerate(escalation_stream):
        evt = {
            "canonical_contact_id": cid,
            "l3_final_score": score,
            "msg_velocity": vel,
            "propagate_to_l4": True
        }
        res = l4_engine.process_safety_event(evt)
        print(f"Step {idx+1} [Score {score:.2f}, Vel {vel}] -> State: {res['l4_temporal_state']} | Action: {res['l4_action']} | Slope: {res['l4_temporal_metrics']['trajectory_slope']}")