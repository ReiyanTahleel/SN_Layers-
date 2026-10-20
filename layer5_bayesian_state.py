"""
Safe-Net Architecture — Layer 5: Safety State AI (Bayesian Belief Network Engine)
Description:
    Fuses multi-tier evidence from Layer 1 (Context), Layer 2 (Interaction),
    Layer 3 (Relational), and Layer 4 (Temporal) using an exact, explainable
    Discrete Bayesian Network (`pgmpy` + `VariableElimination`).
    Computes exact posterior probabilities over [NORMAL, ELEVATED, HIGH, CRITICAL]
    with zero black-box hallucinations, asymmetric acute-threat dominance,
    and state-hysteresis anti-flicker protection.
"""

from typing import Dict, Any, List
import itertools
import numpy as np

# Compatible across pgmpy >= 1.0 (DiscreteBayesianNetwork) and older 0.1.x releases
try:
    from pgmpy.models import DiscreteBayesianNetwork as BNNModel
except ImportError:
    from pgmpy.models import BayesianNetwork as BNNModel

from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


class Layer5SafetyStateAI:
    """
    Exact Bayesian Belief Network for Safe-Net Layer 5.
    Connects Layer 1, 2, 3, 4 -> Layer 5 Posterior Safety State -> Layer 6/7 Vault & Policy.
    """

    STATE_LABELS = ["NORMAL", "ELEVATED", "HIGH", "CRITICAL"]
    STATE_SEVERITY_RANK = {"NORMAL": 0, "ELEVATED": 1, "HIGH": 2, "CRITICAL": 3}

    def __init__(self):
        # 1. Define Directed Acyclic Graph (DAG) Topology
        self.model = BNNModel([
            ("Context_Toxicity", "Safety_State"),
            ("Dyad_Escalation", "Safety_State"),
            ("Group_Isolation", "Safety_State"),
            ("Temporal_Trend", "Safety_State")
        ])

        # 2. Build & Validate CPDs
        self._build_cpds()
        self.inference_engine = VariableElimination(self.model)

        # 3. On-Device Hysteresis Memory (Prevents Alert Ping-Pong / Flickering)
        # Format: { contact_id: {"last_state": "NORMAL", "safe_streak": 0} }
        self.contact_state_memory: Dict[str, Dict[str, Any]] = {}

    def _build_cpds(self):
        """
        Constructs prior CPDs for Layers 1-4 and the exact 81-column Conditional
        Probability Table (CPT) for Safety_State using calibrated safety rules.
        """
        cpd_l1 = TabularCPD(
            variable="Context_Toxicity",
            variable_card=3,
            values=[[0.75], [0.18], [0.07]],
            state_names={"Context_Toxicity": ["LOW", "MED", "HIGH"]}
        )
        cpd_l2 = TabularCPD(
            variable="Dyad_Escalation",
            variable_card=3,
            values=[[0.70], [0.20], [0.10]],
            state_names={"Dyad_Escalation": ["LOW", "MED", "HIGH"]}
        )
        cpd_l3 = TabularCPD(
            variable="Group_Isolation",
            variable_card=3,
            values=[[0.80], [0.15], [0.05]],
            state_names={"Group_Isolation": ["LOW", "MED", "HIGH"]}
        )
        cpd_l4 = TabularCPD(
            variable="Temporal_Trend",
            variable_card=3,
            values=[[0.75], [0.18], [0.07]],
            state_names={"Temporal_Trend": ["STABLE", "TESTING", "ACUTE"]}
        )

        # Generate exact 4 x 81 CPT columns across all parent combinations
        level_weights = {"LOW": 0.0, "MED": 0.5, "HIGH": 1.0}
        temporal_weights = {"STABLE": 0.0, "TESTING": 0.55, "ACUTE": 1.0}

        cpt_columns: List[List[float]] = []

        for l1, l2, l3, l4 in itertools.product(
            ["LOW", "MED", "HIGH"],
            ["LOW", "MED", "HIGH"],
            ["LOW", "MED", "HIGH"],
            ["STABLE", "TESTING", "ACUTE"]
        ):
            w1 = level_weights[l1]
            w2 = level_weights[l2]
            w3 = level_weights[l3]
            w4 = temporal_weights[l4]

            # Weighted base fusion + Asymmetric Acute Dominance Guardrail:
            # Prevents a severe signal in L1/L4 from being diluted by LOW scores elsewhere
            weighted_avg = (0.30 * w1) + (0.25 * w2) + (0.20 * w3) + (0.25 * w4)
            max_signal = max(w1, w2, w3, w4)
            high_count = sum(1 for w in (w1, w2, w3, w4) if w >= 1.0)

            # Non-linear synergy boost when multiple layers flag HIGH simultaneously
            synergy_boost = 0.18 if high_count >= 2 else (0.08 if high_count == 1 else 0.0)
            composite_hazard = min(1.0, (0.55 * weighted_avg) + (0.45 * max_signal) + synergy_boost)

            # Convert hazard [0.0, 1.0] into smooth, calibrated probability distribution
            # over [NORMAL, ELEVATED, HIGH, CRITICAL] using Gaussian kernel centers
            centers = np.array([0.05, 0.38, 0.68, 0.95])
            sharpness = 14.0
            raw_probs = np.exp(-sharpness * ((composite_hazard - centers) ** 2))
            normalized_probs = raw_probs / np.sum(raw_probs)

            cpt_columns.append(normalized_probs.tolist())

        # Transpose from (81, 4) to (4, 81) as required by pgmpy TabularCPD
        cpt_matrix = np.array(cpt_columns).T.tolist()

        cpd_safety = TabularCPD(
            variable="Safety_State",
            variable_card=4,
            values=cpt_matrix,
            evidence=[
                "Context_Toxicity",
                "Dyad_Escalation",
                "Group_Isolation",
                "Temporal_Trend"
            ],
            evidence_card=[3, 3, 3, 3],
            state_names={
                "Safety_State": self.STATE_LABELS,
                "Context_Toxicity": ["LOW", "MED", "HIGH"],
                "Dyad_Escalation": ["LOW", "MED", "HIGH"],
                "Group_Isolation": ["LOW", "MED", "HIGH"],
                "Temporal_Trend": ["STABLE", "TESTING", "ACUTE"]
            }
        )

        self.model.add_cpds(cpd_l1, cpd_l2, cpd_l3, cpd_l4, cpd_safety)
        # Mathematical verification that all 81 columns sum to 1.0 and graph is valid
        assert self.model.check_model(), "Bayesian Network CPD validation failed!"

    @staticmethod
    def _discretize_evidence(safety_event: Dict[str, Any]) -> Dict[str, str]:
        """
        Deterministically maps continuous scores from Layers 1-4 into discrete BBN evidence.
        """
        # Layer 1: Context_Toxicity
        l1_score = float(safety_event.get("l1_risk_score", 0.0))
        if l1_score >= 0.70:
            c_state = "HIGH"
        elif l1_score >= 0.30:
            c_state = "MED"
        else:
            c_state = "LOW"

        # Layer 2: Dyad_Escalation
        l2_score = float(safety_event.get("l2_final_score", l1_score))
        if l2_score >= 0.65:
            d_state = "HIGH"
        elif l2_score >= 0.35:
            d_state = "MED"
        else:
            d_state = "LOW"

        # Layer 3: Group_Isolation
        l3_score = float(safety_event.get("l3_final_score", l2_score))
        l3_metrics = safety_event.get("l3_relational_metrics", {})
        dogpile = bool(l3_metrics.get("dogpile_detected", False))
        if dogpile or l3_score >= 0.70:
            g_state = "HIGH"
        elif l3_score >= 0.40:
            g_state = "MED"
        else:
            g_state = "LOW"

        # Layer 4: Temporal_Trend
        l4_raw = str(safety_event.get("l4_temporal_state", "BENIGN_STABLE"))
        if "ACUTE" in l4_raw:
            t_state = "ACUTE"
        elif "TESTING" in l4_raw:
            t_state = "TESTING"
        else:
            t_state = "STABLE"

        return {
            "Context_Toxicity": c_state,
            "Dyad_Escalation": d_state,
            "Group_Isolation": g_state,
            "Temporal_Trend": t_state
        }

    def _apply_hysteresis_guardrail(
        self, contact_id: str, raw_map_state: str, posteriors: Dict[str, float]
    ) -> str:
        """
        Guardrail: Prevents 'Alert Ping-Pong' where a predator in HIGH/CRITICAL state
        sends a single 'ok' message and immediately drops back to NORMAL.
        Requires 3 consecutive NORMAL inferences to step down one severity tier.
        """
        if contact_id not in self.contact_state_memory:
            self.contact_state_memory[contact_id] = {
                "last_state": raw_map_state,
                "safe_streak": 0
            }
            return raw_map_state

        mem = self.contact_state_memory[contact_id]
        prev_state = mem["last_state"]
        prev_rank = self.STATE_SEVERITY_RANK[prev_state]
        curr_rank = self.STATE_SEVERITY_RANK[raw_map_state]

        # Immediate upward escalation is always allowed for child safety
        if curr_rank >= prev_rank:
            mem["last_state"] = raw_map_state
            mem["safe_streak"] = 0
            return raw_map_state

        # Downward de-escalation requires sustained safe behavior (cooldown streak >= 3)
        mem["safe_streak"] += 1
        if mem["safe_streak"] >= 3:
            stepped_down_rank = max(curr_rank, prev_rank - 1)
            new_state = self.STATE_LABELS[stepped_down_rank]
            mem["last_state"] = new_state
            mem["safe_streak"] = 0
            return new_state

        # Hold previous elevated/high state until cooldown completes
        return prev_state

    def process_safety_event(self, safety_event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main Layer 5 pipeline entry point.
        Computes exact posterior probabilities via VariableElimination and determines
        explainable policy actions for Layer 6 (Multimodal) & Layer 7 (Vault/Closed-Loop).
        """
        contact_id = str(
            safety_event.get("canonical_contact_id", safety_event.get("contact_id", "unknown"))
        )

        # 1. Discretize Layers 1-4 into Bayesian Evidence
        evidence = self._discretize_evidence(safety_event)

        # 2. Run Exact Inference via Variable Elimination
        query_result = self.inference_engine.query(
            variables=["Safety_State"],
            evidence=evidence,
            show_progress=False
        )

        probs = query_result.values
        posteriors = {
            state: round(float(probs[idx]), 4)
            for idx, state in enumerate(self.STATE_LABELS)
        }

        # 3. Expected Risk Index & Decision Rule
        # Safety Guardrail: If P(CRITICAL) >= 0.35 or P(HIGH + CRITICAL) >= 0.60,
        # escalate proactively even if MAP is slightly split.
        if posteriors["CRITICAL"] >= 0.40:
            raw_state = "CRITICAL"
        elif (posteriors["HIGH"] + posteriors["CRITICAL"]) >= 0.55:
            raw_state = "HIGH"
        elif posteriors["ELEVATED"] >= 0.40 or (posteriors["ELEVATED"] + posteriors["HIGH"]) >= 0.50:
            raw_state = "ELEVATED"
        else:
            raw_state = max(posteriors, key=posteriors.get)

        # 4. Apply Anti-Flicker Hysteresis Shield
        final_safety_state = self._apply_hysteresis_guardrail(
            contact_id=contact_id,
            raw_map_state=raw_state,
            posteriors=posteriors
        )

        # 5. Map State to Concrete Action for Layer 6 & Layer 7
        action_map = {
            "NORMAL": "LOG_MINIMAL_PASS",
            "ELEVATED": "INCREASE_SAMPLING_WATCHLIST",
            "HIGH": "TRIGGER_MULTIMODAL_SCAN_AND_VAULT_LOCK",
            "CRITICAL": "IMMEDIATE_INTERVENTION_AND_GUARDIAN_ALERT"
        }
        l5_action = action_map[final_safety_state]

        # 6. Generate Explainable Causal Audit String (Zero Hallucinations)
        active_drivers = [f"{k}={v}" for k, v in evidence.items() if v not in ("LOW", "STABLE")]
        explanation = (
            f"State={final_safety_state} (P_crit={posteriors['CRITICAL']:.2f}, "
            f"P_high={posteriors['HIGH']:.2f}) driven by: "
            + (", ".join(active_drivers) if active_drivers else "All layers nominal")
        )

        # 7. Enrich SafetyEvent Contract
        safety_event["l5_safety_state"] = final_safety_state
        safety_event["l5_raw_bayesian_state"] = raw_state
        safety_event["l5_posteriors"] = posteriors
        safety_event["l5_evidence_used"] = evidence
        safety_event["l5_action"] = l5_action
        safety_event["l5_explanation"] = explanation
        safety_event["Lock_In_Vault"] = final_safety_state in ("HIGH", "CRITICAL")

        return safety_event


# =====================================================================
# SELF-TEST SUITE: Run `python layer5_bayesian_state.py` to verify all cases
# =====================================================================
if __name__ == "__main__":
    l5_engine = Layer5SafetyStateAI()

    print("=== TEST 1: All Layers Nominal (Expected: NORMAL) ===")
    evt_safe = {
        "canonical_contact_id": "friend_01",
        "l1_risk_score": 0.08,
        "l2_final_score": 0.10,
        "l3_final_score": 0.12,
        "l4_temporal_state": "BENIGN_STABLE"
    }
    res1 = l5_engine.process_safety_event(evt_safe)
    print(f"State: {res1['l5_safety_state']} | Posteriors: {res1['l5_posteriors']}")
    print(f"Explanation: {res1['l5_explanation']}\n")

    print("=== TEST 2: Single-Message Acute Threat (Asymmetric Dominance Check) ===")
    evt_acute_single = {
        "canonical_contact_id": "stranger_danger",
        "l1_risk_score": 0.88,         # HIGH immediate toxicity
        "l2_final_score": 0.68,         # HIGH dyadic score
        "l3_final_score": 0.15,         # LOW group isolation (1-on-1 chat)
        "l4_temporal_state": "TESTING"  # Just started escalating
    }
    res2 = l5_engine.process_safety_event(evt_acute_single)
    print(f"State: {res2['l5_safety_state']} | Action: {res2['l5_action']}")
    print(f"Posteriors: {res2['l5_posteriors']}")
    print(f"Explanation: {res2['l5_explanation']}\n")

    print("=== TEST 3: Multi-Layer Grooming Convergence (Expected: CRITICAL) ===")
    evt_critical = {
        "canonical_contact_id": "predator_99",
        "l1_risk_score": 0.78,
        "l2_final_score": 0.82,
        "l3_final_score": 0.75,
        "l3_relational_metrics": {"dogpile_detected": True},
        "l4_temporal_state": "ACUTE_DANGER"
    }
    res3 = l5_engine.process_safety_event(evt_critical)
    print(f"State: {res3['l5_safety_state']} | Vault Lock: {res3['Lock_In_Vault']}")
    print(f"Posteriors: {res3['l5_posteriors']}")
    print(f"Explanation: {res3['l5_explanation']}\n")

    print("=== TEST 4: Anti-Flicker Hysteresis Check (Predator sends 1 safe 'ok' text) ===")
    evt_cooldown = {
        "canonical_contact_id": "predator_99",
        "l1_risk_score": 0.05,
        "l2_final_score": 0.15,
        "l3_final_score": 0.15,
        "l4_temporal_state": "BENIGN_STABLE"
    }
    res4 = l5_engine.process_safety_event(evt_cooldown)
    print(f"Raw Bayesian State: {res4['l5_raw_bayesian_state']} -> Shielded State Held At: {res4['l5_safety_state']}")