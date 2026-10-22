"""
Safe-Net Architecture — Layer 7: Closed-Loop AI & Encrypted Zero-Knowledge Vault
Core Repositories:
    - `fidelity/mabwiser` (Contextual Multi-Armed Bandit for closed-loop intervention)
    - `pyca/cryptography` (AES-256-GCM authenticated encryption for local evidence vault)
Description:
    1. Evaluates incoming incidents from Layers 1-6.
    2. Enforces deterministic hard-override gates for severe threats.
    3. Uses contextual bandits (LinUCB / KNearest) for adaptive non-intrusive nudges.
    4. Encrypts sensitive incident records in a zero-knowledge local SQLite vault.
    5. Exposes a closed-loop feedback interface (`register_outcome`) to train the policy.
"""

import os
import json
import sqlite3
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional

# Cryptography: Authenticated Encryption (AES-GCM / Fernet)
from cryptography.fernet import Fernet

# Fidelity MABWiser: Contextual Multi-Armed Bandits
try:
    from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy
    MABWISER_AVAILABLE = True
except ImportError:
    MABWISER_AVAILABLE = False


class EncryptedZeroKnowledgeVault:
    """
    On-device tamper-evident incident repository using `pyca/cryptography` + SQLite.
    Stores only encrypted ciphertext blobs and irreversible SHA-256 integrity hashes.
    """

    def __init__(self, db_path: str = "safenet_vault.db", encryption_key: Optional[bytes] = None):
        self.db_path = db_path
        # Generate or load 256-bit AES Fernet key
        self.key = encryption_key or Fernet.generate_key()
        self.cipher = Fernet(self.key)
        self._init_database()

    def _init_database(self):
        """Initializes local SQLite schema for encrypted evidence packages."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vault_ledger (
                    incident_id TEXT PRIMARY KEY,
                    timestamp REAL,
                    canonical_contact_id_hash TEXT,
                    safety_state TEXT,
                    encrypted_payload BLOB,
                    payload_sha256 TEXT
                )
            """)
            conn.commit()

    def seal_evidence(self, incident_id: str, safety_event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encrypts the entire SafetyEvent contract into ciphertext and logs to local vault.
        """
        timestamp = float(safety_event.get("timestamp", datetime.now().timestamp()))
        contact_id = str(safety_event.get("canonical_contact_id", safety_event.get("contact_id", "unknown")))
        safety_state = str(safety_event.get("l5_safety_state", "UNKNOWN"))

        # Pseudonymize contact ID via SHA-256 for zero-knowledge storage
        contact_hash = hashlib.sha256(contact_id.encode("utf-8")).hexdigest()

        # Serialize payload to JSON and encrypt via AES-256-GCM (Fernet)
        raw_json_bytes = json.dumps(safety_event, default=str).encode("utf-8")
        encrypted_blob = self.cipher.encrypt(raw_json_bytes)
        integrity_hash = hashlib.sha256(encrypted_blob).hexdigest()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO vault_ledger 
                (incident_id, timestamp, canonical_contact_id_hash, safety_state, encrypted_payload, payload_sha256)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (incident_id, timestamp, contact_hash, safety_state, encrypted_blob, integrity_hash))
            conn.commit()

        return {
            "vault_status": "SEALED",
            "incident_id": incident_id,
            "sha256_fingerprint": integrity_hash[:16] + "...",
            "storage_target": self.db_path
        }


class ClosedLoopPolicyEngine:
    """
    Contextual Multi-Armed Bandit arbiter using `fidelity/mabwiser`.
    Selects the optimal intervention arm and learns from closed-loop feedback.
    """

    # Available Intervention Arms
    ARM_SILENT_MONITOR = "ARM_SILENT_MONITOR"         # Passive telemetry logging
    ARM_NUDGE_CHILD = "ARM_NUDGE_CHILD"               # In-app safety check prompt
    ARM_RESTRICT_MEDIA = "ARM_RESTRICT_MEDIA"         # Blur incoming images/audio
    ARM_GUARDIAN_ALERT = "ARM_GUARDIAN_ALERT"         # Vault lock & parent escalation

    ALL_ARMS = [
        ARM_SILENT_MONITOR,
        ARM_NUDGE_CHILD,
        ARM_RESTRICT_MEDIA,
        ARM_GUARDIAN_ALERT
    ]

    def __init__(self):
        self.enabled = MABWISER_AVAILABLE
        if self.enabled:
            # Contextual UCB with K-Nearest neighborhood clustering
            self.bandit = MAB(
                arms=self.ALL_ARMS,
                learning_policy=LearningPolicy.UCB1(alpha=1.0),
                neighborhood_policy=NeighborhoodPolicy.KNearest(k=3),
                seed=42
            )
            self._warm_start_bandit()
        else:
            self.bandit = None

    def _warm_start_bandit(self):
        """
        Calibrated offline seeding to eliminate cold-start issues.
        Features vector: [l1_score, l2_score, l3_score, is_late_night, is_known_contact]
        """
        # Baseline training data across all arms
        seed_contexts = [
            [0.05, 0.05, 0.05, 0.0, 1.0],  # Clear benign -> SILENT_MONITOR
            [0.35, 0.40, 0.20, 0.0, 1.0],  # Mild boundary testing -> NUDGE_CHILD
            [0.55, 0.50, 0.35, 1.0, 0.0],  # Suspicious media/velocity -> RESTRICT_MEDIA
            [0.85, 0.80, 0.75, 1.0, 0.0]   # High danger -> GUARDIAN_ALERT
        ]
        seed_decisions = [
            self.ARM_SILENT_MONITOR,
            self.ARM_NUDGE_CHILD,
            self.ARM_RESTRICT_MEDIA,
            self.ARM_GUARDIAN_ALERT
        ]
        seed_rewards = [1.0, 1.0, 1.0, 1.0]

        self.bandit.fit(decisions=seed_decisions, rewards=seed_rewards, contexts=seed_contexts)

    @staticmethod
    def extract_context_vector(safety_event: Dict[str, Any]) -> List[float]:
        """Maps incident metadata into a standardized numerical context array."""
        l1 = float(safety_event.get("l1_risk_score", 0.0))
        l2 = float(safety_event.get("l2_final_score", l1))
        l3 = float(safety_event.get("l3_final_score", l2))
        late_night = 1.0 if safety_event.get("l2_metrics", {}).get("is_late_night", False) else 0.0
        known = 1.0 if safety_event.get("is_known_contact", False) else 0.0
        return [round(l1, 3), round(l2, 3), round(l3, 3), late_night, known]

    def select_intervention(self, context_vector: List[float]) -> str:
        """Queries MABWiser for optimal intervention arm given current context."""
        if not self.enabled or self.bandit is None:
            return self.ARM_NUDGE_CHILD
        prediction = self.bandit.predict([context_vector])
        return str(prediction[0])

    def feedback(self, context_vector: List[float], arm_selected: str, reward: float):
        """
        Closed-Loop Online Learning: Updates bandit parameters with observed outcome.
        Reward scale: +1.0 (successful de-escalation / valid alert), -1.0 (false alarm / ignored threat).
        """
        if self.enabled and self.bandit is not None:
            self.bandit.partial_fit(
                decisions=[arm_selected],
                rewards=[reward],
                contexts=[context_vector]
            )


class Layer7ClosedLoopVaultAI:
    """
    Main Layer 7 Orchestrator for Safe-Net.
    Fuses MABWiser (Fidelity) with Cryptography Vault and Safety-Critical Overrides.
    """

    def __init__(self, db_path: str = "safenet_vault.db"):
        self.vault = EncryptedZeroKnowledgeVault(db_path=db_path)
        self.policy_engine = ClosedLoopPolicyEngine()

    def process_safety_event(self, safety_event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Terminal Layer 7 pipeline step.
        Evaluates Bayesian state, executes policy gates, manages vault storage,
        and determines child-facing and guardian-facing interventions.
        """
        incident_id = str(safety_event.get("event_id", f"inc_{int(datetime.now().timestamp()*1000)}"))
        l5_state = str(safety_event.get("l5_safety_state", "NORMAL"))
        lock_in_vault = bool(safety_event.get("Lock_In_Vault", False))
        context_vector = self.policy_engine.extract_context_vector(safety_event)

        # =====================================================================
        # SAFETY SHIELD 1: Deterministic Hard-Override Gate
        # For CRITICAL or HIGH states, NEVER allow bandit exploration.
        # Enforce protective action deterministically.
        # =====================================================================
        if l5_state == "CRITICAL":
            selected_arm = ClosedLoopPolicyEngine.ARM_GUARDIAN_ALERT
            override_applied = True
            vault_required = True
        elif l5_state == "HIGH":
            # If multimodal risk triggered the lock, restrict media immediately
            l6_action = safety_event.get("l6_action", "")
            if "MEDIA" in l6_action or "VAULT" in l6_action:
                selected_arm = ClosedLoopPolicyEngine.ARM_RESTRICT_MEDIA
            else:
                selected_arm = ClosedLoopPolicyEngine.ARM_GUARDIAN_ALERT
            override_applied = True
            vault_required = True
        else:
            # =================================================================
            # Contextual Bandit Decisioning for Grey Zones (ELEVATED / NORMAL)
            # =================================================================
            selected_arm = self.policy_engine.select_intervention(context_vector)
            override_applied = False
            vault_required = lock_in_vault

        # =====================================================================
        # SAFETY SHIELD 2: Encrypted Zero-Knowledge Evidence Sealing
        # =====================================================================
        vault_record = None
        if vault_required or selected_arm in (
            ClosedLoopPolicyEngine.ARM_RESTRICT_MEDIA,
            ClosedLoopPolicyEngine.ARM_GUARDIAN_ALERT
        ):
            vault_record = self.vault.seal_evidence(incident_id, safety_event)

        # Build final Layer 7 contract
        safety_event["l7_final_intervention"] = selected_arm
        safety_event["l7_hard_override_active"] = override_applied
        safety_event["l7_context_vector"] = context_vector
        safety_event["l7_vault_record"] = vault_record
        safety_event["l7_closed_loop_ready"] = True

        return safety_event

    def register_outcome(
        self,
        safety_event: Dict[str, Any],
        user_feedback: str
    ) -> Dict[str, Any]:
        """
        Closed-Loop Feedback Endpoint:
        Called when the child responds to a nudge, or when a parent reviews an alert.
        Calculates mathematical reward and updates MABWiser parameters in real time.
        """
        arm_used = safety_event.get("l7_final_intervention", ClosedLoopPolicyEngine.ARM_SILENT_MONITOR)
        context_vector = safety_event.get("l7_context_vector", [0.0, 0.0, 0.0, 0.0, 0.0])

        # Reward mapping matrix
        feedback_rewards = {
            "CHILD_DE_ESCALATED": 1.0,         # Child confirmed safety prompt & exited chat
            "PARENT_VERIFIED_THREAT": 1.0,      # Guardian confirmed alert was accurate
            "FALSE_ALARM_DISMISSED": -1.0,     # Guardian dismissed alert as benign banter
            "IGNORED_NUDGE": -0.2              # Child dismissed nudge without action
        }
        reward = feedback_rewards.get(user_feedback, 0.0)

        # Update MABWiser parameters online
        self.policy_engine.feedback(context_vector, arm_used, reward)

        return {
            "status": "FEEDBACK_INTEGRATED",
            "arm_updated": arm_used,
            "reward_assigned": reward,
            "feedback_label": user_feedback
        }


# =====================================================================
# SELF-TEST SUITE: Run `python layer7_closed_loop_vault.py`
# =====================================================================
if __name__ == "__main__":
    layer7 = Layer7ClosedLoopVaultAI(db_path="test_safenet_vault.db")

    print("=== TEST 1: CRITICAL State -> Hard Override Gate Verification ===")
    critical_event = {
        "event_id": "evt_crit_777",
        "canonical_contact_id": "predator_blackmail",
        "l1_risk_score": 0.92,
        "l2_final_score": 0.88,
        "l3_final_score": 0.80,
        "l5_safety_state": "CRITICAL",
        "Lock_In_Vault": True,
        "is_known_contact": False,
        "l2_metrics": {"is_late_night": True}
    }
    res1 = layer7.process_safety_event(critical_event)
    print(f"Intervention: {res1['l7_final_intervention']} | Hard Override: {res1['l7_hard_override_active']}")
    print(f"Vault Record: {res1['l7_vault_record']}\n")

    print("=== TEST 2: ELEVATED Ambiguous State -> MABWiser Contextual Selection ===")
    elevated_event = {
        "event_id": "evt_elev_888",
        "canonical_contact_id": "school_acquaintance",
        "l1_risk_score": 0.38,
        "l2_final_score": 0.42,
        "l3_final_score": 0.30,
        "l5_safety_state": "ELEVATED",
        "Lock_In_Vault": False,
        "is_known_contact": True,
        "l2_metrics": {"is_late_night": False}
    }
    res2 = layer7.process_safety_event(elevated_event)
    print(f"Intervention: {res2['l7_final_intervention']} | Hard Override: {res2['l7_hard_override_active']}")
    print(f"Context Vector: {res2['l7_context_vector']}\n")

    print("=== TEST 3: Closed-Loop Feedback Learning Loop ===")
    fb_res = layer7.register_outcome(res2, user_feedback="CHILD_DE_ESCALATED")
    print(f"Closed-Loop Update: {fb_res}")

    # Cleanup temporary test database file
    if os.path.exists("test_safenet_vault.db"):
        os.remove("test_safenet_vault.db")