"""
Safe-Net Architecture — Layer 3: Relational & Group AI
Description:
    Builds and maintains a lightweight on-device dynamic ego-network using NetworkX.
    Detects group bullying (dogpiling), social isolation, and cross-platform
    identity variants without heavy backend entity resolution engines.
"""

import re
import difflib
from datetime import datetime
from typing import Dict, Any, List, Optional
import networkx as nx


class LightweightIdentityResolver:
    """
    Zero-setup on-device handle canonicalizer.
    Replaces heavy probabilistic engines (like Splink) to link cross-app usernames
    (e.g., '@alex_99' on Discord vs 'alex.99' on Instagram) with zero dependencies.
    """

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        # Maps canonical_id -> set of observed raw usernames/handles
        self.canonical_map: Dict[str, str] = {}

    @staticmethod
    def _clean_handle(handle: str) -> str:
        """Strip platform prefixes, dots, underscores, and lowercase."""
        return re.sub(r"[._\-@]", "", handle).strip().lower()

    def resolve(self, raw_contact_id: str) -> str:
        clean = self._clean_handle(raw_contact_id)
        if not clean:
            return raw_contact_id

        # Direct match check
        if raw_contact_id in self.canonical_map:
            return self.canonical_map[raw_contact_id]

        # Fuzzy match against known canonical keys
        for known_raw, canonical in self.canonical_map.items():
            known_clean = self._clean_handle(known_raw)
            ratio = difflib.SequenceMatcher(None, clean, known_clean).ratio()
            if ratio >= self.similarity_threshold:
                self.canonical_map[raw_contact_id] = canonical
                return canonical

        # New distinct entity
        canonical_id = f"entity_{clean}"
        self.canonical_map[raw_contact_id] = canonical_id
        return canonical_id


class Layer3RelationalAI:
    """
    Main Layer 3 Relational Graph Analyzer.
    Connects Layer 2 -> Layer 3 -> Layer 4.
    """

    # Safety Guardrail Constants
    DOGPILING_TIME_WINDOW_SEC = 3600  # 1 hour window for coordinated group attacks
    MIN_ATTACKERS_FOR_DOGPILE = 3     # Must have at least 3 distinct incoming actors
    GROUP_TOXICITY_CEILING = 0.45      # Minimum avg toxicity to avoid false group flags
    ISOLATION_DROP_THRESHOLD = 0.60   # 60% drop in normal incoming interaction volume
    EDGE_RETENTION_HOURS = 24 * 14     # 14 days memory retention for on-device graph

    def __init__(self, child_id: str = "child_user"):
        self.child_id = child_id
        # Directed multigraph to retain directional interaction metadata
        self.graph = nx.DiGraph()
        self.graph.add_node(self.child_id, role="ego")
        self.resolver = LightweightIdentityResolver()
        self.recent_interactions: List[Dict[str, Any]] = []

    def _prune_expired_history(self, current_timestamp: float):
        """Prevents memory bloat on edge hardware by removing stale events, edges, and orphan nodes."""
        cutoff = current_timestamp - (self.EDGE_RETENTION_HOURS * 3600)
        
        # 1. Prune the recent interactions list
        self.recent_interactions = [
            ev for ev in self.recent_interactions if ev["timestamp"] >= cutoff
        ]

        # 2. Prune stale edges from the NetworkX Graph
        stale_edges = [
            (u, v) for u, v, data in self.graph.edges(data=True)
            if data.get("last_timestamp", 0) < cutoff
        ]
        self.graph.remove_edges_from(stale_edges)

        # 3. Prune orphan nodes (except the child ego node)
        orphan_nodes = [
            node for node in self.graph.nodes()
            if node != self.child_id and self.graph.degree(node) == 0
        ]
        self.graph.remove_nodes_from(orphan_nodes)

    def _detect_dogpiling(self, current_timestamp: float) -> Dict[str, Any]:
        """
        Guardrail: Detects coordinated incoming attacks within a 1-hour rolling window.
        Only alerts if BOTH unique attacker count and mean toxicity cross thresholds.
        """
        window_cutoff = current_timestamp - self.DOGPILING_TIME_WINDOW_SEC
        incoming_in_window = [
            ev for ev in self.recent_interactions
            if ev["timestamp"] >= window_cutoff and ev["direction"] == "incoming"
        ]

        distinct_senders = {ev["canonical_id"] for ev in incoming_in_window}
        if len(distinct_senders) < self.MIN_ATTACKERS_FOR_DOGPILE:
            return {"dogpile_detected": False, "attacker_count": len(distinct_senders), "group_risk": 0.0}

        avg_l2_score = sum(ev["l2_score"] for ev in incoming_in_window) / len(incoming_in_window)

        # Benign Chat Guardrail: Group activity with safe scores is not dogpiling
        if avg_l2_score < self.GROUP_TOXICITY_CEILING:
            return {"dogpile_detected": False, "attacker_count": len(distinct_senders), "group_risk": round(avg_l2_score, 4)}

        return {
            "dogpile_detected": True,
            "attacker_count": len(distinct_senders),
            "group_risk": round(min(1.0, avg_l2_score * 1.25), 4)
        }

    def _evaluate_social_isolation(self) -> Dict[str, Any]:
        """
        Monitors graph density and incoming degree connectivity around the child.
        """
        if self.graph.number_of_nodes() <= 1:
            return {"isolation_risk": 0.0, "active_degree": 0}

        in_degree = self.graph.in_degree(self.child_id)
        out_degree = self.graph.out_degree(self.child_id)

        # Child sending messages out but receiving zero replies back across multiple peers
        isolation_risk = 0.0
        if out_degree >= 5 and in_degree == 0:
            isolation_risk = 0.70
        elif out_degree >= 3 and in_degree <= 1:
            isolation_risk = 0.40

        return {
            "isolation_risk": round(isolation_risk, 4),
            "in_degree": in_degree,
            "out_degree": out_degree
        }

    def process_safety_event(self, safety_event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main pipeline entry point receiving the contract from Layer 2.
        """
        timestamp = float(safety_event.get("timestamp", datetime.now().timestamp()))
        raw_contact_id = str(safety_event.get("contact_id", "unknown_user"))
        direction = str(safety_event.get("direction", "incoming"))  # incoming or outgoing
        l2_score = float(safety_event.get("l2_final_score", safety_event.get("l1_risk_score", 0.0)))

        # 1. Identity Resolution
        canonical_id = self.resolver.resolve(raw_contact_id)
        safety_event["canonical_contact_id"] = canonical_id

        # 2. Prune old history & append current event
        self._prune_expired_history(timestamp)
        event_record = {
            "canonical_id": canonical_id,
            "timestamp": timestamp,
            "direction": direction,
            "l2_score": l2_score
        }
        self.recent_interactions.append(event_record)

        # 3. Update NetworkX Graph Topology
        if not self.graph.has_node(canonical_id):
            self.graph.add_node(canonical_id, role="peer")

        if direction == "incoming":
            u, v = canonical_id, self.child_id
        else:
            u, v = self.child_id, canonical_id

        if self.graph.has_edge(u, v):
            self.graph[u][v]["weight"] += 1
            self.graph[u][v]["last_score"] = l2_score
            self.graph[u][v]["last_timestamp"] = timestamp
        else:
            self.graph.add_edge(u, v, weight=1, last_score=l2_score, last_timestamp=timestamp)

        # 4. Run Relational Guardrail Checks
        dogpile_metrics = self._detect_dogpiling(timestamp)
        isolation_metrics = self._evaluate_social_isolation()

        # 5. Composite Layer 3 Scoring & Escalation Gate
        dogpile_boost = 0.35 if dogpile_metrics["dogpile_detected"] else 0.0
        isolation_boost = isolation_metrics["isolation_risk"] * 0.20

        l3_final_score = min(1.0, max(l2_score, dogpile_metrics["group_risk"]) + dogpile_boost + isolation_boost)
        l3_final_score = round(l3_final_score, 4)

        if dogpile_metrics["dogpile_detected"] or l3_final_score >= 0.70:
            l3_action = "ESCALATE_TO_L4_HIGH"
            propagate_to_l4 = True
        elif l3_final_score >= 0.40:
            l3_action = "MONITOR_RELATIONAL"
            propagate_to_l4 = True
        else:
            l3_action = "PASS_RELATIONAL_SAFE"
            propagate_to_l4 = bool(safety_event.get("propagate_to_l3", False))

        # 6. Enrich Contract for Layer 4 (Temporal AI)
        safety_event["l3_final_score"] = l3_final_score
        safety_event["l3_action"] = l3_action
        safety_event["propagate_to_l4"] = propagate_to_l4
        safety_event["l3_relational_metrics"] = {
            "canonical_contact_id": canonical_id,
            "dogpile_detected": dogpile_metrics["dogpile_detected"],
            "attacker_count": dogpile_metrics["attacker_count"],
            "group_risk": dogpile_metrics["group_risk"],
            "isolation_risk": isolation_metrics["isolation_risk"],
            "graph_nodes": self.graph.number_of_nodes(),
            "graph_edges": self.graph.number_of_edges()
        }

        return safety_event


# =====================================================================
# SELF-TEST SUITE: Run `python layer3_relational.py` to test scenarios
# =====================================================================
if __name__ == "__main__":
    layer3 = Layer3RelationalAI(child_id="child_reiyan")
    now = datetime.now().timestamp()

    print("=== TEST 1: Cross-App Canonical Match Check ===")
    evt_ig = {
        "contact_id": "alex_99",
        "l1_risk_score": 0.45,
        "l2_final_score": 0.50,
        "direction": "incoming",
        "timestamp": now
    }
    res_ig = layer3.process_safety_event(evt_ig)
    
    evt_discord = {
        "contact_id": "@alex.99",
        "l1_risk_score": 0.50,
        "l2_final_score": 0.55,
        "direction": "incoming",
        "timestamp": now + 60
    }
    res_discord = layer3.process_safety_event(evt_discord)
    print(f"Original: alex_99 -> Resolved: {res_ig['canonical_contact_id']}")
    print(f"Original: @alex.99 -> Resolved: {res_discord['canonical_contact_id']}")
    print(f"Matched successfully: {res_ig['canonical_contact_id'] == res_discord['canonical_contact_id']}\n")

    print("=== TEST 2: Dogpiling Attack Check (3 Senders within 1 Hour) ===")
    bullies = ["troll_alpha", "troll_beta", "troll_gamma"]
    for idx, bully in enumerate(bullies):
        dogpile_evt = {
            "contact_id": bully,
            "l1_risk_score": 0.60,
            "l2_final_score": 0.65,
            "direction": "incoming",
            "timestamp": now + (idx * 300)
        }
        res_dp = layer3.process_safety_event(dogpile_evt)

    print(f"Action: {res_dp['l3_action']} | Score: {res_dp['l3_final_score']}")
    print(f"Metrics: {res_dp['l3_relational_metrics']}")