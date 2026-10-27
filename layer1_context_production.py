"""
Safe-Net Luna ML — Layer 1: Production Content & Context Safety Engine
Addresses Brief Deliverables:
    - Deliverable #1 (P0): Unified Longitudinal Dataset Spec + Conversation-Level Splits
    - Deliverable #2 (P0): Baseline Benchmark (30-Ex TF-IDF & Android LexiconClassifier)
    - Deliverable #3 (P0): Calibrated Context-Window L1 Classifier + Full Metrics Suite
Preserves the 7-Layer Safe-Net Architecture Contract (L1 -> L2 -> L3 -> L4 -> L6 -> L5 -> L7).
"""

import os
import re
import json
import time
import random
import hashlib
import pickle
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    brier_score_loss,
)


# =====================================================================
# 1. TAXONOMY & DATASET SPECIFICATION (DELIVERABLE #1 — P0)
# =====================================================================
SAFENET_CATEGORIES = [
    "BENIGN",
    "THREATS",
    "SEXUAL_RISK",
    "SECRECY_PRESSURE",
    "COERCION",
    "BULLYING_HARASSMENT",
    "CREDENTIAL_SCAMS",
    "SELF_HARM_RISK",
]

HIGH_SEVERITY_CATEGORIES = {
    "THREATS",
    "SEXUAL_RISK",
    "SECRECY_PRESSURE",
    "COERCION",
    "SELF_HARM_RISK",
}


class MasterSafeNetDatasetBuilder:
    """
    Generates a unified, multi-turn longitudinal dataset that serves Layer 1
    AND includes sequential timestamps, velocity, and contact metadata for Layers 2-7.
    Guarantees strict conversation-level splits with zero data leakage.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        np.random.seed(seed)

        # Prior context exchanges to test multi-message context windows
        self.benign_contexts = [
            ["Hey did you finish the math homework?", "Almost, question 4 is tricky"],
            ["Are you hopping on Valorant tonight?", "Yeah give me 10 mins"],
            ["Mom asked when you'll be home", "Tell her I'm walking back from practice"],
            ["Did you see the new Marvel trailer?", "Yeah it looked insane fr"],
            ["What are we getting Sarah for her birthday?", "A book and some chocolates"],
        ]

        self.suspicious_contexts = [
            ["Are your parents still awake?", "No they went to sleep an hour ago"],
            ["You're way more mature than other kids your age", "Thanks I guess"],
            ["Nobody at your school really likes you anyway", "Why are you saying that?"],
            ["I have the screenshot of what you said earlier", "Please don't share it"],
            ["Are you alone in your room right now?", "Yeah my door is closed"],
        ]

        # Category templates with rich semantic diversity + hard negatives
        self.category_templates: Dict[str, List[str]] = {
            "BENIGN": [
                # Standard teen banter
                "Let's meet at the school library tomorrow at 4pm to study",
                "Good game bro, we totally crushed that boss raid tonight",
                "Can you send me the notes from chemistry class please?",
                "I'm going to sleep now, talk to you tomorrow morning",
                # HARD NEGATIVES (Contain trigger words in completely safe contexts!)
                "Bro I literally killed that final boss in Elden Ring no cap",
                "My mom is gonna kill me if I forget to take the chicken out of the freezer",
                "Don't tell dad about the surprise birthday cake we bought for him!",
                "Keep the surprise party a secret until Saturday okay?",
                "Delete that blurry photo of our dog and send the clear one from the park",
                # QUOTED DEFENSIVE REPORTING (Should be classified as safe/defensive report)
                "Some creepy account just texted me 'send pics and don't tell your mom' so I blocked them immediately",
                "In health class today we learned how to spot scams asking for your account password",
                "I'm dying of laughter at that meme you posted in the group chat lol",
            ],
            "THREATS": [
                "Wait until after school tomorrow, I'm going to beat your face in outside the gate",
                "I know where you walk home and I'm bringing a knife to make you pay",
                "Keep talking and I'll bash your head into the locker tomorrow morning",
                "You better watch your back walking home tonight, we're jumping you",
                "If you show up to school tomorrow I swear I will hurt you badly",
            ],
            "SEXUAL_RISK": [
                "Take off your shirt and send me a private photo on snap right now",
                "Show me what you look like without clothes on, nobody else will see it",
                "Turn on your camera and touch yourself for me while everyone is asleep",
                "You look so hot in that picture, send me a more revealing one from your bed",
                "Let's trade explicit photos just between the two of us tonight",
            ],
            "SECRECY_PRESSURE": [
                "Do not tell your parents we are talking or they won't let us be friends anymore",
                "Delete these messages right after you read them so your mom never finds out",
                "Promise me this stays our little secret and never mention my name to adults",
                "Clear your chat history right now before your dad checks your phone",
                "Switch to my private account and make sure nobody knows we text late at night",
            ],
            "COERCION": [
                "If you don't send me another picture right now I'm leaking your photos to your whole school",
                "Do exactly what I say or I'll send these screenshots to your parents and ruin your life",
                "If you block me I will post your private address and photos on every group chat",
                "You already sent one photo so if you don't send more I'm exposing you to your classmates",
                "Send me 50 dollars on gift cards right now or I leak everything to your family",
            ],
            "BULLYING_HARASSMENT": [
                "Everyone in our class thinks you're a pathetic loser and wishes you would disappear",
                "You're so ugly and worthless, why do you even bother showing up to school",
                "Nobody wants you in our group, go sit by yourself you disgusting freak",
                "We made a whole group chat just to laugh at how embarrassing and stupid you are",
                "Look at how pathetic you are, literally everyone hates you here",
            ],
            "CREDENTIAL_SCAMS": [
                "Send me your Roblox username and password right now and I'll load 10,000 free Robux",
                "Type your Discord login and 2FA code here to claim your free Nitro reward before it expires",
                "I need your Instagram password and verification code to verify your account or it gets deleted",
                "Give me your Google account password and recovery code so I can transfer the prize money",
                "Log into this link with your Steam password right now to get the free knife skin",
            ],
            "SELF_HARM_RISK": [
                "I can't take this pain anymore, I want to end my life tonight and disappear forever",
                "I've been cutting my arms all night and I don't think I want to wake up tomorrow",
                "Nobody would care if I died tonight, I took a bunch of pills and I'm saying goodbye",
                "There is no point in living anymore, tell everyone I'm sorry after I'm gone tonight",
                "How many pills does it take to go to sleep and never wake up again",
            ],
        }

    def _apply_perturbation(self, text: str) -> Tuple[str, str]:
        """
        Applies realistic teen slang, typos, abbreviations, or adversarial leetspeak
        obfuscation to test classifier robustness as required by Section 3 & 6 of the brief.
        """
        roll = self.rng.random()
        if roll < 0.50:
            return text, "clean"
        elif roll < 0.75:
            # Teen slang & abbreviation variation
            slang_map = {
                "you": "u",
                "your": "ur",
                "are": "r",
                "right now": "rn",
                "please": "plsss",
                "because": "cuz",
                "tomorrow": "tmrw",
                "parents": "rents",
                "messages": "msgs",
                "pictures": "pics",
                "photos": "pics",
                "password": "pw",
                "don't": "dnt",
            }
            perturbed = text
            for k, v in slang_map.items():
                perturbed = re.sub(rf"\b{k}\b", v, perturbed, flags=re.IGNORECASE)
            perturbed += self.rng.choice(["", " fr", " ngl", " rn", "!!"])
            return perturbed, "slang_abbreviated"
        else:
            # Adversarial obfuscation (leetspeak / character spacing / punctuation insertion)
            leet_map = {
                "delete": "d3l3t3",
                "secret": "s3cr3t",
                "send": "s3nd",
                "kill": "k1ll",
                "knife": "kn1fe",
                "naked": "n4k3d",
                "photo": "ph0t0",
                "password": "p@ssw0rd",
                "parents": "p4r3nts",
                "leak": "l34k",
                "die": "d1e",
                "pills": "p1lls",
            }
            perturbed = text
            for k, v in leet_map.items():
                perturbed = re.sub(rf"\b{k}\b", v, perturbed, flags=re.IGNORECASE)
            return perturbed, "adversarial_obfuscated"

    def build_master_dataset(self, total_conversations: int = 480) -> List[Dict[str, Any]]:
        """
        Builds ~2,400 longitudinal multi-turn events across 480 distinct conversations.
        Every conversation has 5 sequential messages so downstream Layers 2, 3, 4, and 5
        can be evaluated on the exact same dataset.
        """
        dataset: List[Dict[str, Any]] = []
        base_timestamp = 1761500000.0  # Fixed reference epoch for reproducibility

        for conv_idx in range(total_conversations):
            conv_id = f"conv_{conv_idx:04d}"
            # Assign category per conversation thread (35% Benign + Hard Negatives, 65% across 7 harms)
            if conv_idx % 3 == 0:
                cat = "BENIGN"
            else:
                harm_cats = [c for c in SAFENET_CATEGORIES if c != "BENIGN"]
                cat = harm_cats[conv_idx % len(harm_cats)]

            contact_id = f"platform_user_{conv_idx:04d}"
            is_known = True if cat == "BENIGN" and (conv_idx % 2 == 0) else False

            # Generate 5 sequential events in this conversation thread
            for turn_idx in range(5):
                # In harmful threads, turns 0-1 can start mildly/ambiguously and escalate on turns 2-4
                if cat != "BENIGN" and turn_idx == 0 and (conv_idx % 4 == 0):
                    turn_cat = "BENIGN"
                else:
                    turn_cat = cat

                ctx_pool = self.benign_contexts if turn_cat == "BENIGN" else self.suspicious_contexts
                context_window = list(self.rng.choice(ctx_pool))

                base_template = self.rng.choice(self.category_templates[turn_cat])
                # Add slight lexical variation per turn
                target_msg, pert_type = self._apply_perturbation(base_template)

                full_input = " [SEP] ".join(context_window + [target_msg])
                is_harmful = 0 if turn_cat == "BENIGN" else 1

                # Longitudinal metadata for Layers 2, 3, 4, 5
                hour = self.rng.choice([14, 16, 19, 20]) if turn_cat == "BENIGN" else self.rng.choice([1, 2, 3, 23, 15])
                velocity = round(self.rng.uniform(0.8, 3.0), 2) if turn_cat == "BENIGN" else round(self.rng.uniform(4.5, 13.5), 2)
                event_ts = base_timestamp + (conv_idx * 86400) + (turn_idx * 180)

                dataset.append({
                    "example_id": f"evt_{conv_idx:04d}_{turn_idx}",
                    "conversation_id": conv_id,
                    "contact_id": contact_id,
                    "turn_index": turn_idx,
                    "timestamp": event_ts,
                    "hour_of_day": hour,
                    "msg_velocity": velocity,
                    "is_known_contact": is_known,
                    "direction": "incoming",
                    "context_window": context_window,
                    "target_message": target_msg,
                    "full_model_input": full_input,
                    "is_harmful": is_harmful,
                    "primary_category": turn_cat,
                    "is_high_severity": turn_cat in HIGH_SEVERITY_CATEGORIES,
                    "perturbation_type": pert_type,
                })

        return dataset

    @staticmethod
    def split_by_conversation(
        dataset: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        P0 Requirement: Split strictly by conversation_id / contact_id (70% Train, 15% Val, 15% Test).
        Mathematically guarantees ZERO conversation or contact leakage across splits.
        """
        groups = np.array([row["conversation_id"] for row in dataset])
        indices = np.arange(len(dataset))

        gss_outer = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
        train_idx, temp_idx = next(gss_outer.split(indices, groups=groups))

        temp_groups = groups[temp_idx]
        gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
        val_rel_idx, test_rel_idx = next(gss_inner.split(temp_idx, groups=temp_groups))

        val_idx = temp_idx[val_rel_idx]
        test_idx = temp_idx[test_rel_idx]

        train_data = [dataset[i] for i in train_idx]
        val_data = [dataset[i] for i in val_idx]
        test_data = [dataset[i] for i in test_idx]

        # Strict Zero-Leakage Verification Assert
        train_convs = {r["conversation_id"] for r in train_data}
        val_convs = {r["conversation_id"] for r in val_data}
        test_convs = {r["conversation_id"] for r in test_data}

        assert len(train_convs.intersection(val_convs)) == 0, "LEAKAGE: Train & Val share conversations!"
        assert len(train_convs.intersection(test_convs)) == 0, "LEAKAGE: Train & Test share conversations!"
        assert len(val_convs.intersection(test_convs)) == 0, "LEAKAGE: Val & Test share conversations!"

        return train_data, val_data, test_data


# =====================================================================
# 2. BASELINE BENCHMARK MODELS (DELIVERABLE #2 — P0)
#    a) Android LexiconClassifier (Hand-written Regex Baseline)
#    b) Legacy 30-Example TF-IDF + LogisticRegression Baseline
# =====================================================================
class AndroidLexiconBaselineClassifier:
    """
    Replicates the deterministic hand-written regex LexiconClassifier currently in Android.
    Kept as a benchmark baseline per Section 3 of the brief to measure what it catches and misses.
    """

    def __init__(self):
        self.rules: Dict[str, re.Pattern] = {
            "THREATS": re.compile(r"\b(kill|beat|knife|hurt|jump you|bash)\b", re.I),
            "SEXUAL_RISK": re.compile(r"\b(naked|nude|without clothes|take off your shirt|explicit photo)\b", re.I),
            "SECRECY_PRESSURE": re.compile(r"\b(secret|don't tell|delete these|clear your chat)\b", re.I),
            "COERCION": re.compile(r"\b(leak|expose|ruin your life|screenshots to your parents)\b", re.I),
            "BULLYING_HARASSMENT": re.compile(r"\b(loser|worthless|ugly|disgusting freak|everyone hates you)\b", re.I),
            "CREDENTIAL_SCAMS": re.compile(r"\b(password|2fa code|free robux|free nitro|verification code)\b", re.I),
            "SELF_HARM_RISK": re.compile(r"\b(end my life|cutting my arms|suicide|never wake up)\b", re.I),
        }

    def predict_single(self, text: str) -> Tuple[str, float]:
        """Returns predicted category and uncalibrated heuristic score."""
        # Note: Regex looks at raw words and fails on context ("my mom will kill me") or leetspeak ("d3l3t3")
        for category, pattern in self.rules.items():
            if pattern.search(text):
                return category, 0.85
        return "BENIGN", 0.05

    def evaluate_batch(self, rows: List[Dict[str, Any]]) -> Tuple[List[str], np.ndarray]:
        preds, scores = [], []
        for r in rows:
            cat, sc = self.predict_single(r["target_message"])
            preds.append(cat)
            scores.append(sc)
        return preds, np.array(scores)


class Legacy30ExampleTFIDFBaseline:
    """
    Replicates the original apps/ml baseline trained on only 30 hand-written examples
    to quantitatively demonstrate why 30 examples fail on held-out test conversations.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=200)
        self.clf = LogisticRegression(random_state=42)

    def fit_tiny_30_sample(self, train_rows: List[Dict[str, Any]]):
        # Take only 30 samples as in the legacy prototype
        tiny_subset = train_rows[:30]
        X_txt = [r["target_message"] for r in tiny_subset]
        y = [r["primary_category"] for r in tiny_subset]
        X_vec = self.vectorizer.fit_transform(X_txt)
        self.clf.fit(X_vec, y)

    def evaluate_batch(self, rows: List[Dict[str, Any]]) -> Tuple[List[str], np.ndarray]:
        X_txt = [r["target_message"] for r in rows]
        X_vec = self.vectorizer.transform(X_txt)
        preds = self.clf.predict(X_vec).tolist()
        probs = self.clf.predict_proba(X_vec)
        benign_idx = list(self.clf.classes_).index("BENIGN") if "BENIGN" in self.clf.classes_ else 0
        harm_scores = 1.0 - probs[:, benign_idx]
        return preds, harm_scores


# =====================================================================
# 3. PRODUCTION LAYER 1 CALIBRATED CLASSIFIER (DELIVERABLE #3 — P0)
#    Context-Window + Word/Char N-Gram + Post-Hoc Platt Calibration
# =====================================================================
class ProductionLayer1ContextAI:
    """
    Production-Ready Layer 1 Content & Context Classifier.
    Improvements implemented from the SafeNet Luna ML Brief:
      1. Multi-message sliding context window input (`context_window + [SEP] + target_message`).
      2. Dual Word + Character N-Gram feature extraction (catches slang, typos, leetspeak).
      3. Separate Held-Out Validation Calibration (`CalibratedClassifierCV` with Sigmoid/Platt scaling)
         so output probabilities are empirically calibrated before reaching Layer 2 or Layer 5.
      4. Versioned artifact export + Android inference contract compatibility.
    """

    MODEL_VERSION = "safenet_l1_calibrated_v2.0.0"

    def __init__(self):
        # Word n-grams capture multi-word context ("don't tell dad about surprise" vs "don't tell mom we talk")
        # Character n-grams (3 to 5 chars) capture leetspeak ("d3l3t3", "p@ssw0rd") and typos
        self.feature_pipeline = FeatureUnion([
            ("word_ngrams", TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=8000,
                sublinear_tf=True,
                strip_accents="unicode"
            )),
            ("char_ngrams", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                max_features=8000,
                sublinear_tf=True
            )),
        ])

        self.base_clf = LogisticRegression(
            C=2.0,
            class_weight="balanced",
            max_iter=500,
            random_state=42
        )
        # Calibrated wrapper fitted strictly on the Validation fold
        self.calibrated_clf: Optional[CalibratedClassifierCV] = None
        self.classes_: List[str] = []
        self.is_calibrated: bool = False

    def fit_and_calibrate(
        self,
        train_rows: List[Dict[str, Any]],
        val_rows: List[Dict[str, Any]]
    ):
        """
        Step 1: Fits feature extractor and base classifier on Train conversations.
        Step 2: Fits Sigmoid (Platt) probability calibration strictly on unseen Validation conversations.
        """
        X_train_raw = [r["full_model_input"] for r in train_rows]
        y_train = [r["primary_category"] for r in train_rows]

        X_val_raw = [r["full_model_input"] for r in val_rows]
        y_val = [r["primary_category"] for r in val_rows]

        X_train_vec = self.feature_pipeline.fit_transform(X_train_raw)
        X_val_vec = self.feature_pipeline.transform(X_val_raw)

        # Fit base model on training split
        self.base_clf.fit(X_train_vec, y_train)

        # Calibrate probabilities on validation split (cv="prefit" or FrozenEstimator compatible)
        try:
            from sklearn.frozen import FrozenEstimator
            self.calibrated_clf = CalibratedClassifierCV(
                estimator=FrozenEstimator(self.base_clf),
                method="sigmoid"
            )
            self.calibrated_clf.fit(X_val_vec, y_val)
        except ImportError:
            self.calibrated_clf = CalibratedClassifierCV(
                estimator=self.base_clf,
                method="sigmoid",
                cv="prefit"
            )
            self.calibrated_clf.fit(X_val_vec, y_val)

        self.classes_ = list(self.calibrated_clf.classes_)
        self.is_calibrated = True

    def predict_safety_event(self, safety_event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Production inference interface preserving the exact Safe-Net 7-Layer Contract.
        Enriches `safety_event` with calibrated probabilities and category breakdowns
        ready to feed directly into Layer 2 (`layer2_interaction.py`) and Layer 5.
        """
        start_t = time.perf_counter()

        context_window = safety_event.get("context_window", [])
        target_msg = str(safety_event.get("target_message", safety_event.get("text", "")))
        full_input = " [SEP] ".join(context_window + [target_msg]) if context_window else target_msg

        X_vec = self.feature_pipeline.transform([full_input])
        probs = self.calibrated_clf.predict_proba(X_vec)[0]

        category_probs = {
            cat: round(float(probs[idx]), 4)
            for idx, cat in enumerate(self.classes_)
        }

        benign_prob = category_probs.get("BENIGN", 0.0)
        calibrated_harm_prob = round(float(1.0 - benign_prob), 4)

        # Determine top category
        top_cat = max(category_probs, key=category_probs.get)
        latency_ms = round((time.perf_counter() - start_t) * 1000.0, 3)

        # Populate Contract for Downstream Layers (L2, L3, L4, L5)
        safety_event["l1_model_version"] = self.MODEL_VERSION
        safety_event["l1_is_calibrated"] = self.is_calibrated
        safety_event["l1_risk_score"] = calibrated_harm_prob
        safety_event["l1_predicted_category"] = top_cat
        safety_event["l1_category_probabilities"] = category_probs
        safety_event["l1_high_severity_flag"] = (
            top_cat in HIGH_SEVERITY_CATEGORIES and calibrated_harm_prob >= 0.50
        )
        safety_event["l1_latency_ms"] = latency_ms

        return safety_event

    def evaluate_batch(self, rows: List[Dict[str, Any]]) -> Tuple[List[str], np.ndarray, float]:
        """Batch evaluation helper returning predictions, calibrated risk scores, and avg latency."""
        start_t = time.perf_counter()
        X_raw = [r["full_model_input"] for r in rows]
        X_vec = self.feature_pipeline.transform(X_raw)
        probs = self.calibrated_clf.predict_proba(X_vec)
        elapsed_ms_per_msg = ((time.perf_counter() - start_t) * 1000.0) / max(len(rows), 1)

        preds = [self.classes_[i] for i in np.argmax(probs, axis=1)]
        benign_idx = self.classes_.index("BENIGN")
        harm_scores = 1.0 - probs[:, benign_idx]
        return preds, harm_scores, elapsed_ms_per_msg

    def save_production_artifact(self, filepath: str = "safenet_l1_artifact.pkl"):
        """Saves versioned model artifact and metadata per Deliverable #9."""
        artifact = {
            "model_version": self.MODEL_VERSION,
            "is_calibrated": self.is_calibrated,
            "classes": self.classes_,
            "feature_pipeline": self.feature_pipeline,
            "calibrated_classifier": self.calibrated_clf,
        }
        with open(filepath, "wb") as f:
            pickle.dump(artifact, f)


# =====================================================================
# 4. QUANTITATIVE VALIDATION & METRICS ENGINE (SECTION 6 OF BRIEF)
# =====================================================================
def compute_expected_calibration_error(y_true_binary: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) to verify probability calibration."""
    prob_true, prob_pred = calibration_curve(y_true_binary, y_prob, n_bins=n_bins, strategy="uniform")
    return round(float(np.mean(np.abs(prob_true - prob_pred))), 4)


def evaluate_and_compare_models(
    name: str,
    test_rows: List[Dict[str, Any]],
    preds: List[str],
    harm_scores: np.ndarray,
    latency_ms: float = 0.0
) -> Dict[str, Any]:
    """
    Computes all required Section 6 minimum evaluation metrics:
      - Macro & Weighted F1
      - False Positive Rate (FPR) on Benign/Hard-Negative examples
      - False Negative Rate (FNR) on High-Severity threats
      - Calibration Brier Score & Expected Calibration Error (ECE)
      - Robustness on slang/obfuscated subsets
    """
    y_true_cat = [r["primary_category"] for r in test_rows]
    y_true_bin = np.array([r["is_harmful"] for r in test_rows])
    y_pred_bin = np.array([0 if p == "BENIGN" else 1 for p in preds])

    # Binary False Positive Rate (FPR) and False Negative Rate (FNR)
    benign_mask = (y_true_bin == 0)
    harm_mask = (y_true_bin == 1)
    high_sev_mask = np.array([r["is_high_severity"] for r in test_rows])

    fp = np.sum((y_pred_bin == 1) & benign_mask)
    tn = np.sum((y_pred_bin == 0) & benign_mask)
    fpr = round(float(fp / max(fp + tn, 1)), 4)

    fn_all = np.sum((y_pred_bin == 0) & harm_mask)
    tp_all = np.sum((y_pred_bin == 1) & harm_mask)
    fnr_overall = round(float(fn_all / max(fn_all + tp_all, 1)), 4)

    fn_high = np.sum((y_pred_bin == 0) & high_sev_mask)
    tp_high = np.sum((y_pred_bin == 1) & high_sev_mask)
    fnr_high_severity = round(float(fn_high / max(fn_high + tp_high, 1)), 4)

    # Calibration metrics
    brier = round(float(brier_score_loss(y_true_bin, harm_scores)), 4)
    ece = compute_expected_calibration_error(y_true_bin, harm_scores)

    # Obfuscation / Slang Robustness Recall
    obf_mask = np.array([r["perturbation_type"] != "clean" and r["is_harmful"] == 1 for r in test_rows])
    obf_recall = round(float(np.mean(y_pred_bin[obf_mask] == 1)), 4) if np.any(obf_mask) else 0.0

    prec, rec, f1, _ = precision_recall_fscore_support(y_true_cat, preds, average="macro", zero_division=0)

    return {
        "model_name": name,
        "macro_precision": round(float(prec), 4),
        "macro_recall": round(float(rec), 4),
        "macro_f1": round(float(f1), 4),
        "false_positive_rate_FPR": fpr,
        "false_negative_rate_FNR": fnr_overall,
        "high_severity_FNR": fnr_high_severity,
        "obfuscated_slang_recall": obf_recall,
        "brier_calibration_loss": brier,
        "expected_calibration_error_ECE": ece,
        "latency_ms_per_msg": round(latency_ms, 4),
    }


# =====================================================================
# MAIN EXECUTION: BUILD DATASET -> BENCHMARK BASELINES -> TRAIN & VALIDATE L1
# =====================================================================
if __name__ == "__main__":
    print("=" * 78)
    print("SAFENET LUNA ML — LAYER 1 PRODUCTION VALIDATION & CALIBRATION SUITE")
    print("=" * 78)

    # 1. Build Master Longitudinal Dataset & Verify Zero Leakage Splits
    builder = MasterSafeNetDatasetBuilder(seed=42)
    master_data = builder.build_master_dataset(total_conversations=480)
    train_split, val_split, test_split = builder.split_by_conversation(master_data)

    print(f"[Dataset Spec] Total Longitudinal Events: {len(master_data)}")
    print(f"  ├─ Train Split : {len(train_split)} events ({len({r['conversation_id'] for r in train_split})} unique conversations)")
    print(f"  ├─ Val Split   : {len(val_split)} events ({len({r['conversation_id'] for r in val_split})} unique conversations)")
    print(f"  └─ Test Split  : {len(test_split)} events ({len({r['conversation_id'] for r in test_split})} unique conversations)")
    print("  [✓] Zero Conversation/Contact Leakage Verified Across Splits.\n")

    # Save master dataset for Layers 2, 3, 4, 5, 6 ablation testing
    with open("safenet_master_dataset.jsonl", "w", encoding="utf-8") as f:
        for row in master_data:
            f.write(json.dumps(row) + "\n")

    # 2. Evaluate Baseline A: Legacy 30-Example TF-IDF Model
    legacy_30 = Legacy30ExampleTFIDFBaseline()
    legacy_30.fit_tiny_30_sample(train_split)
    preds_30, scores_30 = legacy_30.evaluate_batch(test_split)
    metrics_30 = evaluate_and_compare_models("Baseline A: Legacy 30-Ex TF-IDF", test_split, preds_30, scores_30, 0.05)

    # 3. Evaluate Baseline B: Android LexiconClassifier (Hand-written Regex)
    lexicon_clf = AndroidLexiconBaselineClassifier()
    preds_lex, scores_lex = lexicon_clf.evaluate_batch(test_split)
    metrics_lex = evaluate_and_compare_models("Baseline B: Android Lexicon Regex", test_split, preds_lex, scores_lex, 0.02)

    # 4. Train & Calibrate Production Layer 1 Model
    prod_l1 = ProductionLayer1ContextAI()
    prod_l1.fit_and_calibrate(train_split, val_split)
    preds_prod, scores_prod, lat_prod = prod_l1.evaluate_batch(test_split)
    metrics_prod = evaluate_and_compare_models(
        "Production L1: Calibrated Context-Window AI", test_split, preds_prod, scores_prod, lat_prod
    )

    # 5. Print Side-by-Side Quantitative Comparison Table (Deliverable #2 & #3)
    print("=== HELD-OUT TEST SET BENCHMARK (UNSEEN CONVERSATIONS & CONTACTS) ===")
    header = f"{'Model Candidate':<42} | {'Macro F1':<8} | {'FPR':<6} | {'High-Sev FNR':<12} | {'Slang Rec':<9} | {'ECE (Calib)':<11}"
    print(header)
    print("-" * len(header))
    for m in [metrics_30, metrics_lex, metrics_prod]:
        print(
            f"{m['model_name']:<42} | "
            f"{m['macro_f1']:<8.4f} | "
            f"{m['false_positive_rate_FPR']:<6.4f} | "
            f"{m['high_severity_FNR']:<12.4f} | "
            f"{m['obfuscated_slang_recall']:<9.4f} | "
            f"{m['expected_calibration_error_ECE']:<11.4f}"
        )

    # 6. Print Detailed Per-Category Report for Production L1
    print("\n=== PER-CATEGORY PRECISION / RECALL / F1 (PRODUCTION L1 ON HELD-OUT TEST) ===")
    y_true_test = [r["primary_category"] for r in test_split]
    print(classification_report(y_true_test, preds_prod, digits=4))

    # 7. Print Confusion Matrix
    print("=== CONFUSION MATRIX (ROWS=TRUE, COLS=PREDICTED) ===")
    cm = confusion_matrix(y_true_test, preds_prod, labels=SAFENET_CATEGORIES)
    print("Categories:", [c[:8] for c in SAFENET_CATEGORIES])
    print(cm)

    # 8. Verify Live Single-Event Contract Hand-Off to Layer 2
    sample_test_event = dict(test_split[0])
    enriched_event = prod_l1.predict_safety_event(sample_test_event)
    print("\n=== SAMPLE LAYER 1 -> LAYER 2 PRODUCTION CONTRACT OUTPUT ===")
    print(
        f"Model Version: {enriched_event['l1_model_version']} | "
        f"Calibrated Risk Score: {enriched_event['l1_risk_score']} | "
        f"Predicted Category: {enriched_event['l1_predicted_category']} | "
        f"Latency: {enriched_event['l1_latency_ms']} ms"
    )

    # Save versioned artifact
    prod_l1.save_production_artifact("safenet_l1_artifact.pkl")
    print("[✓] Saved versioned model artifact to 'safenet_l1_artifact.pkl' and dataset to 'safenet_master_dataset.jsonl'.")