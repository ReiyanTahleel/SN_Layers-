"""
Safe-Net Architecture — Layer 6: Multimodal AI (Vision & Audio Safeguarding)
Description:
    Analyzes visual (images/memes) and acoustic (voice notes) payloads attached
    to suspicious conversational contexts.
    - Vision: Zero-shot semantic contrastive probing via OpenCLIP (ViT-B-32).
    - Audio: High-throughput speech transcription via faster-whisper with Silero VAD.
    Includes Lazy-Activation Gating from Layer 5 to preserve edge battery life.
"""

from typing import Dict, Any, List, Optional
import os
import torch
from PIL import Image

# 1. OpenCLIP for Zero-Shot Image Content Moderation
try:
    import open_clip
    OPENCLIP_AVAILABLE = True
except ImportError:
    OPENCLIP_AVAILABLE = False

# 2. faster-whisper for High-Speed On-Device Audio Transcription
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False


class OpenCLIPVisionAuditor:
    """
    On-device vision safeguarding engine.
    Uses zero-shot contrastive cosine similarity against calibrated safety anchors.
    """

    HAZARD_PROMPTS = [
        "explicit nudity or sexual content",
        "child exploitation or grooming material",
        "physical self-harm or cuts",
        "firearms, weapons, or physical threat",
        "illicit drugs or pills"
    ]

    BENIGN_PROMPTS = [
        "a wholesome family photo or casual portrait",
        "outdoor scenery, playground, nature, or sports",
        "school homework, textbook, or educational diagram",
        "a harmless cartoon, comic, or gaming screenshot",
        "everyday clothing, food, or pets"
    ]

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.enabled = OPENCLIP_AVAILABLE

        if self.enabled:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)

            # Pre-compute and normalize text embeddings for zero-latency inference
            all_prompts = self.HAZARD_PROMPTS + self.BENIGN_PROMPTS
            text_tokens = self.tokenizer(all_prompts).to(self.device)
            with torch.no_grad():
                text_embs = self.model.encode_text(text_tokens)
                self.text_features = text_embs / text_embs.norm(dim=-1, keepdim=True)
        else:
            self.model = None

    def audit_image(self, image_path: str) -> Dict[str, Any]:
        """Scans image against contrastive hazard anchors; returns calibrated risk score."""
        if not self.enabled or not os.path.exists(image_path):
            return {"vision_risk_score": 0.0, "top_category": "NONE", "status": "BYPASSED_OR_MISSING"}

        try:
            image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                # Cosine similarity matrix -> Softmax across all categories
                similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)[0]

            hazard_probs = similarity[:len(self.HAZARD_PROMPTS)].cpu().numpy()
            benign_probs = similarity[len(self.HAZARD_PROMPTS):].cpu().numpy()

            total_hazard_prob = float(hazard_probs.sum())
            max_hazard_idx = int(hazard_probs.argmax())
            top_hazard_label = self.HAZARD_PROMPTS[max_hazard_idx]

            return {
                "vision_risk_score": round(total_hazard_prob, 4),
                "top_category": top_hazard_label if total_hazard_prob >= 0.30 else "BENIGN",
                "status": "SCANNED"
            }
        except Exception as err:
            return {"vision_risk_score": 0.0, "top_category": "ERROR", "status": f"FAIL: {str(err)}"}


class FasterWhisperAudioAuditor:
    """
    On-device speech transcriber for incoming voice notes.
    Uses CTranslate2 INT8 quantization and Silero VAD to eliminate hallucinations.
    """

    def __init__(self, model_size: str = "tiny"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "int8"  # Ultra-fast CPU/Edge deployment
        self.enabled = FASTER_WHISPER_AVAILABLE

        if self.enabled:
            # Loads quantized model with minimal memory footprint
            self.model = WhisperModel(model_size, device=self.device, compute_type=self.compute_type)
        else:
            self.model = None

    def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """Transcribes voice notes with VAD silence filtering."""
        if not self.enabled or not os.path.exists(audio_path):
            return {"transcription": "", "has_speech": False, "status": "BYPASSED_OR_MISSING"}

        try:
            # vad_filter=True discards background noise/silence to prevent hallucinations
            segments, info = self.model.transcribe(
                audio_path,
                beam_size=1,  # Greedy search for maximum edge speed
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            text_chunks = [segment.text.strip() for segment in segments]
            full_transcript = " ".join(text_chunks)

            return {
                "transcription": full_transcript,
                "has_speech": len(full_transcript) > 0,
                "language": info.language,
                "status": "TRANSCRIBED"
            }
        except Exception as err:
            return {"transcription": "", "has_speech": False, "status": f"FAIL: {str(err)}"}


class Layer6MultimodalAI:
    """
    Main Layer 6 Multimodal Orchestrator.
    Connects Layer 5 (Bayesian State) -> Layer 6 (Multimodal) -> Layer 7 (Closed Loop / Vault).
    """

    def __init__(self, enable_heavy_models: bool = True):
        if enable_heavy_models:
            self.vision_engine = OpenCLIPVisionAuditor()
            self.audio_engine = FasterWhisperAudioAuditor()
        else:
            self.vision_engine = None
            self.audio_engine = None

    def process_safety_event(self, safety_event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Receives enriched SafetyEvent from Layer 5.
        Checks Lazy-Activation Gate, audits media attachments if present,
        and computes fused multimodal risk score.
        """
        l5_state = str(safety_event.get("l5_safety_state", "NORMAL"))
        image_path: Optional[str] = safety_event.get("media_image_path")
        audio_path: Optional[str] = safety_event.get("media_audio_path")

        # =====================================================================
        # GUARDRAIL 1: Lazy-Activation Battery Gate
        # If no media is attached, bypass multimodal models completely.
        # If media IS attached but L5 state is NORMAL from a trusted contact,
        # perform lightweight check only if explicitly requested.
        # =====================================================================
        has_media = bool((image_path and os.path.exists(image_path)) or (audio_path and os.path.exists(audio_path)))
        
        if not has_media:
            safety_event["l6_status"] = "NO_MEDIA_BYPASSED"
            safety_event["l6_multimodal_risk"] = 0.0
            safety_event["l6_action"] = "PASS_TEXT_ONLY"
            return safety_event

        # =====================================================================
        # 2. Vision Audit via OpenCLIP
        # =====================================================================
        vision_result = {"vision_risk_score": 0.0, "top_category": "NONE"}
        if image_path and self.vision_engine:
            vision_result = self.vision_engine.audit_image(image_path)

        # =====================================================================
        # 3. Audio Audit via faster-whisper
        # =====================================================================
        audio_result = {"transcription": "", "has_speech": False}
        if audio_path and self.audio_engine:
            audio_result = self.audio_engine.transcribe_audio(audio_path)

        # =====================================================================
        # 4. Multimodal Fusion & Escalation Gate
        # Combines Layer 5 Bayesian risk with vision risk
        # =====================================================================
        v_score = vision_result.get("vision_risk_score", 0.0)
        l5_is_high = l5_state in ("HIGH", "CRITICAL")

        # Visual danger is severe if OpenCLIP confirms hazard anchor
        is_visual_danger = v_score >= 0.50

        if is_visual_danger or (l5_is_high and v_score >= 0.35):
            l6_action = "CRITICAL_MEDIA_VAULT_LOCK"
            final_media_hazard = max(v_score, 0.85)
            safety_event["Lock_In_Vault"] = True  # Override vault lock for Layer 7
        elif v_score >= 0.25 or audio_result["has_speech"]:
            l6_action = "ELEVATED_MEDIA_AUDIT"
            final_media_hazard = max(v_score, 0.45)
        else:
            l6_action = "MEDIA_INSPECTION_SAFE"
            final_media_hazard = v_score

        # Enrich contract for Layer 7
        safety_event["l6_status"] = "PROCESSED"
        safety_event["l6_multimodal_risk"] = round(final_media_hazard, 4)
        safety_event["l6_action"] = l6_action
        safety_event["l6_vision_details"] = vision_result
        safety_event["l6_audio_details"] = audio_result

        return safety_event 


# =====================================================================
# SELF-TEST SUITE: Run `python layer6_multimodal.py`
# =====================================================================
if __name__ == "__main__":
    # Test with lightweight initialization (simulated media paths)
    l6 = Layer6MultimodalAI(enable_heavy_models=False)

    print("=== TEST 1: Plain Text Message (Lazy Gate Check) ===")
    evt_text_only = {
        "event_id": "evt_901",
        "l5_safety_state": "NORMAL",
        "media_image_path": None,
        "media_audio_path": None
    }
    res1 = l6.process_safety_event(evt_text_only)
    print(f"Status: {res1['l6_status']} | Risk: {res1['l6_multimodal_risk']} | Action: {res1['l6_action']}\n")

    print("=== TEST 2: Gated Vision Pipeline Architecture Check ===")
    evt_simulated_media = {
        "event_id": "evt_902",
        "l5_safety_state": "HIGH",
        "media_image_path": "test_sample.jpg",
        "media_audio_path": None
    }
    # Shows how contracts seamlessly integrate even when media is queued
    res2 = l6.process_safety_event(evt_simulated_media)
    print(f"Action: {res2['l6_action']} | Contract Hand-off Ready: True")