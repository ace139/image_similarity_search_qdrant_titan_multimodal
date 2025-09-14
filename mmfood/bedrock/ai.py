from __future__ import annotations

import os
import json
import base64
from typing import Optional, Tuple


# Default inference profile ID for Claude Vision (Converse API requires a profile for Anthropic models)
DEFAULT_CLAUDE_VISION_PROFILE = os.getenv(
    "CLAUDE_VISION_MODEL_ID", "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
)


def _is_inference_profile_identifier(s: str) -> bool:
    """Return True if the string looks like a Bedrock inference profile ID or ARN.

    Accepts IDs beginning with 'us.' (cross-region profiles) or 'global.' as seen in some
    Bedrock console listings, and full ARNs containing ':inference-profile/'.
    """
    if not isinstance(s, str):
        return False
    return (
        s.startswith("us.")
        or s.startswith("global.")
        or (s.startswith("arn:aws:bedrock:") and ":inference-profile/" in s)
    )


def _resolve_converse_model_id(raw_id: Optional[str]) -> str:
    """Normalize model identifier for Converse to use an inference profile ID/ARN."""
    model = (raw_id or DEFAULT_CLAUDE_VISION_PROFILE or "").strip()
    if not model:
        raise ValueError(
            "CLAUDE_VISION_MODEL_ID is not set. Set it to an inference profile ID (e.g., "
            "'us.anthropic.claude-3-5-sonnet-20241022-v2:0') or its ARN."
        )
    if _is_inference_profile_identifier(model):
        return model

    # Known mappings: foundation model ID -> inference profile ID
    known_map = {
        "anthropic.claude-3-5-sonnet-20241022-v2:0": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "anthropic.claude-3-haiku-20240307-v1:0": "us.anthropic.claude-3-haiku-20240307-v1:0",
        "anthropic.claude-3-7-sonnet-20250219-v1:0": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    }
    if model in known_map:
        return known_map[model]

    raise ValueError(
        (
            "Bedrock Converse requires an inference profile ID/ARN for this model. "
            f"Got foundation model ID '{model}'. Set CLAUDE_VISION_MODEL_ID to an inference profile (e.g., "
            "'us.anthropic.claude-3-5-sonnet-20241022-v2:0') or the corresponding profile ARN from the Bedrock console."
        )
    )


def _titan_mm_safe_text(text: str, max_words: int = 80) -> str:
    """Clean and truncate text for Titan MM input."""
    if not text:
        return ""
    words = text.replace("\n", " ").split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return " ".join(words)


def _safe_json_loads(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _build_embed_text_from_struct(d: dict) -> str:
    # Build a compact, search-friendly string for Titan MM (<= ~60-80 words)
    items = ", ".join((d.get("items") or [])[:8])
    prep = ", ".join((d.get("preparation") or [])[:5])
    tags = ", ".join((d.get("dietary_tags") or [])[:10])
    health = (d.get("perceived_healthiness") or "").strip()
    parts = []
    if items:
        parts.append(f"items: {items}")
    if prep:
        parts.append(f"prep: {prep}")
    if health:
        parts.append(f"health: {health}")
    if tags:
        parts.append(f"tags: {tags}")
    compact = "; ".join(parts)
    return _titan_mm_safe_text(compact, max_words=60) or compact


def generate_mm_embedding(
    *,
    bedrock_client,
    model_id: str,
    output_dim: int,
    input_text: Optional[str] = None,
    input_image_bytes: Optional[bytes] = None,
) -> list:
    """Generate an embedding using Titan Multimodal for text and/or image.

    - Caps input text to stay within Titan's ~128-token budget.
    - Treats Titan body `message` as a warning if an embedding is present.
    - Retries once with a tighter cap if Titan refuses due to token limits.
    """
    if not input_text and not input_image_bytes:
        raise ValueError("Either input_text or input_image_bytes must be provided")

    # Keep text within Titan's limit (conservative caps)
    safe_text = None
    if input_text:
        safe_text = _titan_mm_safe_text(input_text, max_words=80)

    def _invoke(text_for_body: Optional[str]):
        body_dict = {"embeddingConfig": {"outputEmbeddingLength": output_dim}}
        if text_for_body:
            body_dict["inputText"] = text_for_body
        if input_image_bytes:
            body_dict["inputImage"] = base64.b64encode(input_image_bytes).decode("utf-8")
        resp = bedrock_client.invoke_model(
            body=json.dumps(body_dict),
            modelId=model_id,
            accept="application/json",
            contentType="application/json",
        )
        body = json.loads(resp.get("body").read())
        return body

    body = _invoke(safe_text)
    warning = (body or {}).get("message")
    embedding = (body or {}).get("embedding")
    if isinstance(embedding, list):
        if warning:
            print(f"[Titan warning:mm] {warning}")
        return embedding

    # Retry once if token/limit related and we had text
    if warning and ("token" in warning.lower() or "limit" in warning.lower()) and safe_text:
        tighter = _titan_mm_safe_text(safe_text, max_words=60)
        body = _invoke(tighter)
        embedding = (body or {}).get("embedding")
        if isinstance(embedding, list):
            return embedding
        warning = (body or {}).get("message") or warning

    if warning:
        raise RuntimeError(f"Model error: {warning}")
    raise RuntimeError("Embedding not found in model response.")


def generate_image_description(
    image_bytes: bytes,
    meal_data: dict,
    bedrock_client,
    claude_model_id: str = None,
) -> Tuple[str, str]:
    """
    Generate a detailed description of the food image using Claude Vision.
    Combines the meal metadata to create rich, searchable descriptions.
    Returns a tuple: (display_text, embed_text)
    """
    claude_model_id = _resolve_converse_model_id(claude_model_id)
    try:
        prompt_text = (
            "You are a nutrition-conscious vision assistant. Analyze the food image. "
            "Return ONLY compact JSON with these keys: "
            "items (array), preparation (array), sauces_or_sides (array), "
            "perceived_healthiness (one of: 'healthy','moderate','indulgent'), "
            "health_reasons (string, <=25 words), dietary_tags (array; choose from: "
            "['high_protein','whole_grain','fresh_fruit','vegetables','nuts_seeds','dairy',"
            "'fried','refined_carbs','processed_meat','added_syrup','high_sugar','high_salt']), "
            "portion ('small'|'medium'|'large'). No prose, no markdown."
            f"\n\nContext: meal_type={meal_data.get('meal_type','meal')}"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"text": prompt_text},
                    {"image": {"format": "jpeg", "source": {"bytes": image_bytes}}},
                ],
            }
        ]

        response = bedrock_client.converse(
            modelId=claude_model_id,
            messages=messages,
            inferenceConfig={"maxTokens": 160, "temperature": 0.1},
        )

        raw = response["output"]["message"]["content"][0]["text"].strip()
        data = _safe_json_loads(raw)
        if isinstance(data, dict):
            items = ", ".join((data.get("items") or [])[:6])
            health = data.get("perceived_healthiness") or "unknown"
            reason = (data.get("health_reasons") or "").strip()
            display = f"Items: {items}. Health: {health}. {reason}".strip()
            embed_text = _build_embed_text_from_struct(data)
            return display, embed_text
        return raw, _titan_mm_safe_text(raw, max_words=80)
    except Exception:
        # Surface the error to the caller so Streamlit shows it instead of silently falling back.
        raise
