"""
Configuration management for KORA
"""
import json
from pathlib import Path
from typing import Optional, Dict


CONFIG_DIR = Path(".kora")
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULT_CONFIG = {
    "default_model": "granite3.3:2b",
    "default_temperature": 0.7,
    "default_top_k": 8,
    "system_prompt": "default"
}

# Predefined system prompts
SYSTEM_PROMPTS = {
    "default": (
        "You are KORA - the Knowledge Oriented Retrieval Assistant. You are a helpful assistant created by researchers at Georgia Tech to help students with course content. "
        "Use ONLY the provided context to answer. If the answer is not in the context, say you don't know. Be concise. "
        "When using mathematical notation, always use LaTeX format with proper delimiters: use $...$ for inline math and $$...$$ for display math. "
        "For example: 'The equation is $E = mc^2$' or for display: '$$\\int_0^\\infty e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}$$'"
    ),
    "concise": (
        "You are KORA, a concise knowledge assistant. Answer questions using ONLY the provided context. "
        "Keep answers brief and to the point. If the answer is not in the context, say 'I don't know.' "
        "Use LaTeX for math: $...$ for inline, $$...$$ for display."
    ),
    "detailed": (
        "You are KORA - the Knowledge Oriented Retrieval Assistant. You are a thorough and detailed assistant helping students understand course content. "
        "Use ONLY the provided context to answer. Provide comprehensive explanations when appropriate. "
        "If the answer is not in the context, clearly state that you don't have that information. "
        "When using mathematical notation, always use LaTeX format with proper delimiters: use $...$ for inline math and $$...$$ for display math. "
        "Break down complex concepts step by step."
    ),
    "socratic": (
        "You are KORA, a Socratic teaching assistant. When answering questions, guide students to understanding through thoughtful questions and hints. "
        "Use ONLY the provided context. If appropriate, ask follow-up questions to deepen understanding. "
        "If the answer is not in the context, say you don't know. Use LaTeX for mathematical notation: $...$ for inline, $$...$$ for display."
    ),
    "technical": (
        "You are KORA, a technical knowledge assistant. Provide precise, technical answers using ONLY the provided context. "
        "Include relevant terminology, definitions, and technical details. Be accurate and specific. "
        "If the answer is not in the context, state 'Information not available in context.' "
        "Use LaTeX format for all mathematical expressions: $...$ for inline math and $$...$$ for display math."
    ),
    "simple": (
        "You are KORA, a friendly learning assistant. Explain concepts in simple, easy-to-understand language using ONLY the provided context. "
        "Avoid jargon when possible. Use analogies and examples to clarify. "
        "If the answer is not in the context, say you don't know. Use LaTeX for math when needed: $...$ for inline, $$...$$ for display."
    )
}


def ensure_config_dir():
    """Ensure config directory exists."""
    CONFIG_DIR.mkdir(exist_ok=True)


def load_config() -> dict:
    """Load configuration from file or return defaults."""
    ensure_config_dir()
    
    if not CONFIG_FILE.exists():
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()
    
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
        
        # Merge with defaults in case new keys were added
        merged = DEFAULT_CONFIG.copy()
        merged.update(config)
        return merged
    except Exception:
        return DEFAULT_CONFIG.copy()


def save_config(config: dict) -> None:
    """Save configuration to file."""
    ensure_config_dir()
    
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def get_default_model() -> str:
    """Get the default model from config."""
    config = load_config()
    return config.get("default_model", DEFAULT_CONFIG["default_model"])


def set_default_model(model: str) -> None:
    """Set the default model in config."""
    config = load_config()
    config["default_model"] = model
    save_config(config)


def get_default_temperature() -> float:
    """Get the default temperature from config."""
    config = load_config()
    return config.get("default_temperature", DEFAULT_CONFIG["default_temperature"])


def get_default_top_k() -> int:
    """Get the default top_k from config."""
    config = load_config()
    return config.get("default_top_k", DEFAULT_CONFIG["default_top_k"])


def get_system_prompt_name() -> str:
    """Get the current system prompt name from config."""
    config = load_config()
    return config.get("system_prompt", DEFAULT_CONFIG["system_prompt"])


def set_system_prompt(prompt_name: str) -> None:
    """Set the system prompt in config."""
    if prompt_name not in SYSTEM_PROMPTS:
        raise ValueError(f"Unknown prompt name: {prompt_name}. Available: {list(SYSTEM_PROMPTS.keys())}")
    
    config = load_config()
    config["system_prompt"] = prompt_name
    save_config(config)


def get_system_prompt_text(prompt_name: Optional[str] = None) -> str:
    """
    Get the system prompt text.
    
    Args:
        prompt_name: Name of the prompt to retrieve. If None, uses the configured default.
    
    Returns:
        The system prompt text.
    """
    if prompt_name is None:
        prompt_name = get_system_prompt_name()
    
    return SYSTEM_PROMPTS.get(prompt_name, SYSTEM_PROMPTS["default"])


def get_available_prompts() -> Dict[str, str]:
    """Get all available system prompts with descriptions."""
    descriptions = {
        "default": "Default KORA assistant - balanced, helpful, context-focused",
        "concise": "Brief and to-the-point responses",
        "detailed": "Comprehensive explanations with step-by-step breakdowns",
        "socratic": "Guides learning through questions and hints",
        "technical": "Precise, technical answers with terminology",
        "simple": "Easy-to-understand explanations for beginners"
    }
    return descriptions
