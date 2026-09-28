"""
services/translation_service.py

TranslationService is deliberately modular: the underlying translation
backend is isolated inside _translate_backend() so it can be swapped
(e.g. deep-translator → LibreTranslate → an LLM endpoint) without
touching any other file.

Current backend: deep-translator (Helsinki-NLP / Google fallback).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class TranslationService:
    """
    Translates a piece of text from source_language to target_language.

    Usage
    -----
    svc = TranslationService()
    result = svc.translate("Open", source_language="English", target_language="French")
    # → "Ouvrir"
    """

    def __init__(self):
        # Optionally pre-load resources here (e.g. a local model)
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────
    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        """
        Translate *text* from *source_language* to *target_language*.

        Args:
            text:            The text string to translate.
            source_language: Human-readable language name, e.g. "English".
            target_language: Human-readable language name, e.g. "French".

        Returns:
            Translated string, or the original text if translation fails.
        """
        if not text.strip():
            return text

        # Skip translation when source == target
        if source_language.strip().lower() == target_language.strip().lower():
            return text

        t0 = time.perf_counter()
        try:
            result = self._translate_backend(text, source_language, target_language)
            logger.debug(
                "Translation | '%.40s' → '%.40s' (%.2fs)",
                text,
                result,
                time.perf_counter() - t0,
            )
            return result
        except Exception as exc:
            logger.error(
                "Translation | failed for text='%.40s': %s — returning original",
                text,
                exc,
            )
            return text  # graceful degradation

    # ─────────────────────────────────────────────────────────────────────────
    # Backend (swap this method to change the translation engine)
    # ─────────────────────────────────────────────────────────────────────────
    def _translate_backend(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        """
        Current backend: deep-translator with Google Translate.

        To replace: implement the same signature and return the translated str.
        """
        from deep_translator import GoogleTranslator  # type: ignore

        src_code = _language_to_code(source_language)
        tgt_code = _language_to_code(target_language)

        translator = GoogleTranslator(source=src_code, target=tgt_code)
        return translator.translate(text)

    # ─────────────────────────────────────────────────────────────────────────
    # Batch helper (optional, used by orchestrator for efficiency)
    # ─────────────────────────────────────────────────────────────────────────
    def translate_batch(
        self,
        texts: list[str],
        source_language: str,
        target_language: str,
    ) -> list[str]:
        """Translate a list of texts, returning results in the same order."""
        return [
            self.translate(t, source_language, target_language) for t in texts
        ]


# ──────────────────────────────────────────────────────────────────────────────
# Language-name → ISO 639-1 code helper
# ──────────────────────────────────────────────────────────────────────────────
_LANGUAGE_MAP: dict[str, str] = {
    # Common languages — extend as needed
    "afrikaans": "af",
    "albanian": "sq",
    "amharic": "am",
    "arabic": "ar",
    "azerbaijani": "az",
    "basque": "eu",
    "belarusian": "be",
    "bengali": "bn",
    "bosnian": "bs",
    "bulgarian": "bg",
    "catalan": "ca",
    "cebuano": "ceb",
    "chinese": "zh-CN",
    "chinese simplified": "zh-CN",
    "chinese traditional": "zh-TW",
    "mandarin": "zh-CN",
    "corsican": "co",
    "croatian": "hr",
    "czech": "cs",
    "danish": "da",
    "dutch": "nl",
    "english": "en",
    "esperanto": "eo",
    "estonian": "et",
    "filipino": "tl",
    "tagalog": "tl",
    "finnish": "fi",
    "french": "fr",
    "frisian": "fy",
    "galician": "gl",
    "georgian": "ka",
    "german": "de",
    "greek": "el",
    "gujarati": "gu",
    "haitian creole": "ht",
    "hausa": "ha",
    "hawaiian": "haw",
    "hebrew": "iw",
    "hindi": "hi",
    "hmong": "hmn",
    "hungarian": "hu",
    "icelandic": "is",
    "igbo": "ig",
    "indonesian": "id",
    "irish": "ga",
    "italian": "it",
    "japanese": "ja",
    "javanese": "jw",
    "kannada": "kn",
    "kazakh": "kk",
    "khmer": "km",
    "korean": "ko",
    "kurdish": "ku",
    "kyrgyz": "ky",
    "lao": "lo",
    "latin": "la",
    "latvian": "lv",
    "lithuanian": "lt",
    "luxembourgish": "lb",
    "macedonian": "mk",
    "malagasy": "mg",
    "malay": "ms",
    "malayalam": "ml",
    "maltese": "mt",
    "maori": "mi",
    "marathi": "mr",
    "mongolian": "mn",
    "myanmar": "my",
    "burmese": "my",
    "nepali": "ne",
    "norwegian": "no",
    "nyanja": "ny",
    "chichewa": "ny",
    "pashto": "ps",
    "persian": "fa",
    "farsi": "fa",
    "polish": "pl",
    "portuguese": "pt",
    "punjabi": "pa",
    "romanian": "ro",
    "russian": "ru",
    "samoan": "sm",
    "scots gaelic": "gd",
    "serbian": "sr",
    "sesotho": "st",
    "shona": "sn",
    "sindhi": "sd",
    "sinhala": "si",
    "slovak": "sk",
    "slovenian": "sl",
    "somali": "so",
    "spanish": "es",
    "sundanese": "su",
    "swahili": "sw",
    "swedish": "sv",
    "tajik": "tg",
    "tamil": "ta",
    "telugu": "te",
    "thai": "th",
    "turkish": "tr",
    "ukrainian": "uk",
    "urdu": "ur",
    "uzbek": "uz",
    "vietnamese": "vi",
    "welsh": "cy",
    "xhosa": "xh",
    "yiddish": "yi",
    "yoruba": "yo",
    "zulu": "zu",
}


def _language_to_code(language_name: str) -> str:
    """
    Convert a human-readable language name to its ISO 639-1 code.
    Falls back to the input as-is (deep-translator accepts codes directly).
    """
    code = _LANGUAGE_MAP.get(language_name.strip().lower())
    if code:
        return code
    # If it's already a short code, pass through
    if len(language_name.strip()) <= 5:
        return language_name.strip().lower()
    # Last resort: pass the full name and let deep-translator handle it
    logger.warning(
        "Translation | unknown language '%s', passing raw to backend", language_name
    )
    return language_name.strip().lower()
