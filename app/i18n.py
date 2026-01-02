"""
Internationalization (i18n) module for multi-language support.
Supports: Indonesian (id), English (en), Arabic (ar)

Translations are loaded from JSON files in the i18n folder:
  app/i18n/{lang}/{module}.json

Example:
  app/i18n/id/main.json
  app/i18n/en/rag.json
  app/i18n/ar/tradingview.json
"""

import json
from pathlib import Path
from typing import Dict, Any
from contextvars import ContextVar

from fastapi import Request

# Default language
DEFAULT_LANG = "id"

# Supported languages
SUPPORTED_LANGS = ["id", "en", "ar"]

# i18n directory path
I18N_DIR = Path(__file__).resolve().parent / "i18n"

# Cache for loaded translations: {lang: {module: {key: value}}}
_translations_cache: Dict[str, Dict[str, Dict[str, str]]] = {}


def _load_translations(lang: str, module: str) -> Dict[str, str]:
    """
    Load translations from JSON file for a specific language and module.
    Returns empty dict if file not found.
    """
    cache_key = f"{lang}:{module}"
    
    # Check cache first
    if lang in _translations_cache and module in _translations_cache[lang]:
        return _translations_cache[lang][module]
    
    # Initialize cache structure
    if lang not in _translations_cache:
        _translations_cache[lang] = {}
    
    # Load from file
    file_path = I18N_DIR / lang / f"{module}.json"
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                translations = json.load(f)
                _translations_cache[lang][module] = translations
                return translations
        except (json.JSONDecodeError, IOError):
            _translations_cache[lang][module] = {}
            return {}
    
    _translations_cache[lang][module] = {}
    return {}


def _get_all_translations(lang: str) -> Dict[str, str]:
    """
    Load and merge all translations for a language from all module files.
    """
    all_translations = {}
    
    lang_dir = I18N_DIR / lang
    if lang_dir.exists():
        for json_file in lang_dir.glob("*.json"):
            module = json_file.stem
            translations = _load_translations(lang, module)
            all_translations.update(translations)
    
    return all_translations


def get_lang_from_request(request: Request) -> str:
    """
    Extract language code from Accept-Language header.
    Returns the first supported language found, or default language.
    """
    accept_language = request.headers.get("accept-language", "")
    
    if not accept_language:
        return DEFAULT_LANG
    
    # Parse Accept-Language header (e.g., "en-US,en;q=0.9,id;q=0.8")
    languages = []
    for part in accept_language.split(","):
        part = part.strip()
        if ";" in part:
            lang = part.split(";")[0].strip()
        else:
            lang = part
        
        # Get base language code (e.g., "en-US" -> "en")
        base_lang = lang.split("-")[0].lower()
        languages.append(base_lang)
    
    # Return first supported language
    for lang in languages:
        if lang in SUPPORTED_LANGS:
            return lang
    
    return DEFAULT_LANG


def t(key: str, lang: str = DEFAULT_LANG, module: str = None, **kwargs) -> str:
    """
    Get translation for a key in the specified language.
    
    Args:
        key: Translation key
        lang: Language code (id, en, ar)
        module: Optional module name to search in specific file first
        **kwargs: Format parameters for string interpolation
    
    Returns:
        Translated string or the key if not found
    """
    translation = None
    
    # If module specified, search there first
    if module:
        translations = _load_translations(lang, module)
        translation = translations.get(key)
        
        # Fallback to default language if not found
        if translation is None and lang != DEFAULT_LANG:
            translations = _load_translations(DEFAULT_LANG, module)
            translation = translations.get(key)
    
    # If not found in specific module, search all modules
    if translation is None:
        all_translations = _get_all_translations(lang)
        translation = all_translations.get(key)
        
        # Fallback to default language
        if translation is None and lang != DEFAULT_LANG:
            all_translations = _get_all_translations(DEFAULT_LANG)
            translation = all_translations.get(key)
    
    # Return key if still not found
    if translation is None:
        return key
    
    # Apply string formatting if kwargs provided
    if kwargs:
        try:
            return translation.format(**kwargs)
        except KeyError:
            return translation
    
    return translation


def reload_translations():
    """
    Clear the translation cache to force reload from files.
    Useful during development or when translations are updated.
    """
    global _translations_cache
    _translations_cache = {}


# Context variable to store current language (for use in async contexts)
current_lang: ContextVar[str] = ContextVar("current_lang", default=DEFAULT_LANG)


def set_current_lang(lang: str):
    """Set the current language in context."""
    current_lang.set(lang)


def get_current_lang() -> str:
    """Get the current language from context."""
    return current_lang.get()


def tr(key: str, module: str = None, **kwargs) -> str:
    """
    Get translation using the current context language.
    Shorthand for t(key, get_current_lang(), module, **kwargs)
    """
    return t(key, get_current_lang(), module, **kwargs)
