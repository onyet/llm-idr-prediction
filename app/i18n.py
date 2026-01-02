"""
Modul Internasionalisasi (i18n) untuk dukungan multi-bahasa.
Mendukung: Indonesian (id), English (en), Arabic (ar)

Terjemahan dimuat dari file JSON di folder i18n:
  app/i18n/{lang}/{module}.json

Contoh:
  app/i18n/id/main.json
  app/i18n/en/rag.json
  app/i18n/ar/tradingview.json
"""

import json
from pathlib import Path
from typing import Dict, Any
from contextvars import ContextVar

from fastapi import Request

# Bahasa default
DEFAULT_LANG = "id"

# Bahasa yang didukung
SUPPORTED_LANGS = ["id", "en", "ar"]

# Path direktori i18n
I18N_DIR = Path(__file__).resolve().parent / "i18n"

# Cache untuk terjemahan yang sudah dimuat: {lang: {module: {key: value}}}
_translations_cache: Dict[str, Dict[str, Dict[str, str]]] = {}


def _load_translations(lang: str, module: str) -> Dict[str, str]:
    """
    Memuat terjemahan dari file JSON untuk bahasa dan modul tertentu.
    Mengembalikan dict kosong jika file tidak ditemukan.
    """
    cache_key = f"{lang}:{module}"
    
    # Cek cache terlebih dahulu
    if lang in _translations_cache and module in _translations_cache[lang]:
        return _translations_cache[lang][module]
    
    # Inisialisasi struktur cache
    if lang not in _translations_cache:
        _translations_cache[lang] = {}
    
    # Muat dari file
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
    Memuat dan menggabungkan semua terjemahan untuk suatu bahasa dari semua file modul.
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
    Mengekstrak kode bahasa dari header Accept-Language.
    Mengembalikan bahasa pertama yang didukung, atau bahasa default.
    """
    accept_language = request.headers.get("accept-language", "")
    
    if not accept_language:
        return DEFAULT_LANG
    
    # Parse header Accept-Language (contoh: "en-US,en;q=0.9,id;q=0.8")
    languages = []
    for part in accept_language.split(","):
        part = part.strip()
        if ";" in part:
            lang = part.split(";")[0].strip()
        else:
            lang = part
        
        # Ambil kode bahasa dasar (contoh: "en-US" -> "en")
        base_lang = lang.split("-")[0].lower()
        languages.append(base_lang)
    
    # Kembalikan bahasa pertama yang didukung
    for lang in languages:
        if lang in SUPPORTED_LANGS:
            return lang
    
    return DEFAULT_LANG


def t(key: str, lang: str = DEFAULT_LANG, module: str = None, **kwargs) -> str:
    """
    Mendapatkan terjemahan untuk sebuah key dalam bahasa yang ditentukan.
    
    Args:
        key: Kunci terjemahan
        lang: Kode bahasa (id, en, ar)
        module: Nama modul opsional untuk mencari di file spesifik terlebih dahulu
        **kwargs: Parameter format untuk interpolasi string
    
    Returns:
        String yang diterjemahkan atau key jika tidak ditemukan
    """
    translation = None
    
    # Jika modul ditentukan, cari di sana terlebih dahulu
    if module:
        translations = _load_translations(lang, module)
        translation = translations.get(key)
        
        # Fallback ke bahasa default jika tidak ditemukan
        if translation is None and lang != DEFAULT_LANG:
            translations = _load_translations(DEFAULT_LANG, module)
            translation = translations.get(key)
    
    # Jika tidak ditemukan di modul spesifik, cari di semua modul
    if translation is None:
        all_translations = _get_all_translations(lang)
        translation = all_translations.get(key)
        
        # Fallback ke bahasa default
        if translation is None and lang != DEFAULT_LANG:
            all_translations = _get_all_translations(DEFAULT_LANG)
            translation = all_translations.get(key)
    
    # Kembalikan key jika masih tidak ditemukan
    if translation is None:
        return key
    
    # Terapkan format string jika kwargs diberikan
    if kwargs:
        try:
            return translation.format(**kwargs)
        except KeyError:
            return translation
    
    return translation


def reload_translations():
    """
    Menghapus cache terjemahan untuk memaksa reload dari file.
    Berguna saat development atau ketika terjemahan diperbarui.
    """
    global _translations_cache
    _translations_cache = {}


# Variabel context untuk menyimpan bahasa saat ini (untuk digunakan dalam context async)
current_lang: ContextVar[str] = ContextVar("current_lang", default=DEFAULT_LANG)


def set_current_lang(lang: str):
    """Mengatur bahasa saat ini dalam context."""
    current_lang.set(lang)


def get_current_lang() -> str:
    """Mendapatkan bahasa saat ini dari context."""
    return current_lang.get()


def tr(key: str, module: str = None, **kwargs) -> str:
    """
    Mendapatkan terjemahan menggunakan bahasa context saat ini.
    Singkatan untuk t(key, get_current_lang(), module, **kwargs)
    """
    return t(key, get_current_lang(), module, **kwargs)
