import re
import json
from pathlib import Path
from app.i18n import SUPPORTED_LANGS, I18N_DIR

# Scan repository for usages of t('key' or tr('key')
pattern = re.compile(r"\b(?:t|tr)\(\s*['\"]([^'\"]+)['\"]")

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / 'app'
py_files = list(APP_DIR.glob('**/*.py'))
used_keys = set()
for p in py_files:
    text = p.read_text(encoding='utf-8')
    for m in pattern.finditer(text):
        used_keys.add(m.group(1))

# Load translations per language (merge all module files for each lang)
lang_translations = {}
for lang in SUPPORTED_LANGS:
    lang_dir = I18N_DIR / lang
    keys = set()
    if lang_dir.exists():
        for jf in lang_dir.glob('*.json'):
            try:
                data = json.loads(jf.read_text(encoding='utf-8'))
                keys.update(data.keys())
            except Exception:
                pass
    lang_translations[lang] = keys


import warnings

def test_all_used_keys_present_in_default_and_english():
    # Ensure every used key exists in the default language ('id') and in English ('en')
    missing_in_required = {}
    for key in sorted(used_keys):
        not_present_required = [lang for lang in ('id', 'en') if key not in lang_translations.get(lang, set())]
        if not_present_required:
            missing_in_required[key] = not_present_required
    assert not missing_in_required, f"Missing translation keys in required languages: {missing_in_required}"


def test_warn_missing_in_other_languages():
    # Warn if keys are absent in other supported languages, but do not fail CI for now
    warn_missing = {}
    for key in sorted(used_keys):
        not_present = [lang for lang in SUPPORTED_LANGS if key not in lang_translations.get(lang, set())]
        if not_present:
            warn_missing[key] = not_present
    if warn_missing:
        warnings.warn(f"Some translation keys are missing in non-required languages: {warn_missing}")
