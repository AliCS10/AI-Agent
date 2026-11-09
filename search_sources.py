"""Search public dataset catalog metadata; no dataset files are downloaded."""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import requests

KAGGLE_ENDPOINT = "https://www.kaggle.com/api/v1/datasets/list"
HF_ENDPOINT = "https://huggingface.co/api/datasets"

# Tiny transparent query helper, not a language model.
TRANSLATIONS = {
    "السعودية": "saudi", "السعوديه": "saudi", "الرياض": "riyadh",
    "جدة": "jeddah", "جده": "jeddah", "مبيعات": "sales",
    "المبيعات": "sales", "طقس": "weather", "الطقس": "weather",
    "حوادث": "accidents", "مرور": "traffic", "العقارات": "real estate",
    "عقارات": "real estate", "اسعار": "prices", "أسعار": "prices",
    "الأسعار": "prices", "العملاء": "customers", "عملاء": "customers",
    "الصحة": "health", "صحة": "health", "تعليم": "education",
    "الزراعة": "agriculture", "زراعة": "agriculture", "المانجو": "mango",
    "مانجو": "mango", "مترو": "metro", "النقل": "transport",
    "الرياضة": "sports", "بيانات": "", "البيانات": "", "عن": "",
    "في": "", "من": "", "لـ": "", "أبغى": "", "ابي": "",
    "احتاج": "", "احتاجها": "", "جدول": "", "ملف": "", "مشروع": "",
}


def query_variants(q: str) -> list[str]:
    q = " ".join(q.strip().split())[:180]
    terms = []
    for token in re.findall(r"[\w\u0600-\u06ff]+", q, flags=re.UNICODE):
        mapped = TRANSLATIONS.get(token)
        if mapped is None:
            # Keep English phrases and unknown Arabic tokens as-is.
            mapped = token
        if mapped:
            terms.append(mapped)
    translated = " ".join(terms)[:120]
    variants = []
    for item in [translated, q]:
        if item and item.casefold() not in [v.casefold() for v in variants]:
            variants.append(item)
    return variants[:2]


def iso_date(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, datetime):
        return value.date().isoformat()
    text = str(value)
    return text[:10]


def kaggle_search(query: str, limit: int = 20, token: str = "", username: str = "", key: str = "") -> list[dict]:
    """Official Kaggle datasets listing endpoint. Some environments require authentication."""
    headers = {"Accept": "application/json", "User-Agent": "DataFinderAI-MVP/1.0"}
    auth = None
    if token:
        headers["Authorization"] = f"Bearer {token}"
    elif username and key:
        auth = (username, key)
    response = requests.get(
        KAGGLE_ENDPOINT,
        params={"search": query, "page": 1, "pageSize": min(limit, 50)},
        headers=headers, auth=auth, timeout=14,
    )
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict):
        data = data.get("datasets", data.get("results", []))
    if not isinstance(data, list):
        raise ValueError("Unexpected Kaggle response")
    records = []
    for x in data[:limit]:
        if not isinstance(x, dict):
            continue
        ref = x.get("ref") or ""
        url = x.get("url") or (f"https://www.kaggle.com/datasets/{quote(ref, safe='/')}" if ref else "")
        if not url.startswith("https://www.kaggle.com/datasets/"):
            continue
        resources = x.get("resources") or []
        formats = []
        if isinstance(resources, list):
            for r in resources:
                ext = str((r or {}).get("fileType", "")).upper()
                if ext and ext not in formats:
                    formats.append(ext)
        license_value = x.get("licenseName") or x.get("license") or ""
        if isinstance(license_value, dict):
            license_value = license_value.get("name", "")
        records.append({
            "source": "Kaggle", "id": str(ref or x.get("id", "")),
            "title": str(x.get("title") or ref or "Dataset"),
            "description": str(x.get("subtitle") or x.get("description") or "")[:500],
            "url": url, "updated": iso_date(x.get("lastUpdated") or x.get("lastUpdatedDate") or x.get("created")),
            "downloads": x.get("downloadCount"), "likes": x.get("voteCount"),
            "license": str(license_value), "formats": formats,
        })
    return records


def hf_search(query: str, limit: int = 20, token: str = "") -> list[dict]:
    headers = {"Accept": "application/json", "User-Agent": "DataFinderAI-MVP/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.get(
        HF_ENDPOINT,
        params={"search": query, "sort": "downloads", "direction": -1, "limit": min(limit, 50), "full": "true"},
        headers=headers, timeout=14,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        raise ValueError("Unexpected Hugging Face response")
    records = []
    for x in data[:limit]:
        if not isinstance(x, dict) or not x.get("id"):
            continue
        tags = x.get("tags") or []
        license_tags = [t.split(":", 1)[1] for t in tags if isinstance(t, str) and t.startswith("license:")]
        records.append({
            "source": "Hugging Face", "id": str(x["id"]),
            "title": str(x["id"].split("/")[-1]).replace("-", " ").replace("_", " "),
            "description": "", "url": f"https://huggingface.co/datasets/{quote(str(x['id']), safe='/')}",
            "updated": iso_date(x.get("lastModified") or x.get("createdAt")),
            "downloads": x.get("downloads"), "likes": x.get("likes"),
            "license": ", ".join(license_tags), "formats": [],
        })
    return records


def rank_results(records: list[dict], query: str, sort: str) -> list[dict]:
    words = [w.lower() for w in re.findall(r"[a-zA-Z0-9\u0600-\u06ff]+", query) if len(w) >= 3]
    for r in records:
        hay = (r["title"] + " " + r["description"] + " " + r["id"]).lower()
        hits = sum(1 for w in set(words) if w in hay)
        r["match_score"] = round(100 * hits / len(set(words))) if words else 0
        # Heuristic text match indicator, NOT actual dataset quality or AI confidence.
    def safe_int(x):
        try: return int(x or 0)
        except (ValueError, TypeError): return 0
    if sort == "الأحدث":
        return sorted(records, key=lambda r: r["updated"] or "", reverse=True)
    if sort == "الأكثر تحميلًا":
        return sorted(records, key=lambda r: safe_int(r["downloads"]), reverse=True)
    return sorted(records, key=lambda r: (r["match_score"], safe_int(r["downloads"])), reverse=True)


def unique(records: list[dict]) -> list[dict]:
    found = {}
    for r in records:
        found.setdefault(r["url"], r)
    return list(found.values())
