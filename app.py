"""DataFinder AI — Arabic dataset discovery MVP."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import pandas as pd
import streamlit as st
from search_sources import hf_search, kaggle_search, query_variants, rank_results, unique

st.set_page_config(page_title="DataFinder AI | ابحث عن بيانات", page_icon="🔎", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], [data-testid="stApp"] {font-family:'Cairo',sans-serif;}
.block-container {max-width:1120px;padding-top:2rem;}
.hero {direction:rtl;text-align:right;background:linear-gradient(115deg,#e9f7f2,#f6f8fb);padding:2rem;border-radius:20px;color:#102c2b;margin-bottom:1.3rem;}
.hero h1 {font-size:2.4rem;margin:0;font-weight:800;color:#153d37;}
.hero p {margin:.6rem 0 0;color:#4f706e;}
[data-testid="stSidebar"] {direction:rtl;}
.result {direction:rtl;border:1px solid #dfe7e5;border-radius:14px;padding:1.1rem 1.2rem;margin:0 0 1rem;background:#ffffff;color:#152a2a;}
.result h3 {margin:0 0 .4rem;font-size:1.17rem;color:#143c35;overflow-wrap:anywhere;}
.result .meta {font-size:.79rem;color:#5e7775;margin-bottom:.55rem;}
.result .summary {font-size:.9rem;color:#334c4a;line-height:1.8;}
.result a {color:#097f6c!important;text-decoration:none;font-weight:700;}
.small-note {color:#687e7b;font-size:.83rem;direction:rtl;}
</style>
""", unsafe_allow_html=True)


def secret(key: str) -> str:
    value = os.getenv(key, "")
    if value:
        return value
    try:
        return str(st.secrets.get(key, ""))
    except (FileNotFoundError, KeyError, AttributeError):
        return ""


@st.cache_data(ttl=900, show_spinner=False)
def search_cached(source: str, phrase: str, kaggle_token: str, kaggle_username: str, kaggle_key: str, hf_token: str):
    if source == "Kaggle":
        return kaggle_search(phrase, 24, kaggle_token, kaggle_username, kaggle_key)
    return hf_search(phrase, 24, hf_token)


with st.sidebar:
    st.header("خيارات البحث")
    sources = st.multiselect("المصادر", ["Kaggle", "Hugging Face"], default=["Kaggle", "Hugging Face"])
    sorting = st.selectbox("ترتيب النتائج", ["الأكثر صلة", "الأحدث", "الأكثر تحميلًا"])
    fmt = st.selectbox("صيغة الملف", ["الكل", "CSV", "JSON", "PARQUET", "SQLITE"])
    min_year = st.selectbox("آخر تحديث منذ", ["أي وقت", "2025", "2024", "2023", "2020"])
    verified_only = st.checkbox("إظهار النتائج ذات ترخيص معروف فقط", value=False)
    st.divider()
    st.caption("المصادر الرسمية: Kaggle وHugging Face. البحث يسترجع بيانات وصفية وروابط، ولا ينزل الملفات.")
    st.caption("كلمة AI في الاسم تشير لرؤية المشروع. هذه النسخة تستخدم توسيع كلمات بسيط وترتيبًا نصيًا، وليس LLM.")

st.markdown('''<div class="hero"><h1>DataFinder AI</h1><p>محرك بحث عربي لاكتشاف مجموعات البيانات من مصادر حقيقية — ابحث، قارن، وافتح المصدر الأصلي.</p></div>''', unsafe_allow_html=True)

with st.form("search_form"):
    q = st.text_input("وش البيانات اللي تبحث عنها؟", placeholder="مثال: بيانات مبيعات القهوة أو Saudi traffic accidents", max_chars=180)
    submitted = st.form_submit_button("🔎 ابحث عن Datasets", type="primary", use_container_width=True)

if "search_records" not in st.session_state:
    st.session_state.search_records = []
    st.session_state.errors = {}
    st.session_state.terms = []
    st.session_state.searched = False

if submitted:
    st.session_state.search_records = []
    st.session_state.errors = {}
    st.session_state.searched = True
    if not q.strip():
        st.warning("اكتب موضوع البيانات أولًا.")
    elif not sources:
        st.warning("حدد مصدرًا واحدًا على الأقل من القائمة الجانبية.")
    else:
        phrases = query_variants(q)
        st.session_state.terms = phrases
        jobs = [(s, p) for s in sources for p in phrases]
        success = []
        errors = {}
        with st.spinner("جاري البحث في فهارس البيانات الحقيقية..."):
            with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as pool:
                future_map = {
                    pool.submit(search_cached, s, p,
                                secret("KAGGLE_API_TOKEN"), secret("KAGGLE_USERNAME"),
                                secret("KAGGLE_KEY"), secret("HF_TOKEN")): (s, p)
                    for s, p in jobs
                }
                for fut in as_completed(future_map):
                    s, p = future_map[fut]
                    try:
                        success.extend(fut.result())
                    except Exception as exc:
                        errors.setdefault(s, []).append(type(exc).__name__ + ": " + str(exc)[:160])
        st.session_state.search_records = unique(success)
        st.session_state.errors = errors

if st.session_state.searched:
    if st.session_state.terms:
        st.caption("كلمات البحث المستخدمة: " + " | ".join(st.session_state.terms))
    for source, messages in st.session_state.errors.items():
        st.warning(f"تعذّر البحث في {source}: {messages[0]}. قد يتطلب المصدر تسجيل دخول أو توفّر الخدمة.")
    records = st.session_state.search_records
    if fmt != "الكل":
        # Unknown formats are excluded, not guessed from platform names.
        records = [r for r in records if fmt in r["formats"]]
        st.info("فلتر الصيغة يعرض فقط الملفات التي أكد المصدر صيغتها. قد تُستبعد نتائج Hugging Face لأن الصيغة غير متاحة في فهرس البحث.")
    if min_year != "أي وقت":
        records = [r for r in records if r["updated"] and r["updated"][:4] >= min_year]
    if verified_only:
        records = [r for r in records if r["license"]]
    records = rank_results(records, " ".join(st.session_state.terms), sorting)
    a, b, c = st.columns(3)
    a.metric("النتائج المعروضة", len(records))
    b.metric("نتائج Kaggle", sum(r["source"] == "Kaggle" for r in records))
    c.metric("نتائج Hugging Face", sum(r["source"] == "Hugging Face" for r in records))

    if not records:
        st.info("ما ظهرت نتائج تطابق الفلاتر. جرّب كلمات إنجليزية أبسط أو شيل فلتر الصيغة أو حدّث إعدادات اتصال Kaggle.")
    else:
        export = pd.DataFrame([{k: r.get(k, "") for k in ["source", "title", "url", "updated", "downloads", "likes", "license", "formats", "description"]} for r in records])
        st.download_button("تحميل النتائج CSV", export.to_csv(index=False).encode("utf-8-sig"), "datafinder_results.csv", "text/csv")
        for r in records[:70]:
            title = r["title"].replace("<", "&lt;").replace(">", "&gt;")
            # Source URL is constructed from provider metadata and allowlisted.
            domain = urlparse(r["url"]).hostname or ""
            if domain not in {"www.kaggle.com", "huggingface.co"}:
                continue
            with st.container(border=True):
                st.markdown(f"### {title}")
                st.caption(f"{r['source']}  •  تاريخ التحديث: {r['updated'] or 'غير متوفر'}  •  الترخيص: {r['license'] or 'غير معروف'}")
                if r["description"]:
                    st.write(r["description"][:350])
                info = []
                if r["downloads"] is not None: info.append(f"التنزيلات: {r['downloads']:,}" if isinstance(r['downloads'], int) else f"التنزيلات: {r['downloads']}")
                if r["formats"]: info.append("الصيغ: " + ", ".join(r["formats"]))
                if info: st.caption(" | ".join(info))
                st.link_button("فتح Dataset على المصدر الأصلي ↗", r["url"])
        st.caption("النتائج مأخوذة من فهارس المصادر، وقد لا تعكس محتوى الملفات بدقة. راجع الترخيص والملفات قبل الاستخدام.")
else:
    st.markdown("**جرّب البحث عن:**")
    st.write("`Saudi traffic`  ·  `Coffee sales`  ·  `Mango disease`  ·  `Arabic sentiment`")
    st.info("اضغط بحث لعرض نتائج حقيقية. لا تُعرض أي مجموعات بيانات افتراضية على أنها نتائج.")

st.divider()
st.caption("DataFinder AI • MVP 1.0 • Independent discovery tool; not affiliated with Kaggle or Hugging Face.")
