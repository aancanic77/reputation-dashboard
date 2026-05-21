import requests
import pandas as pd
from typing import List, Dict
from utils import load_reddit_dataframe, ensure_mock_data

REDDIT_BASE_URL = "https://www.reddit.com"
HEADERS = {"User-Agent": "ReputationDashboard/1.0 by student"}

# ============================================================
# CONTROL MANUAL DEMO MODE (ca pe localhost)
# ============================================================
DEMO_MODE = False   # <<< EXACT ca pe localhost

# ============================================================
# FLAG — dacă pipeline-ul a folosit fallback
# ============================================================
PIPELINE_USED_FALLBACK = False

# ============================================================
# DEMO MOCK — EXACT 60 comentarii
# ============================================================
def generate_demo_mock(n=60):
    companies = ["Apple", "Google", "Samsung"]
    rows = []
    for i in range(n):
        comp = companies[i % 3]
        rows.append({
            "comment_id": f"demo_{i}",
            "text": f"Demo comment {i} about {comp}",
            "subreddit": comp.lower(),
            "author": f"user_{i}",
            "created_utc": 1710000000 + i,
            "company": comp,
        })
    return pd.DataFrame(rows)

# ============================================================
# FETCH REDDIT POSTS (modern: /new.json)
# ============================================================
def fetch_reddit_comments(subreddits: List[str], limit: int = 50) -> pd.DataFrame:
    global PIPELINE_USED_FALLBACK

    if DEMO_MODE:
        PIPELINE_USED_FALLBACK = True
        return generate_demo_mock(60)

    rows: List[Dict] = []

    for sub in subreddits:
        url = f"{REDDIT_BASE_URL}/r/{sub}/new.json?limit={limit}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            if resp.status_code != 200:
                raise Exception("Reddit returned non-200")

            data = resp.json()
            for child in data.get("data", {}).get("children", []):
                d = child.get("data", {})

                text = d.get("selftext") or d.get("title") or ""

                rows.append(
                    {
                        "comment_id": d.get("id"),
                        "text": text,
                        "subreddit": d.get("subreddit", sub),
                        "author": d.get("author"),
                        "created_utc": d.get("created_utc"),
                    }
                )

        except Exception:
            PIPELINE_USED_FALLBACK = True
            return fallback_data()

    if not rows:
        PIPELINE_USED_FALLBACK = True
        return fallback_data()

    return pd.DataFrame(rows)

# ============================================================
# FALLBACK DATA
# ============================================================
def fallback_data() -> pd.DataFrame:
    global PIPELINE_USED_FALLBACK
    PIPELINE_USED_FALLBACK = True

    # 1) DB fallback
    try:
        df = load_reddit_dataframe()
        if df is not None and len(df) > 0:
            df = df.rename(columns={"content": "text"})
            return df[["comment_id", "text", "subreddit", "author", "created_utc"]]
    except:
        pass

    # 2) CSV fallback
    try:
        csv_path = ensure_mock_data()
        df = pd.read_csv(csv_path)
        return df
    except:
        pass

    # 3) mock local
    return generate_demo_mock(60)

# ============================================================
# EXTRACT FIELDS
# ============================================================
def extract_fields(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["comment_id", "text", "subreddit", "author", "created_utc"]
    existing = [c for c in cols if c in df.columns]
    return df[existing].copy()

# ============================================================
# MAP TO COMPANY
# ============================================================
def map_to_company(df: pd.DataFrame) -> pd.DataFrame:
    def _map(row):
        sub = str(row.get("subreddit", "")).lower()
        text = str(row.get("text", "")).lower()

        if "apple" in sub or "iphone" in text or "mac" in text:
            return "Apple"
        if "google" in sub or "android" in text or "pixel" in text:
            return "Google"
        if "samsung" in sub or "galaxy" in text:
            return "Samsung"
        return "Other"

    df = df.copy()
    df["company"] = df.apply(_map, axis=1)
    return df[df["company"].isin(["Apple", "Google", "Samsung"])]

# ============================================================
# PIPELINE COMPLET
# ============================================================
def run_reddit_pipeline_live(limit_per_sub: int = 20) -> pd.DataFrame:
    global PIPELINE_USED_FALLBACK
    PIPELINE_USED_FALLBACK = False

    subreddits = ["apple", "google", "samsung"]

    try:
        raw = fetch_reddit_comments(subreddits, limit_per_sub)
    except Exception:
        PIPELINE_USED_FALLBACK = True
        raw = fallback_data()

    extracted = extract_fields(raw)
    mapped = map_to_company(extracted)

    if DEMO_MODE:
        mapped["dl_label"] = "disabled"

    return mapped.reset_index(drop=True)
