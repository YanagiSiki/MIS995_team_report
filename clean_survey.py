"""
Survey data cleaning script
Input : 生成式 AI 個人化記憶與持續使用意圖調查 (回覆) - 表單回覆 1.csv
Output: survey_cleaned.csv
"""

import pandas as pd
import numpy as np
import re

# ── 1. Load ──────────────────────────────────────────────────────────────────
src = "生成式 AI 個人化記憶與持續使用意圖調查 (回覆) - 表單回覆 1.csv"
df = pd.read_csv(src)

# ── 2. Drop timestamp ─────────────────────────────────────────────────────────
df = df.drop(columns=[df.columns[0]])  # 時間戳記

# ── 3. Filter: keep only "是" for memory/personalisation feature ──────────────
screen_col = df.columns[0]             # 是否已開啟「記憶/個人化指令」功能
df = df[df[screen_col].str.startswith("是")].copy()
df = df.drop(columns=[screen_col])     # screening column no longer needed

# ── 4. Rename columns ─────────────────────────────────────────────────────────
# Original order after dropping timestamp + screen col:
# PM1 PM2 PM3 PM4  SC1 SC2 SC3 SC4 SC5 SC6  CU1 CU2 CU3 CU4  TR1 TR2 TR3 TR4
# CV1(platform) CV2(months) CV3(freq) CV4(export) EMAIL PM5 CU5
# GENDER AGE EDU JOB WORK_AI AI_FAMILIARITY
new_cols = [
    "PM1", "PM2", "PM3", "PM4",
    "SC1", "SC2", "SC3", "SC4", "SC5", "SC6",
    "CU1", "CU2", "CU3", "CU4",
    "TR1", "TR2", "TR3", "TR4",
    "CV1", "CV2", "CV3", "CV4",
    "EMAIL",
    "PM5", "CU5",
    "GENDER", "AGE", "EDU", "JOB", "WORK_AI", "AI_FAM",
]
df.columns = new_cols

# ── 5. Drop email ─────────────────────────────────────────────────────────────
df = df.drop(columns=["EMAIL"])

# ── 6. Normalise CV2 (usage duration → integer months) ───────────────────────
def parse_months(val):
    val = str(val).strip()
    # "2年"  →  24
    m = re.match(r"(\d+)\s*年", val)
    if m:
        return int(m.group(1)) * 12
    # "24+" or "24以上"  →  24
    m = re.match(r"(\d+)\s*[\+以]", val)
    if m:
        return int(m.group(1))
    # plain integer
    m = re.match(r"^(\d+)$", val)
    if m:
        return int(m.group(1))
    return np.nan

df["CV2"] = df["CV2"].apply(parse_months)

# ── 7. Encode CV4 (export experience) ────────────────────────────────────────
df["CV4"] = df["CV4"].map({"是": 1, "否": 0})

# ── 8. Encode CV1 (AI platform) ──────────────────────────────────────────────
platform_map = {"ChatGPT": 1, "Gemini": 2, "Claude": 3}
df["CV1"] = df["CV1"].apply(lambda x: platform_map.get(str(x).strip(), 4))

# ── 9. Encode demographic variables ──────────────────────────────────────────

# GENDER  男=0  女=1  其他/不明=NaN
def encode_gender(v):
    v = str(v).strip()
    if v == "男":
        return 0
    if v == "女":
        return 1
    return np.nan  # 秘密 / 麵包 / etc.

df["GENDER"] = df["GENDER"].apply(encode_gender)

# AGE  21–30=1  31–40=2  41歲以上=3
age_map = {"21–30歲": 1, "31–40歲": 2, "41歲以上": 3}
df["AGE"] = df["AGE"].map(age_map)

# EDU  大學=1  碩士=2  博士以上=3
edu_map = {"大學": 1, "碩士": 2, "博士以上": 3}
df["EDU"] = df["EDU"].map(edu_map)

# JOB  上班族=1  學生=2  自由業=3  科技業→上班族=1  無職=4  役男=5
job_map = {"上班族": 1, "學生": 2, "自由業": 3, "科技業": 1, "無職": 4, "役男": 5}
df["JOB"] = df["JOB"].map(job_map)

# WORK_AI  是=1  否=0
df["WORK_AI"] = df["WORK_AI"].map({"是": 1, "否": 0})

# AI_FAM  不熟=1  普通=2  熟悉=3  非常熟悉=4
fam_map = {"不熟": 1, "普通": 2, "熟悉": 3, "非常熟悉": 4}
df["AI_FAM"] = df["AI_FAM"].map(fam_map)

# ── 10. Drop first row (2026/4/30 – no demographics, likely test entry) ────────
df = df.iloc[1:].copy()

# ── 11. Impute PM5 missing values with each respondent's row-mean of PM1–PM4 ──
# PM5 belongs to the same IV construct; intra-construct mean imputation is
# appropriate for structural missingness (item added after early responses).
df["PM5"] = df.apply(
    lambda r: round(df[["PM1","PM2","PM3","PM4"]].loc[r.name].mean(), 2)
    if pd.isna(r["PM5"]) else r["PM5"],
    axis=1,
)

# ── 12. Drop CU5 (almost entirely missing, not usable) ────────────────────────
df = df.drop(columns=["CU5"])

# ── 13. Reset index ──────────────────────────────────────────────────────────
df = df.reset_index(drop=True)

# ── 14. Save ─────────────────────────────────────────────────────────────────
out = "survey_cleaned.csv"
df.to_csv(out, index=False, encoding="utf-8-sig")
print(f"Done. {len(df)} valid rows written to {out}")
print(df.dtypes)
print(df.head(3).to_string())
