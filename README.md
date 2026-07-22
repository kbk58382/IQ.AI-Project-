# IQ.AI-Project-
This repository includes both the python script and csv file used in my power bi dashboard project.


**Python Script**
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

SEED = 42
N_DOCS = 50_000

IQAI_COST_PER_DOC = 0.10          
IQAI_SIZE_LIMIT_MB = 25.0         
HUMAN_HOURLY_RATE = 65.0          

ISSUE_CODING_RATE = 0.35          
ISSUE_CODING_MULT_RANGE = (1.4, 1.9)  

rng = np.random.default_rng(SEED)

EXTENSION_MAP = {
    "msg":  ("Email", True, 75.34),
    "eml":  ("Email", True, 1.96),
    "docx": ("Office Document", True, 1.47),
    "doc":  ("Office Document", True, 0.49),
    "xlsx": ("Office Document", True, 3.70),
    "xls":  ("Office Document", True, 1.18),
    "xlsm": ("Office Document", True, 0.17),
    "pptx": ("Office Document", True, 0.52),
    "csv":  ("Office Document", True, 0.11),
    "pdf":  ("PDF", True, 6.27),
    "html": ("Other Text/Markup", True, 1.36),
    "htm":  ("Other Text/Markup", True, 0.23),
    "xml":  ("Other Text/Markup", True, 2.09),
    "ics":  ("Other Text/Markup", True, 0.14),
    "json": ("Chat/Messaging Export", True, 0.10),
    "txt":  ("Chat/Messaging Export", True, 0.09),
    "jpg":  ("Image", False, 1.28),
    "png":  ("Image", False, 0.87),
    "tif":  ("Image", False, 0.38),
    "jpeg": ("Image", False, 0.19),
    "heic": ("Image", False, 0.16),
    "mp3":  ("Audio", False, 0.26),
    "mp4":  ("Video", False, 0.14),
    "wav":  ("Audio", False, 0.06),
    "scd":  ("Other Non-Text", False, 0.15),
    "(blank)": ("Other Non-Text", False, 0.37),
}

exts = list(EXTENSION_MAP.keys())
weights = np.array([EXTENSION_MAP[e][2] for e in exts], dtype=float)
weights = weights / weights.sum()

BASE_REVIEW_MIN = {
    "Email": 1.5,
    "Office Document": 3.0,
    "PDF": 2.5,
    "Other Text/Markup": 2.0,
    "Chat/Messaging Export": 1.0,
    "Image": 4.0,
    "Audio": 9.0,   
    "Video": 10.0,  
    "Other Non-Text": 3.5,
}

COMPLEXITY_LEVELS = ["Low", "Medium", "High"]
COMPLEXITY_WEIGHTS = {
    "Email": [0.55, 0.35, 0.10],
    "Office Document": [0.25, 0.45, 0.30],
    "PDF": [0.30, 0.45, 0.25],
    "Other Text/Markup": [0.45, 0.40, 0.15],
    "Chat/Messaging Export": [0.60, 0.30, 0.10],
    "Image": [0.40, 0.40, 0.20],
    "Audio": [0.20, 0.40, 0.40],
    "Video": [0.20, 0.40, 0.40],
    "Other Non-Text": [0.40, 0.40, 0.20],
}
COMPLEXITY_MULT = {"Low": 0.8, "Medium": 1.0, "High": 1.6}

MATTERS = [
    {
        "name": "Matter A - Commercial Contract Dispute",
        "share": 0.6,
        "custodians": [f"Custodian A-{i:02d}" for i in range(1, 26)],
        "date_start": datetime(2026, 1, 5),
        "date_end": datetime(2026, 4, 30),
    },
    {
        "name": "Matter B - Internal Compliance Investigation",
        "share": 0.4,
        "custodians": [f"Custodian B-{i:02d}" for i in range(1, 19)],
        "date_start": datetime(2026, 2, 1),
        "date_end": datetime(2026, 6, 15),
    },
]


def random_dates(start, end, n):
    delta_days = (end - start).days
    offsets = rng.integers(0, delta_days + 1, size=n)
    return [start + timedelta(days=int(d)) for d in offsets]


def build_matter(matter, n):
    ext_choices = rng.choice(exts, size=n, p=weights)
    categories = np.array([EXTENSION_MAP[e][0] for e in ext_choices])
    has_text = np.array([EXTENSION_MAP[e][1] for e in ext_choices])

    # File size (MB) - lognormal, roughly scaled by category
    size_base = {
        "Email": 0.3, "Office Document": 1.5, "PDF": 2.5,
        "Other Text/Markup": 0.5, "Chat/Messaging Export": 0.2,
        "Image": 4.0, "Audio": 8.0, "Video": 45.0, "Other Non-Text": 3.0,
    }
    sizes = np.array([
        max(0.01, rng.lognormal(mean=np.log(size_base[c]), sigma=0.9))
        for c in categories
    ])

    exceeds_limit = sizes > IQAI_SIZE_LIMIT_MB
    iqai_eligible = has_text & (~exceeds_limit)

    complexity = np.array([
        rng.choice(COMPLEXITY_LEVELS, p=COMPLEXITY_WEIGHTS[c]) for c in categories
    ])
    complexity_mult = np.array([COMPLEXITY_MULT[c] for c in complexity])

    issue_coding = rng.random(n) < ISSUE_CODING_RATE
    issue_mult = np.where(
        issue_coding,
        rng.uniform(ISSUE_CODING_MULT_RANGE[0], ISSUE_CODING_MULT_RANGE[1], size=n),
        1.0,
    )

    noise = rng.lognormal(mean=0.0, sigma=0.25, size=n)

    base_minutes = np.array([BASE_REVIEW_MIN[c] for c in categories])
    human_minutes = base_minutes * complexity_mult * issue_mult * noise
    human_cost = (human_minutes / 60.0) * HUMAN_HOURLY_RATE

    iqai_cost = np.where(iqai_eligible, IQAI_COST_PER_DOC, np.nan)

    iqai_seconds = np.where(
        iqai_eligible, rng.uniform(1.5, 6.0, size=n), np.nan
    )

    cost_diff = human_cost - iqai_cost  # NaN where not eligible

    custodians = rng.choice(matter["custodians"], size=n)
    received = random_dates(matter["date_start"], matter["date_end"], n)

    review_status = np.where(
        iqai_eligible,
        "Eligible - IQ.AI Processed",
        np.where(has_text, "Text Extracted, Exceeds Size Limit - Human Only",
                 "No Extracted Text - Manual/Transcript Review Only"),
    )

    df = pd.DataFrame({
        "MatterName": matter["name"],
        "Custodian": custodians,
        "FileExtension": ext_choices,
        "FileCategory": categories,
        "FileSizeMB": np.round(sizes, 3),
        "HasExtractedText": has_text,
        "ExceedsIQAISizeLimit": exceeds_limit,
        "IQAI_Eligible": iqai_eligible,
        "ReviewComplexity": complexity,
        "IssueCodingRequired": issue_coding,
        "HumanReviewMinutes": np.round(human_minutes, 2),
        "HumanHourlyRate": HUMAN_HOURLY_RATE,
        "HumanReviewCost": np.round(human_cost, 4),
        "IQAI_CostPerDoc": iqai_cost,
        "IQAI_ProcessingSeconds": np.round(iqai_seconds, 2) if hasattr(iqai_seconds, "round") else iqai_seconds,
        "CostDifference_HumanMinusIQAI": np.round(cost_diff, 4),
        "ReceivedDate": received,
        "ReviewStatus": review_status,
    })
    return df


def main():
    frames = []
    remaining = N_DOCS
    for i, matter in enumerate(MATTERS):
        n = int(round(N_DOCS * matter["share"])) if i < len(MATTERS) - 1 else remaining
        n = min(n, remaining)
        frames.append(build_matter(matter, n))
        remaining -= n

    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "DocID", [f"DOC-{i+1:06d}" for i in range(len(out))])

    out_path = "/mnt/user-data/outputs/mock_doc_review_dataset.csv"
    out.to_csv(out_path, index=False)

    print(f"Wrote {len(out):,} rows to {out_path}")
    print("\n--- Quick sanity check ---")
    print(out["MatterName"].value_counts())
    print()
    print(out["FileCategory"].value_counts())
    print()
    print("IQ.AI eligible share:", round(out["IQAI_Eligible"].mean() * 100, 1), "%")
    print()
    eligible = out[out["IQAI_Eligible"]]
    print("Total human cost (eligible docs only): $", round(eligible["HumanReviewCost"].sum(), 2))
    print("Total IQ.AI cost (eligible docs only):  $", round(eligible["IQAI_CostPerDoc"].sum(), 2))
    print("Implied savings:                        $",
          round(eligible["HumanReviewCost"].sum() - eligible["IQAI_CostPerDoc"].sum(), 2))


if __name__ == "__main__":
    main()
