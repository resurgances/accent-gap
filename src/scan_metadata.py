"""
Step 1: scan the EdAcc dataset's METADATA only (no audio download).

EdAcc is stored on Hugging Face as Parquet files (a column-based table format).
Because Parquet stores each column separately, we can read just the small text
columns (speaker, text, accent, l1, ...) over the internet and skip the big
"audio" column. That lets us look at the whole dataset (~7 GB) while only
downloading a few MB.

Outputs:
  data_cache/metadata.csv        one row per clip (all splits), with where it lives
  results/dataset_overview.csv   clips and speakers per accent / l1 / split
"""
import os
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

REPO = "datasets/edinburghcstr/edacc/data"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META_CSV = os.path.join(ROOT, "data_cache", "metadata.csv")
OVERVIEW_CSV = os.path.join(ROOT, "results", "dataset_overview.csv")


def scan():
    fs = HfFileSystem()
    rows = []
    for path in sorted(fs.ls(REPO, detail=False)):
        if not path.endswith(".parquet"):
            continue
        split = os.path.basename(path).split("-")[0]  # "validation" or "test"
        pf = pq.ParquetFile(path, filesystem=fs)
        row_offset = 0
        # Remember which "row group" (chunk) each clip is in, so later we can
        # download only the chunks that contain clips we actually use.
        for rg in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(rg, columns=["speaker", "text", "accent", "raw_accent",
                                               "gender", "l1", "audio.path"])
            for i, r in enumerate(t.to_pylist()):
                rows.append({
                    "split": split,
                    "shard": path,
                    "row_group": rg,
                    "row_in_group": i,
                    "clip_id": r["audio"]["path"].replace(".wav", ""),
                    "speaker": r["speaker"],
                    "accent": r["accent"],
                    "raw_accent": r["raw_accent"],
                    "gender": r["gender"],
                    "l1": r["l1"],
                    "text": r["text"],
                })
            row_offset += t.num_rows
        print(f"scanned {os.path.basename(path)}: total rows so far {len(rows)}")
    return pd.DataFrame(rows)


def main():
    os.makedirs(os.path.dirname(META_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(OVERVIEW_CSV), exist_ok=True)
    if os.path.exists(META_CSV):
        df = pd.read_csv(META_CSV)
        print("metadata already cached, loaded", len(df), "rows")
    else:
        df = scan()
        # Clip number inside the conversation, e.g. EDACC-C06-17 -> 17.
        # Used later to keep clips in conversation order.
        df["clip_num"] = df["clip_id"].str.rsplit("-", n=1).str[1].astype(int)
        df.to_csv(META_CSV, index=False)

    overview = []
    for field in ["accent", "l1"]:
        g = df.groupby(["split", field]).agg(clips=("clip_id", "count"),
                                              speakers=("speaker", "nunique")).reset_index()
        g = g.rename(columns={field: "value"})
        g.insert(0, "field", field)
        overview.append(g)
    ov = pd.concat(overview).sort_values(["field", "split", "clips"], ascending=[True, True, False])
    ov.to_csv(OVERVIEW_CSV, index=False)
    pd.set_option("display.max_rows", 500, "display.width", 200)
    print(ov.to_string(index=False))


if __name__ == "__main__":
    main()
