import json
import pandas as pd

INP = "out/retrieval_results.jsonl"
OUT = "out/retrieval_intervals.csv"

rows = []
with open(INP, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        qidx = r.get("query_index")
        qts = r.get("query_timestamp_sec")
        preds = r.get("predicted_labels", [])
        pred_label = preds[0]["label"] if preds else None
        pred_conf = preds[0]["conf"] if preds else None

        intervals = r.get("intervals", [])
        if not intervals:
            rows.append({
                "query_index": qidx,
                "query_timestamp_sec": qts,
                "pred_label": pred_label,
                "pred_conf": pred_conf,
                "start_timestamp": None,
                "end_timestamp": None,
                "class_label": None,
                "number_of_supporting_detections": 0,
            })
        else:
            for itv in intervals:
                rows.append({
                    "query_index": qidx,
                    "query_timestamp_sec": qts,
                    "pred_label": pred_label,
                    "pred_conf": pred_conf,
                    "start_timestamp": itv.get("start_timestamp"),
                    "end_timestamp": itv.get("end_timestamp"),
                    "class_label": itv.get("class_label"),
                    "number_of_supporting_detections": itv.get("number_of_supporting_detections"),
                })

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print("Wrote:", OUT)
print(df.head(20))