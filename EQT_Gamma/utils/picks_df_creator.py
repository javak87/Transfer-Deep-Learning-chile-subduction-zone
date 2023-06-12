import pandas as pd

def picks_df_creator (picks):
    pick_df = []
    for p in picks:
        pick_df.append({
            "id": p.trace_id,
            "timestamp": p.peak_time.datetime,
            "prob": p.peak_value,
            "type": p.phase.lower()
        })
    
    pick_df = pd.DataFrame(pick_df)
    return pick_df
    








