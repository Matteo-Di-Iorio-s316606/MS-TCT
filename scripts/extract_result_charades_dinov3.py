#!/usr/bin/env python
import pandas as pd
from pathlib import Path

p = Path("runs_256/mstct_charades_dinov3_24fps/logs/train_log.csv")
df = pd.read_csv(p)

# Best epoch (massimo val_map)
best = df.loc[df["val_map"].idxmax()]

# Ultime 15 epoche
last_15 = df[["epoch","train_map","val_map","val_loss","lr"]].tail(15)

# Costruzione summary
summary = {
    "best_epoch": int(best["epoch"]),
    "best_val_map": float(best["val_map"]),
    "best_val_loss": float(best["val_loss"]),
    "final_epoch": int(df["epoch"].iloc[-1]),
    "final_val_map": float(df["val_map"].iloc[-1]),
}

# -----------------------
# Salvataggio TXT
# -----------------------
txt_out = p.parent / "results_summary.txt"

with open(txt_out, "w") as f:
    f.write("==== LAST 15 EPOCHS ====\n")
    f.write(last_15.to_string(index=False))
    f.write("\n\n==== SUMMARY ====\n")
    for k, v in summary.items():
        f.write(f"{k}: {v}\n")

# -----------------------
# Salvataggio CSV compatto
# -----------------------
csv_out = p.parent / "results_summary.csv"
pd.DataFrame([summary]).to_csv(csv_out, index=False)

print("Saved to:")
print(txt_out)
print(csv_out)