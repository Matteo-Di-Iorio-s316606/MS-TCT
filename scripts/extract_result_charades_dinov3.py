import pandas as pd

p="runs/mstct_charades_dinov3_6fps/logs/train_log.csv"
df=pd.read_csv(p)
best=df.loc[df["val_map"].idxmax()]

print(df[["epoch","train_map","val_map","val_loss","lr"]].tail(15).to_string(index=False))
print("Best epoch:", int(best["epoch"]))
print("Best val_map:", best["val_map"])
print("Best val_loss:", best["val_loss"])
print("Final epoch:", int(df["epoch"].iloc[-1]), "final val_map:", df["val_map"].iloc[-1])