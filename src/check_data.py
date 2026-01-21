from src.config import CFG
from src.dataset import build_dataframe, split_dataframe

if __name__ == "__main__":
    cfg = CFG()

    print("ROOT:", cfg.ROOT)
    print("DATA_DIR:", cfg.DATA_DIR)
    print("CSV_PATH exists:", cfg.CSV_PATH.exists())
    print("ARCHIVE_DIR exists:", cfg.ARCHIVE_DIR.exists())

    df = build_dataframe(cfg)
    print("\nTotal samples:", len(df))
    print("\nSamples per class:\n", df["label"].value_counts())

    train_df, val_df, test_df = split_dataframe(cfg, df)
    print("\nSplit sizes:")
    print("train:", len(train_df), "val:", len(val_df), "test:", len(test_df))

    print("\nExample rows:\n", df.head())
