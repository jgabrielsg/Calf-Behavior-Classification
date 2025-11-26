import polars as pl

# Function to better understand the Time_AJD parquet
def get_colisions(parquet_path):
    print(f"🕵️ Hunting collisions in: {parquet_path}")
    df = pl.read_parquet(parquet_path, columns=["dateTime"])
    df = df.sort("dateTime")
    
    # Compute delta
    df = df.with_columns(
        (pl.col("dateTime") - pl.col("dateTime").shift(1)).alias("delta_t")
    )
    
    # Count collisions (checks if there is more than one calf, or if it was not 25hz at certain points)
    # 25hz -> 40000 microseconds
    colisions = df.filter(pl.col("delta_t") <= pl.duration(microseconds=38500))
    
    total = len(df)
    n_colisions = len(colisions)
    
    print(f"Total Rows: {total}")
    print(f"Total Collisions: {n_colisions}")
    
    if n_colisions > 0:
        print("\nCollision Examples:")
        print(colisions.head())

get_colisions('Time_Adj_Raw_Data.parquet')
