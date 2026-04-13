import pandas as pd
from pathlib import Path

ROOT = Path('g:/5G-energy-comsuption-DL/Small-sample-MLM')
bs_path = ROOT / 'BSinfo.csv'
cl_path = ROOT / 'CLdata.csv'
ec_path = ROOT / 'ECdata.csv'

print("--- Loading Files ---")
bs = pd.read_csv(bs_path, encoding='gbk')
bs.columns = bs.columns.str.strip()
cl = pd.read_csv(cl_path)
cl.columns = cl.columns.str.strip()
ec = pd.read_csv(ec_path)
ec.columns = ec.columns.str.strip()

print(f"CLdata Rows: {len(cl)}")
print(f"ECdata Rows: {len(ec)}")
print(f"BSinfo Rows: {len(bs)}")

# Step 1: CL x EC
cl_renamed = cl.rename(columns={'Time': 'time', 'BS': 'bs_id'})
ec_renamed = ec.rename(columns={'Time': 'time', 'BS': 'bs_id'})
m1 = pd.merge(cl_renamed, ec_renamed[['time', 'bs_id', 'Energy']], on=['time', 'bs_id'], how='inner')
print(f"After Step 1 (CL x EC inner join): {len(m1)}")

# Step 2: M1 x BSinfo
bs_renamed = bs.rename(columns={'BS': 'bs_id', 'CellName': 'cell_name'})
m2 = pd.merge(m1, bs_renamed, on='bs_id', how='left')
print(f"After Step 2 (M1 x BSinfo left join on bs_id): {len(m2)}")

# Check for duplicates per BS in BSinfo
bs_counts = bs_renamed['bs_id'].value_counts()
print(f"\nBSinfo duplicate check:")
print(f"Total Unique BS: {len(bs_counts)}")
print(f"BS with >1 record: {(bs_counts > 1).sum()}")
if (bs_counts > 1).any():
    print("Example BS with multiple rows in BSinfo:")
    example_bs = bs_counts.index[0]
    print(bs_renamed[bs_renamed['bs_id'] == example_bs])
