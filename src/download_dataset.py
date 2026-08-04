"""
Step 1 & 1B: Download the Bank Customer Churn Prediction dataset from Kaggle
and generate the dataset authenticity proof.
"""
import os
import sys
import shutil
import traceback

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
REPORT_DIR = os.path.join(PROJECT_ROOT, "report")
RAW_FILE = os.path.join(RAW_DIR, "churn_data.csv")
PROOF_FILE = os.path.join(REPORT_DIR, "dataset_source_proof.txt")

KAGGLE_SLUG = "shrutimechlearn/churn-modelling"
EXPECTED_CSV = "Churn_Modelling.csv"

def download_dataset():
    """Download the dataset from Kaggle using kagglehub."""
    print(f"[INFO] Attempting to download dataset: {KAGGLE_SLUG}")
    
    try:
        import kagglehub
        path = kagglehub.dataset_download(KAGGLE_SLUG)
        print(f"[INFO] kagglehub download path: {path}")
        
        # Find the CSV file in the downloaded path
        csv_path = None
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.lower() == EXPECTED_CSV.lower():
                    csv_path = os.path.join(root, f)
                    break
            if csv_path:
                break
        
        if csv_path is None:
            # List all files found
            all_files = []
            for root, dirs, files in os.walk(path):
                for f in files:
                    all_files.append(os.path.join(root, f))
            print(f"[WARNING] {EXPECTED_CSV} not found. Files found: {all_files}")
            # Try to use any CSV file
            for fp in all_files:
                if fp.endswith('.csv'):
                    csv_path = fp
                    print(f"[INFO] Using CSV file: {csv_path}")
                    break
        
        if csv_path is None:
            print("[ERROR] No CSV file found in the downloaded dataset.")
            return False
        
        # Copy to raw data directory
        os.makedirs(RAW_DIR, exist_ok=True)
        shutil.copy2(csv_path, RAW_FILE)
        print(f"[SUCCESS] Dataset saved to: {RAW_FILE}")
        return True
        
    except Exception as e:
        print(f"[ERROR] kagglehub download failed: {e}")
        traceback.print_exc()
        return False


def generate_proof():
    """Generate the dataset authenticity proof file."""
    import pandas as pd
    
    print("\n" + "=" * 70)
    print("STEP 1B: DATASET AUTHENTICITY PROOF")
    print("=" * 70)
    
    df = pd.read_csv(RAW_FILE)
    
    proof_lines = []
    
    # 1. Source
    proof_lines.append("=" * 70)
    proof_lines.append("DATASET SOURCE PROOF")
    proof_lines.append("=" * 70)
    proof_lines.append("")
    proof_lines.append("1. EXACT SOURCE URL / KAGGLE DATASET SLUG:")
    proof_lines.append(f"   Kaggle slug: {KAGGLE_SLUG}")
    proof_lines.append(f"   URL: https://www.kaggle.com/datasets/{KAGGLE_SLUG}")
    proof_lines.append("")
    
    # 2. Download method
    proof_lines.append("2. DOWNLOAD METHOD & COMMAND:")
    proof_lines.append(f"   Method: kagglehub.dataset_download('{KAGGLE_SLUG}')")
    proof_lines.append(f"   Package: kagglehub (pip install kagglehub)")
    proof_lines.append(f"   File saved as: data/raw/churn_data.csv")
    proof_lines.append("")
    
    # 3. Shape
    proof_lines.append("3. df.shape:")
    proof_lines.append(f"   {df.shape}")
    proof_lines.append(f"   ({df.shape[0]} rows, {df.shape[1]} columns)")
    proof_lines.append("")
    
    # 4. Head(10)
    proof_lines.append("4. df.head(10):")
    proof_lines.append(df.head(10).to_string(index=True))
    proof_lines.append("")
    
    # 5. dtypes and null counts
    proof_lines.append("5. df.dtypes:")
    proof_lines.append(df.dtypes.to_string())
    proof_lines.append("")
    proof_lines.append("   df.isnull().sum():")
    proof_lines.append(df.isnull().sum().to_string())
    proof_lines.append("")
    
    # 6. Authenticity statement
    proof_lines.append("6. AUTHENTICITY STATEMENT:")
    proof_lines.append(f'   "This dataset was downloaded from Kaggle (slug: {KAGGLE_SLUG}, '
                       f'URL: https://www.kaggle.com/datasets/{KAGGLE_SLUG}) '
                       f'and has not been synthetically generated or altered beyond '
                       f'standard cleaning documented in Step 4."')
    proof_lines.append("")
    proof_lines.append("=" * 70)
    
    proof_text = "\n".join(proof_lines)
    
    # Print to console
    print(proof_text)
    
    # Save to file
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(PROOF_FILE, "w", encoding="utf-8") as f:
        f.write(proof_text)
    
    print(f"\n[INFO] Proof saved to: {PROOF_FILE}")
    
    return proof_text


if __name__ == "__main__":
    success = download_dataset()
    if not success:
        print("\n[FATAL] Dataset download failed. Stopping.")
        sys.exit(1)
    
    generate_proof()
