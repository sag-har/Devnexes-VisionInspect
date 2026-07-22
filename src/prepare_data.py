import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

def audit_and_split_data(data_dir="data/raw", seed=42):
    # Search for all images inside the raw directory or subdirectories
    image_paths = glob.glob(os.path.join(data_dir, "**", "*.jpg"), recursive=True)
    
    if not image_paths:
        print(f"\n⚠️ 0 Images found! Double-check your setup.")
        print(f"Make sure your downloaded images are pasted inside the folder: '{os.path.abspath(data_dir)}'")
        return None, None, None

    data = []
    for path in image_paths:
        filename = os.path.basename(path)
        # Parse standard NEU class names (e.g., crazing_1.jpg -> crazing)
        if "_" in filename:
            class_name = filename.split('_')[0]
        else:
            class_name = "unknown"
        data.append({"path": path, "label_name": class_name})
        
    df = pd.DataFrame(data)
    
    # Map text classes to dynamic numerical values
    unique_classes = sorted(df['label_name'].unique())
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    df['label'] = df['label_name'].map(class_to_idx)
    
    # --- WEEK 1 QUALITY GATE: Dataset Audit Logs ---
    print("\n" + "="*50)
    print("📊 DEVNEXES VISIONINSPECT - WEEK 1 DATASET AUDIT")
    print("="*50)
    print(f"✅ Total Defect Images Discovered: {len(df)}")
    print("\n🗂️ Class Distribution Matrix:")
    print(df['label_name'].value_counts())
    print("="*50)
    
    # --- WEEK 2 QUALITY GATE: Leakage-Free Stratified Split ---
    train_df, val_test_df = train_test_split(
        df, test_size=0.30, random_state=seed, stratify=df['label']
    )
    val_df, test_df = train_test_split(
        val_test_df, test_size=0.50, random_state=seed, stratify=val_test_df['label']
    )
    
    print(f"📈 Split Verification (70/15/15):")
    print(f"   ↳ Train Split : {len(train_df)} images")
    print(f"   ↳ Val Split   : {len(val_df)} images")
    print(f"   ↳ Test Split  : {len(test_df)} images")
    print("="*50 + "\n")
    
    return train_df.to_dict('records'), val_df.to_dict('records'), test_df.to_dict('records')

if __name__ == "__main__":
    audit_and_split_data()