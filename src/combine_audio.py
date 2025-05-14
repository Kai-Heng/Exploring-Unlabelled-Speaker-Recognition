import os
import soundfile as sf
import numpy as np
from tqdm import tqdm

RAW_DIR = "data/raw"
OUT_DIR = "data/combined"
os.makedirs(OUT_DIR, exist_ok=True)

speakers = [f"{i:02d}" for i in range(1, 61)]   # "01" to "60"
positions = range(50)                           # 0 to 49
digits = range(10)                              # 0 to 9

for spk in tqdm(speakers, desc="Speakers"):
    for pos in positions:
        combined = []
        sr = None
        success = True
        for d in digits:
            filename = f"{spk}_{d}_{spk}_{pos}.wav"
            filepath = os.path.join(RAW_DIR, filename)
            if not os.path.exists(filepath):
                print(f"[Missing] {filepath}")
                success = False
                break
            audio, sr = sf.read(filepath)
            combined.append(audio)
        if success:
            combined_audio = np.concatenate(combined)
            out_path = os.path.join(OUT_DIR, f"{spk}_combined_{pos}.wav")
            sf.write(out_path, combined_audio, sr)
