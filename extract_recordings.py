import os
import shutil
from glob import glob

SRC_ROOT = "/Users/kaiheng/Downloads/data/"
DST_RAW  = os.path.join("/Users/kaiheng/Documents/Exploring-Unlabelled-Speaker-Recognition/data/", "raw")

os.makedirs(DST_RAW, exist_ok=True)

# Look for all .wav files under data/*/
wav_files = glob(os.path.join(SRC_ROOT, "*", "*.wav"))

count = 0
for path in wav_files:
    speaker_id = os.path.basename(os.path.dirname(path))
    filename = os.path.basename(path)
    new_name = f"{speaker_id}_{filename}"
    dst_path = os.path.join(DST_RAW, new_name)
    shutil.copy(path, dst_path)
    count += 1

print(f"Copied {count} files to {DST_RAW}")
