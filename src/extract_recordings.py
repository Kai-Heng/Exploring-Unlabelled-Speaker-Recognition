import argparse
import os
import shutil
from glob import glob
from pathlib import Path
from tqdm import tqdm


def main(src_root: Path, dst_raw: Path) -> None:
    if not src_root.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src_root}")

    dst_raw.mkdir(parents=True, exist_ok=True)

    # look for all *.wav files two levels down:  <src>/<speaker>/<digit>.wav
    wav_files = glob(str(src_root / "*" / "*.wav"))
    if not wav_files:
        raise RuntimeError(f"No *.wav files detected under {src_root}")

    for path in tqdm(wav_files, desc="Copying"):
        path = Path(path)
        speaker_id = path.parent.name          # folder name == speaker ID
        new_name   = f"{speaker_id}_{path.name}"
        shutil.copy(path, dst_raw / new_name)

    print(f"Copied {len(wav_files)} files →  {dst_raw.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flatten Audio‑MNIST folder tree into data/raw/"
    )
    parser.add_argument(
        "--src",
        required=True,
        help="Path to Audio‑MNIST *data* directory (contains 60 speaker folders)",
    )
    parser.add_argument(
        "--dst",
        default="data/raw",
        help="Destination directory for the flattened WAV files "
             "(default: %(default)s)",
    )
    args = parser.parse_args()
    main(Path(args.src).expanduser(), Path(args.dst).expanduser())
