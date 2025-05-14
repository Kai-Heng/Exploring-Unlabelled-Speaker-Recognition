#!/usr/bin/env python
"""Embed each cleaned WAV using SpeechBrain ECAPA‑TDNN."""
import os, argparse, torch, numpy as np, soundfile as sf
from tqdm import tqdm
from speechbrain.inference import EncoderClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def read_wav(path):
    wav, sr = sf.read(path)
    return torch.tensor(wav, dtype=torch.float32).to(DEVICE), sr

def main(clean_dir, emb_dir):
    os.makedirs(emb_dir, exist_ok=True)
    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.join(emb_dir, "ecapa"),
    ).to(DEVICE)
    files = sorted(p for p in os.listdir(clean_dir) if p.lower().endswith(".wav"))
    embeddings = []
    names = []
    for fname in tqdm(files, desc="Embedding"):
        wav, sr = read_wav(os.path.join(clean_dir, fname))
        if sr != 16_000:
            raise ValueError("Pre‑processing should resample to 16 kHz.")
        with torch.no_grad():
            emb = encoder.encode_batch(wav.unsqueeze(0))
        embeddings.append(emb.squeeze().cpu().numpy())
        names.append(fname)
    embeddings = np.stack(embeddings)
    np.save(os.path.join(emb_dir, "embeddings.npy"), embeddings)
    with open(os.path.join(emb_dir, "filenames.txt"), "w") as f:
        for name in names:
            f.write(name + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_dir", default="data/clean")
    ap.add_argument("--emb_dir",   default="data/embeddings")
    args = ap.parse_args()
    main(args.clean_dir, args.emb_dir)
