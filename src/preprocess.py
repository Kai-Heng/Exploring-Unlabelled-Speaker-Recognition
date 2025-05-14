"""Pre‑processing: resample ➜ VAD ➜ loudness normalise ➜ save cleaned WAV."""
import os, argparse, math, wave, contextlib, soundfile as sf, numpy as np
import librosa, webrtcvad
from tqdm import tqdm

TARGET_SR = 16_000
FRAME_MS  = 20        # frame size for VAD
VAD_MODE  = 2         # 0‑3, higher = more aggressive silence removal

def read_wav(path, sr=TARGET_SR):
    wav, native_sr = sf.read(path)
    if wav.ndim > 1:                       # stereo → mono
        wav = np.mean(wav, axis=1)
    if native_sr != sr:
        wav = librosa.resample(wav, orig_sr=native_sr, target_sr=sr)
    return wav.astype(np.float32)

def apply_vad(wav, sr=TARGET_SR):
    vad = webrtcvad.Vad(VAD_MODE)
    frame_len = int(sr * FRAME_MS / 1000)
    bytes_per_sample = 2
    def frame_generator():
        for i in range(0, len(wav), frame_len):
            chunk = wav[i:i+frame_len]
            if len(chunk) < frame_len:
                pad = np.zeros(frame_len - len(chunk))
                chunk = np.concatenate([chunk, pad])
            pcm = (chunk * 32768).astype(np.int16).tobytes()
            yield pcm, chunk
    voiced = []
    for pcm, chunk in frame_generator():
        if vad.is_speech(pcm, sr):
            voiced.append(chunk)
    return np.concatenate(voiced) if voiced else wav  # fallback: return original

def loudness_normalise(wav):
    peak = np.max(np.abs(wav))
    if peak < 1e-3:        # silent file
        return wav
    return wav / peak * 0.95

def main(raw_dir, clean_dir):
    os.makedirs(clean_dir, exist_ok=True)
    files = sorted(p for p in os.listdir(raw_dir) if p.lower().endswith(".wav"))
    for fname in tqdm(files, desc="Pre‑processing"):
        in_path  = os.path.join(raw_dir, fname)
        out_path = os.path.join(clean_dir, fname)
        wav = read_wav(in_path)
        wav = apply_vad(wav)
        wav = loudness_normalise(wav)
        sf.write(out_path, wav, TARGET_SR)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir",   default="data/raw")
    ap.add_argument("--clean_dir", default="data/clean")
    args = ap.parse_args()
    main(args.raw_dir, args.clean_dir)