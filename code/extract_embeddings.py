from speechbrain.pretrained import EncoderClassifier
encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
embeddings = encoder.encode_batch(wav_tensor)  # → 192‑D