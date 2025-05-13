wav = load(file).resample(16000).to_mono()
speech_segments = rVAD(wav)              # unsupervised, energy‑based
clean = noise_reduce(speech_segments)    # optional
norm  = loudness_normalise(clean)
save(norm)