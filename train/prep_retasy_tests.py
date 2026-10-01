"""RetaSy (CC-BY-4.0; internal testing only): labelled Quran recordings as 16 kHz WAVs,
each matched to its surah:ayah by text, plus manifest.json."""
import glob, io, json, pathlib, re, difflib
import numpy as np, pyarrow.parquet as pq, soundfile as sf, librosa
from quran_transcript import Aya

here = pathlib.Path(__file__).parent
out = here / "wav"; out.mkdir(exist_ok=True)
SURAH = {"Al-Faatihah": 1, "Al-Ikhlas": 112, "An-Nas": 114, "Al-Kafiroon": 109, "Ayat al-Kursi": 2, "Al-Falaq": 113,
         "Al-Asr": 103, "Al-Kauthar": 108, "Al-Masad": 111, "Al-Humazah": 104, "An-Nasr": 110, "Quraish": 106,
         "Al-Maun": 107, "Al-Fil": 105, "At-Takathur": 102, "Al-Qariah": 101}
letters = lambda s: re.sub(r"[^ء-غف-يٱ]", "", s.replace("ٱ", "ا"))
texts = {}
def ayahs(s):
    if s not in texts:
        n = 7 if s == 1 else 286 if s == 2 else 30
        texts[s] = []
        for a in range(1, n + 1):
            try: texts[s].append((a, letters(Aya(s, a).get().uthmani)))
            except Exception: break
    return texts[s]

t = pq.read_table(glob.glob(str(here / "raw/data/*.parquet"))).to_pandas()
t = t[t.final_label.isin(["correct", "in_correct"])]
manifest, unmatched = [], 0
for i, r in t.iterrows():
    s = SURAH.get(r["Surah"])
    if not s: continue
    want = letters(r["Aya"])
    best = max(ayahs(s), key=lambda x: difflib.SequenceMatcher(None, want, x[1]).ratio())
    if difflib.SequenceMatcher(None, want, best[1]).ratio() < 0.85:
        unmatched += 1; continue
    if s == 2 and best[0] != 255: continue
    wave, rate = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
    if wave.ndim > 1: wave = wave.mean(axis=1)
    if rate != 16000: wave = librosa.resample(wave, orig_sr=rate, target_sr=16000)
    if len(wave) < 8000: continue
    name = f"r{i}.wav"
    sf.write(out / name, wave, 16000, subtype="PCM_16")
    manifest.append({"file": name, "key": f"{s}:{best[0]}", "label": r["final_label"], "reciter": r["reciter_id"],
                     "seconds": round(len(wave) / 16000, 1)})
(here / "manifest.json").write_text(json.dumps(manifest, indent=0))
print(len(manifest), "recordings;", unmatched, "unmatched;", {l: sum(m["label"] == l for m in manifest) for l in ("correct", "in_correct")})
