"""Keeps the unlabelled learner recordings that NVIDIA's original model heard word for word
(as the app compares words), labelled with the ayah's text; writes a training mix:
those recordings several times over, plus a sample of the professional training data so
the model keeps what it learned there.

  python filter_learners.py <unlabelled.jsonl> <full train.jsonl> <out mix.jsonl> [repeat] [professional sample]
"""
import json, random, re, sys
import torch, soundfile as sf
import nemo.collections.asr as nemo_asr

src, pro, out = sys.argv[1], sys.argv[2], sys.argv[3]
REPEAT = int(sys.argv[4]) if len(sys.argv) > 4 else 6
PRO = int(sys.argv[5]) if len(sys.argv) > 5 else 40000
random.seed(3)
m = nemo_asr.models.ASRModel.from_pretrained("nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0", map_location="cuda")
m.eval()
tok = m.tokenizer
BLANK = m.ctc_decoder.num_classes_with_blank - 1
TASHKEEL = re.compile("[ً-ٰٟۖ-ۭـ]")


def plain(s):
    s = TASHKEEL.sub("", s)
    for a, b in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ٱ", "ا"), ("ى", "ي"), ("ة", "ه"), ("ؤ", "و"), ("ئ", "ي")):
        s = s.replace(a, b)
    return re.sub(r"(.)\1+", r"\1", re.sub(r"[^ء-ي ]", " ", s)).split()


def edits(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(b)]


rows = [json.loads(line) for line in open(src, encoding="utf8")]
kept = []
with torch.no_grad():
    for r in rows:
        w, _ = sf.read(r["audio_filepath"], dtype="float32")
        audio = torch.from_numpy(w)[None].cuda()
        sig, sl = m.preprocessor(input_signal=audio, length=torch.tensor([w.size]).cuda())
        enc, _ = m.encoder(audio_signal=sig, length=sl)
        best = m.ctc_decoder(encoder_output=enc).argmax(-1)[0].cpu().numpy()
        ids = [int(t) for j, t in enumerate(best) if t != BLANK and (j == 0 or t != best[j - 1])]
        heard = plain(tok.ids_to_text(ids) if ids else "")
        expected = plain(r["text"])
        # Word for word: same words in order, each within one letter.
        if len(heard) == len(expected) and all(edits(list(e), list(h)) <= 1 for e, h in zip(expected, heard)):
            kept.append({**r, "text": r["text"].replace("ٰ", ""), "kind": "learner"})
print(f"kept {len(kept)} of {len(rows)} learner recordings ({sum(r['duration'] for r in kept) / 3600:.2f} h), {len({r['reciter'] for r in kept})} learners", flush=True)
pros = [json.loads(line) for line in open(pro, encoding="utf8")]
sample = random.sample(pros, min(PRO, len(pros)))
mix = kept * REPEAT + sample
random.shuffle(mix)
with open(out, "w", encoding="utf8") as fh:
    for r in mix:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"mix: {len(kept)} x {REPEAT} learner + {len(sample)} professional = {len(mix)} clips")
