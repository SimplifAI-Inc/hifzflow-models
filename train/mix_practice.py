"""Checker v3's training mix: v2's mix (professional segments with planted
mistakes and harder copies, plus learner recordings six times) and 20,000
practice situations (prep_practice.py: repeats, going back, pauses, a voice
behind, a similar passage, a cough), each labelled with the words actually
recited, repetitions kept.

  python mix_practice.py <full/train.jsonl> <practice_train.jsonl> <mix_learners.jsonl> <out.jsonl>
"""
import json, os, random, sys

full, practice, learners, out = sys.argv[1:5]
text = {}
for line in open(full, encoding="utf8"):
    r = json.loads(line)
    if r["kind"] == "correct":
        text[os.path.basename(r["audio_filepath"])[:-4]] = r["text"]
clips = []
for line in open(practice, encoding="utf8"):
    r = json.loads(line)
    parts = [text.get(p) for p in r["parts"]]
    if all(parts):
        clips.append({"audio_filepath": r["path"], "duration": r["duration"], "text": " ".join(parts), "kind": "practice-" + r["kind"]})
random.Random(7).shuffle(clips)
mix = [json.loads(line) for line in open(learners, encoding="utf8")] + clips[:20000]
random.Random(8).shuffle(mix)
with open(out, "w", encoding="utf8") as fh:
    for m in mix:
        fh.write(json.dumps(m, ensure_ascii=False) + "\n")
print(len(mix), "clips")
