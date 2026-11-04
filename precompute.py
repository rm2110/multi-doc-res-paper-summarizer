"""Run once: builds the pipeline inputs and generates v3 summaries for the first N test examples,
so those examples open instantly in the demo. All other test examples are generated live in the app.
Usage: python precompute.py [number_of_test_examples]   (default 50)"""
import json, sys, time
import torch
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from transformers import BartTokenizerFast, BartForConditionalGeneration
from pipeline import ensure_nltk, build_v2, generate, V2_DECODING, V3_PATH

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
device = "cuda" if torch.cuda.is_available() else "cpu"
ensure_nltk()
print(f"Device: {device} | examples: {N}")

test = json.load(open("processed_test.json", encoding="utf-8"))[:N]
sbert = SentenceTransformer("all-MiniLM-L6-v2", device=device)
tok = BartTokenizerFast.from_pretrained(V3_PATH)
model = BartForConditionalGeneration.from_pretrained(V3_PATH).to(device).eval()
if device == "cuda":
    model = model.half()
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

records, t0 = {}, time.time()
for i, ex in enumerate(test):
    text, tagged, n_avail = build_v2(ex["target_abstract"], ex["cited_abstracts"], sbert)
    out = generate(model, tok, text, 768, device, **V2_DECODING)
    s = scorer.score(ex["gold_summary"], out)
    records[str(i)] = {"output": out, "selected": tagged, "n_available": n_avail,
                       "rouge": {k: round(s[k].fmeasure, 4) for k in ["rouge1", "rouge2", "rougeL"]}}
    if (i + 1) % 10 == 0:
        print(f"{i + 1}/{N} | {(time.time() - t0) / 60:.1f} min", flush=True)

json.dump(records, open("demo_data.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("Saved demo_data.json")
