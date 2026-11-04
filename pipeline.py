"""Shared pipeline code: role classifier, MMR selection and input builders (same as the report)."""
from nltk.tokenize import sent_tokenize

BG = ["problem", "challenge", "existing", "prior work", "however", "limitation", "propose", "address",
      "motivated", "prior", "previous", "traditionally", "remains", "lack of", "difficult", "issue"]
ME = ["method", "approach", "algorithm", "model", "framework", "technique", "using", "based on",
      "we present", "we introduce", "we develop", "architecture", "training", "trained", "design"]
RE = ["show", "results", "achieve", "outperform", "accuracy", "evaluate", "experiment", "demonstrate",
      "improve", "performance", "state-of-the-art", "compared to", "significantly", "we find"]
ROLES = ["background", "method", "result"]
V2_DECODING = {"no_repeat_ngram_size": 3, "length_penalty": 1.2}
V3_PATH = "./saved_models/bart_large_final_model"


def ensure_nltk():
    import nltk
    for pkg in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


def classify(sentence):
    s = sentence.lower()
    sc = {"background": sum(c in s for c in BG), "method": sum(c in s for c in ME), "result": sum(c in s for c in RE)}
    best = max(sc, key=sc.get)
    return "background" if sc[best] == 0 else best


def mmr(cited, target, sbert, k=10, lam=0.7):
    """Returns (selected sentences in original order, number of sentences available)."""
    from sentence_transformers import util
    sents = [s for a in cited for s in sent_tokenize(a)]
    if len(sents) <= k:
        return sents, len(sents)
    q = sbert.encode(target, convert_to_tensor=True)
    e = sbert.encode(sents, convert_to_tensor=True)
    rel = util.cos_sim(e, q).squeeze(1).cpu().numpy()
    sim = util.cos_sim(e, e).cpu().numpy()
    sel, rem = [], list(range(len(sents)))
    while len(sel) < k and rem:
        best = max(rem, key=lambda i: rel[i] if not sel else lam * rel[i] - (1 - lam) * sim[i, sel].max())
        sel.append(best)
        rem.remove(best)
    return [sents[i] for i in sorted(sel)], len(sents)


def build_v2(target_abstract, cited_abstracts, sbert):
    """v2/v3 input. Returns (input text, list of (sentence, role) in input order, sentences available)."""
    selected, n_avail = mmr(cited_abstracts, target_abstract, sbert)
    groups = {r: [] for r in ROLES}
    for s in selected:
        groups[classify(s)].append(s)
    text = " ".join([f"<target> {target_abstract}"] + [f"<{r}> " + " ".join(groups[r]) for r in ROLES if groups[r]])
    tagged = [(s, r) for r in ROLES for s in groups[r]]
    return text, tagged, n_avail


def build_baseline(cited_abstracts):
    return " </s> ".join(cited_abstracts)


def generate(model, tokenizer, text, max_in, device, **decoding):
    import torch
    enc = tokenizer(text, return_tensors="pt", max_length=max_in, truncation=True).to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_length=128, num_beams=4, early_stopping=True, **decoding)
    return tokenizer.decode(out[0], skip_special_tokens=True)
