"""Streamlit demo: related-work generation with the v3 model (BART-large + target conditioning + MMR + role tags)."""
import html
import json
import os
import time

import streamlit as st

from pipeline import ROLES, V2_DECODING, V3_PATH, build_v2, ensure_nltk, generate

st.set_page_config(page_title="Multi-Document Abstractive Summarizer", layout="wide", initial_sidebar_state="collapsed")

TEST_FILE = "processed_test.json"
DEMO_FILE = "demo_data.json"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
:root {--ink:#0a0a0a; --mute:#6e6e6e; --line:#0a0a0a; --soft:#e4e4e4; --paper:#ffffff;
       --mono:'JetBrains Mono', ui-monospace, 'Cascadia Mono', Consolas, monospace;
       --sans:'Space Grotesk', 'Segoe UI', system-ui, sans-serif;}
html, body, [data-testid="stAppViewContainer"], .stApp {background: var(--paper); color: var(--ink); font-family: var(--sans);}
header[data-testid="stHeader"], [data-testid="stToolbar"], footer, #MainMenu,
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {display: none !important;}
.block-container {padding-top: 2.6rem; padding-bottom: 4rem; max-width: 1320px;}
p, li, label, .stMarkdown {font-family: var(--sans);}
h1, h2, h3 {font-family: var(--sans) !important; font-weight: 500 !important; letter-spacing: -0.02em; color: var(--ink);}

/* header */
.hero-tag {font-family: var(--mono); font-size: .72rem; letter-spacing: .22em; text-transform: uppercase; color: var(--mute);}
.hero-tag::before {content: ""; display: inline-block; width: 8px; height: 8px; background: var(--ink); margin-right: 10px;
                   vertical-align: 1px; animation: blink 1.6s steps(1) infinite;}
@keyframes blink {50% {opacity: 0;}}
.hero-title {font-family: var(--sans); font-size: 2.7rem; font-weight: 500; letter-spacing: -0.035em; line-height: 1.08;
             margin: 14px 0 8px 0; color: var(--ink);}
.hero-sub {font-family: var(--mono); font-size: .78rem; color: var(--mute); letter-spacing: .04em;}
.rule {height: 1px; background: var(--line); margin: 26px 0 8px 0;}

/* section labels */
.sec {font-family: var(--mono); font-size: .74rem; letter-spacing: .2em; text-transform: uppercase; color: var(--ink);
      border-bottom: 1px solid var(--line); padding-bottom: 8px; margin: 22px 0 16px 0;}
.sec span {color: var(--mute); margin-right: 12px;}

/* cards */
.card {border: 1px solid var(--line); padding: 18px 20px; background: var(--paper); line-height: 1.68; margin-bottom: 14px;
       font-size: .98rem;}
.card.accent {background: var(--ink); color: #f5f5f5; border-color: var(--ink);}
.card-title {font-family: var(--mono); font-size: .68rem; letter-spacing: .18em; text-transform: uppercase; color: var(--mute);
             margin-bottom: 10px;}
.card.accent .card-title {color: #9a9a9a;}
.note {border: 1px dashed var(--mute); padding: 14px 18px; font-family: var(--mono); font-size: .8rem; color: var(--mute);
       margin-bottom: 14px;}

/* role chips: original role colours */
.chip {display: inline-block; min-width: 92px; text-align: center; padding: 2px 8px; font-family: var(--mono); font-size: .64rem;
       letter-spacing: .14em; text-transform: uppercase; margin-right: 12px; vertical-align: 1px; border: 1px solid var(--ink);}
.chip.background {background: #2a78d6; color: #fff !important; border-color: #2a78d6;}
.chip.method {background: #eb6834; color: #fff !important; border-color: #eb6834;}
.chip.result {background: #1baf7a; color: #fff !important; border-color: #1baf7a;}
.sent {padding: 10px 0; border-bottom: 1px solid var(--soft); font-size: .95rem; line-height: 1.6;}
.sent:last-child {border-bottom: none;}

/* stats */
.stats {display: grid; grid-auto-flow: column; grid-auto-columns: 1fr; border: 1px solid var(--line); margin-bottom: 14px;}
.stat {padding: 12px 16px; border-right: 1px solid var(--line);}
.stat:last-child {border-right: none;}
.stat-label {font-family: var(--mono); font-size: .64rem; letter-spacing: .18em; text-transform: uppercase; color: var(--mute);}
.stat-value {font-family: var(--mono); font-size: 1.6rem; font-weight: 500; margin-top: 2px;}

/* widgets */
.stButton > button {border-radius: 0 !important; border: 1px solid var(--ink) !important; background: var(--paper) !important;
                    color: var(--ink) !important; font-family: var(--mono) !important; font-size: .74rem !important;
                    letter-spacing: .16em; text-transform: uppercase; padding: .55rem 1.2rem !important; transition: all .15s;}
.stButton > button:hover {background: var(--ink) !important; color: #fff !important;}
.stButton > button[kind="primary"] {background: var(--ink) !important; color: #fff !important;}
.stButton > button[kind="primary"]:hover {background: #fff !important; color: var(--ink) !important;}
.stButton > button p {font-family: var(--mono) !important;}
[data-baseweb="tab-list"] {gap: 28px; border-bottom: 1px solid var(--line);}
[data-baseweb="tab"] {font-family: var(--mono) !important; font-size: .74rem !important; letter-spacing: .18em;
                      text-transform: uppercase; padding: 10px 0 !important; background: transparent !important;}
[data-baseweb="tab"] p {font-family: var(--mono) !important; font-size: .74rem !important;}
[data-baseweb="tab-highlight"] {background: var(--ink) !important; height: 2px;}
[data-baseweb="tab-border"] {display: none;}
[data-baseweb="select"] > div, .stTextArea textarea, [data-baseweb="textarea"] {border-radius: 0 !important;}
[data-baseweb="select"] > div {border: 1px solid var(--ink) !important; background: var(--paper) !important;
                               font-family: var(--mono); font-size: .82rem;}
.stTextArea textarea {border: 1px solid var(--ink) !important; background: var(--paper) !important; font-family: var(--sans);}
[data-baseweb="textarea"] {border: none !important;}
.stSelectbox label p, .stTextArea label p {font-family: var(--mono) !important; font-size: .68rem !important;
                                          letter-spacing: .18em; text-transform: uppercase; color: var(--mute);}
[data-testid="stExpander"] details {border-radius: 0 !important; border: 1px solid var(--soft) !important;}
[data-testid="stExpander"] summary p {font-family: var(--sans); font-size: .9rem;}
[data-testid="stCaptionContainer"], .stCaption {font-family: var(--mono) !important; font-size: .74rem !important; color: var(--mute) !important;}
.cited-head {font-family: var(--mono); font-size: .68rem; letter-spacing: .18em; text-transform: uppercase; color: var(--mute);
             margin: 18px 0 8px 0;}
[role="tab"] p, [role="tab"] div {font-family: var(--mono) !important; font-size: .72rem !important;
                                   letter-spacing: .18em; text-transform: uppercase;}
.stSelectbox [role="group"], .stSelectbox [data-baseweb="select"] > div {background: #fff !important; border: 1px solid var(--ink) !important;
                                   border-radius: 0 !important;}
.stSelectbox input, .stSelectbox [data-baseweb="select"] {font-family: var(--mono) !important; font-size: .8rem !important;
                                   background: transparent !important;}
/* force dark text everywhere, even if the browser or Streamlit is in dark mode */
.stApp, .stApp p, .stApp label, .stApp li, [data-testid="stMarkdownContainer"] {color: var(--ink);}
.card.accent, .card.accent * {color: #f5f5f5;}
.card.accent .card-title {color: #9a9a9a !important;}
.stSelectbox label p, .stTextArea label p {color: var(--ink) !important;}
[role="tab"], [role="tab"] p, [data-baseweb="tab"], [data-baseweb="tab"] p {color: #8a8a8a !important;}
[role="tab"][aria-selected="true"], [role="tab"][aria-selected="true"] p,
[data-baseweb="tab"][aria-selected="true"], [data-baseweb="tab"][aria-selected="true"] p {color: var(--ink) !important;}
[role="tab"]:hover p, [data-baseweb="tab"]:hover p {color: var(--ink) !important;}
[data-baseweb="tab-highlight"], .react-aria-SelectionIndicator {background-color: var(--ink) !important;}
.stSelectbox input, .stSelectbox [data-baseweb="select"] div, .stSelectbox [data-baseweb="select"] span {
    color: var(--ink) !important; -webkit-text-fill-color: var(--ink) !important;}
.stSelectbox svg {fill: var(--ink) !important; color: var(--ink) !important;}
.stTextArea textarea {color: var(--ink) !important; -webkit-text-fill-color: var(--ink) !important;}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p, [data-testid="stExpander"] svg {color: var(--ink) !important;}
[data-testid="stExpander"] details {background: var(--paper) !important;}
[data-testid="stSpinner"], [data-testid="stSpinner"] p {color: var(--ink) !important; background: transparent !important;}
/* button text colours (must come after the global dark-text rule) */
.stButton > button p, .stButton > button div {color: var(--ink) !important;}
.stButton > button:hover p, .stButton > button:hover div {color: #fff !important;}
.stButton > button[kind="primary"] p, .stButton > button[kind="primary"] div,
[data-testid="stBaseButton-primary"] p, [data-testid="stBaseButton-primary"] div {color: #fff !important;}
.stButton > button[kind="primary"]:hover p, .stButton > button[kind="primary"]:hover div,
[data-testid="stBaseButton-primary"]:hover p, [data-testid="stBaseButton-primary"]:hover div {color: var(--ink) !important;}
/* cited papers (custom collapsible rows) */
details.cite {border: 1px solid var(--soft); background: var(--paper); margin-bottom: 8px;}
details.cite summary {list-style: none; cursor: pointer; padding: 11px 14px; font-size: .92rem; color: var(--ink);
                      background: var(--paper); display: flex; align-items: baseline; gap: 12px; transition: background .12s;}
details.cite summary::-webkit-details-marker {display: none;}
details.cite summary::after {content: "+"; margin-left: auto; font-family: var(--mono); color: var(--mute);}
details.cite[open] summary::after {content: "-";}
details.cite summary:hover {background: #f3f3f3;}
details.cite[open] summary {border-bottom: 1px solid var(--soft);}
.cite-num {font-family: var(--mono); font-size: .7rem; color: var(--mute); letter-spacing: .1em;}
.cite-body {padding: 14px 16px; line-height: 1.65; font-size: .95rem; color: var(--ink);}
</style>
""", unsafe_allow_html=True)


# ---------------- data and model loading ----------------
@st.cache_data
def load_test():
    with open(TEST_FILE, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_precomputed():
    if not os.path.exists(DEMO_FILE):
        return {}
    with open(DEMO_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}   # ignore files from the older app version


@st.cache_resource(show_spinner=False)
def load_v3():
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import BartForConditionalGeneration, BartTokenizerFast
    ensure_nltk()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = BartTokenizerFast.from_pretrained(V3_PATH)
    model = BartForConditionalGeneration.from_pretrained(V3_PATH).to(device).eval()
    if device == "cuda":
        model = model.half()
    sbert = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    return tok, model, sbert, device


@st.cache_resource
def load_scorer():
    from rouge_score import rouge_scorer
    return rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def run_v3(target, cited, reference=None):
    """Full pipeline + v3 generation. Returns a result dict in the same format as demo_data.json."""
    tok, model, sbert, device = load_v3()
    t0 = time.time()
    text, tagged, n_avail = build_v2(target, cited, sbert)
    out = generate(model, tok, text, 768, device, **V2_DECODING)
    res = {"output": out, "selected": tagged, "n_available": n_avail, "seconds": time.time() - t0}
    if reference:
        s = load_scorer().score(reference, out)
        res["rouge"] = {k: round(s[k].fmeasure, 4) for k in ["rouge1", "rouge2", "rougeL"]}
    return res


# ---------------- small UI helpers ----------------
def esc(text):
    return html.escape(text).replace("\n", "<br>")


def md(markup):
    st.markdown(markup, unsafe_allow_html=True)


def card(title, text, accent=False):
    cls = "card accent" if accent else "card"
    md(f'<div class="{cls}"><div class="card-title">{esc(title)}</div>{esc(text)}</div>')


def section(num, title):
    md(f'<div class="sec"><span>{num}</span>{esc(title)}</div>')


def stats(items):
    cells = "".join(f'<div class="stat"><div class="stat-label">{esc(k)}</div><div class="stat-value">{esc(str(v))}</div></div>'
                    for k, v in items)
    md(f'<div class="stats">{cells}</div>')


def chip(role):
    return f'<span class="chip {role}">{role}</span>'


def cited_paper(k, abstract):
    """Collapsible cited abstract; hovering the row shows the paper's opening sentence as a tooltip."""
    first = abstract.split(". ")[0].strip()
    first = first if first.endswith(".") else first + "."
    md(f'<details class="cite"><summary title="{html.escape(first, quote=True)}">'
       f'<span class="cite-num">{k:02d}</span>{esc(short(abstract, 9))}</summary>'
       f'<div class="cite-body">{esc(abstract)}</div></details>')


def short(text, n=11):
    words = text.split()
    return " ".join(words[:n]) + ("..." if len(words) > n else "")


def render_pipeline(tagged, n_available):
    counts = {r: sum(1 for _, role in tagged if role == r) for r in ROLES}
    stats([("Cited sentences", n_available)] + [(r, counts[r]) for r in ROLES])
    rows = "".join(f'<div class="sent">{chip(role)}{esc(s)}</div>' for s, role in tagged)
    md(f'<div class="card">{rows}</div>')
    st.caption("The sentences most relevant to the target paper are selected from the cited abstracts, "
               "labelled by their rhetorical role and grouped under a tag before being passed to the model.")


def render_rouge(scores):
    stats([("ROUGE-1", f"{scores['rouge1']:.3f}"), ("ROUGE-2", f"{scores['rouge2']:.3f}"),
           ("ROUGE-L", f"{scores['rougeL']:.3f}")])


# ---------------- header ----------------
md(
   '<div class="hero-title">Multi-Document Abstractive Summarizer<br>for Research Papers</div>'
   '<div class="rule"></div>')

tab_explore, tab_custom = st.tabs(["Explore test papers", "Try your own papers"])

# ---------------- tab 1: all test examples ----------------
with tab_explore:
    if not os.path.exists(TEST_FILE):
        st.error(f"{TEST_FILE} not found. Put the app files in the project folder.")
        st.stop()
    test = load_test()
    pre = load_precomputed()
    st.caption(f"{len(test):,} test papers available.")
    labels = [f"Test example {i}{' (ready)' if str(i) in pre else ''} - {short(ex['target_abstract'])}"
              for i, ex in enumerate(test)]
    idx = st.selectbox("Choose a target paper", range(len(test)), format_func=lambda i: labels[i])
    ex = test[idx]

    result = pre.get(str(idx))
    live = st.session_state.get(f"live_{idx}")
    if live:
        result = live

    col_in, col_out = st.columns(2, gap="large")
    with col_in:
        section("01", "Input papers")
        card("Target paper abstract", ex["target_abstract"])
        md(f'<div class="cited-head">Cited papers ({len(ex["cited_abstracts"])})</div>')
        for k, a in enumerate(ex["cited_abstracts"], 1):
            cited_paper(k, a)

    with col_out:
        section("02", "Generated related work")
        if result:
            title = "v3 output" + (f" / generated live in {result['seconds']:.1f} s" if "seconds" in result else "")
            card(title, result["output"], accent=True)
            render_rouge(result["rouge"])
            label = "Generate again"
        else:
            md('<div class="note">This paper has not been generated yet. Run the model to generate it.</div>')
            label = "Generate summary"
        if st.button(label, key=f"btn_{idx}", type="primary" if not result else "secondary"):
            with st.spinner("Selecting sentences and generating (the first run also loads the model)..."):
                st.session_state[f"live_{idx}"] = run_v3(ex["target_abstract"], ex["cited_abstracts"],
                                                         ex["gold_summary"])
            st.rerun()
        st.write("")
        card("Reference (written by the paper's authors)", ex["gold_summary"])

    if result:
        section("03", "Inside the pipeline")
        render_pipeline([tuple(x) for x in result["selected"]], result["n_available"])

# ---------------- tab 2: custom input ----------------
with tab_custom:
    st.caption("Paste the abstract of a target paper and the abstracts of the papers it cites. "
               "Separate cited abstracts with a blank line.")
    target = st.text_area("Target paper abstract", height=150)
    cited_raw = st.text_area("Cited paper abstracts (one per paragraph)", height=250)
    if st.button("Generate related work", type="primary"):
        cited = [c.strip() for c in cited_raw.split("\n\n") if c.strip()]
        if not target.strip() or not cited:
            st.error("Please enter a target abstract and at least one cited abstract.")
        else:
            with st.spinner("Selecting sentences and generating..."):
                res = run_v3(target.strip(), cited)
            section("01", "Generated related work")
            card(f"v3 output / generated in {res['seconds']:.1f} s", res["output"], accent=True)
            section("02", "Inside the pipeline")
            render_pipeline(res["selected"], res["n_available"])
