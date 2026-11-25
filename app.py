"""BILINGO: Arabic / English customer feedback sentiment (Streamlit demo).

Run:  streamlit run app.py
"""
import json
from html import escape
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.service import LABELS, SentimentService

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"

st.set_page_config(page_title="BILINGO · Feedback Sentiment", page_icon="◉", layout="wide")

# palette: navy #2F4156 · pale pink #FFE1E6 · azalea #F7C9D4 · sky blue #C8D9E6 · beige #F5EFEB
INDIGO, INDIGO_L, INK, MUTED = "#2F4156", "#6F8AA6", "#2F4156", "#6B7686"
SKY, PINK, AZALEA, BEIGE = "#C8D9E6", "#FFE1E6", "#F7C9D4", "#F5EFEB"
SENT = {"negative": "#D9738C", "mixed": "#8EA8C0", "positive": "#2F4156"}
MODEL_COLORS = {"Pretrained, no fine-tuning": "#C8D9E6", "TF-IDF + Logistic Regression": "#F0A9BA",
                "Fine-tuned XLM-RoBERTa": INDIGO}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Noto+Sans+Arabic:wght@400;600;700&display=swap');
.stApp, .stApp p, .stApp label, .stApp textarea, .stMarkdown {{ font-family:'Plus Jakarta Sans','Noto Sans Arabic',sans-serif; }}
.stApp {{ background: radial-gradient(900px 380px at 0% -10%, rgba(200,217,230,.55), transparent 60%),
                      radial-gradient(700px 300px at 100% 0%, rgba(247,201,212,.45), transparent 60%), #F5EFEB; }}
[data-testid="stSidebar"] {{ background:#C8D9E6; border-right:1px solid #B5C9D8; }}
[data-testid="stHeader"] {{ background:transparent; }}
.block-container {{ padding-top:1.4rem; max-width:1300px; }}
[data-tag] {{ background:{INDIGO} !important; }}
[data-tag], [data-tag] * {{ color:#fff !important; fill:#fff !important; }}

.hero {{ display:flex; align-items:center; gap:1.4rem; background:#fff; border-radius:18px; padding:1.1rem 1.4rem;
         box-shadow:0 8px 30px rgba(30,42,58,.08); margin-bottom:1rem; border:1px solid #EADFE0; }}
.hero svg {{ flex-shrink:0; }}
.hero h1 {{ font-size:2.1rem; margin:0; color:#2F4156 !important; letter-spacing:.2em; font-weight:800; padding:0; }}
.wm {{ font-size:2.1rem; font-weight:800; letter-spacing:.2em; line-height:1.1; }}
.wm .a {{ color:#2F4156; }} .wm .b {{ color:#D9738C; }}
.hero .ar {{ font-family:'Noto Sans Arabic'; color:{INDIGO}; font-size:1.1rem; margin-left:.5rem; letter-spacing:0; }}
.hero p {{ margin:.2rem 0 0; color:{MUTED}; font-size:.92rem; }}
.mv {{ animation:mv 2.4s ease-in-out infinite; }}
@keyframes mv {{ 0%,100%{{transform:translateX(-8px)}} 50%{{transform:translateX(8px)}} }}
.stTextArea textarea {{ background:#FFFFFF !important; border:1px solid #D9CFCB !important; }}

.h {{ font-size:1.15rem; font-weight:700; margin:.4rem 0 .6rem; color:{INK}; }}
.h small {{ color:{MUTED}; font-weight:400; font-size:.82rem; margin-left:.4rem; }}
.note {{ color:{MUTED}; font-size:.88rem; line-height:1.55; }}
.note b {{ color:{INK}; }}
.card {{ background:#fff; border-radius:16px; padding:1rem 1.15rem; box-shadow:0 6px 22px rgba(30,42,58,.07);
         border:1px solid #EADFE0; height:100%; }}
.card .model {{ font-size:.7rem; letter-spacing:.14em; text-transform:uppercase; color:{MUTED}; font-weight:700; }}
.card .label {{ font-size:1.7rem; font-weight:800; margin:.2rem 0 .1rem; text-transform:capitalize; }}
.card .conf {{ color:{MUTED}; font-size:.85rem; margin-bottom:.6rem; }}
.row {{ display:grid; grid-template-columns:72px 1fr 44px; gap:.5rem; align-items:center; font-size:.8rem; color:{INK}; margin:.25rem 0; }}
.track {{ height:9px; background:#EFE6E1; border-radius:99px; overflow:hidden; }}
.track > div {{ height:100%; border-radius:99px; }}
.chip {{ display:inline-block; padding:.18rem .6rem; border-radius:99px; background:#FFE1E6; color:{INDIGO};
         font-size:.78rem; margin:.15rem .2rem 0 0; font-weight:600; }}
.pill {{ display:inline-block; padding:.12rem .55rem; border-radius:99px; color:#fff; font-size:.72rem; font-weight:700;
         text-transform:uppercase; letter-spacing:.06em; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(190px, 1fr)); gap:.7rem; margin:.3rem 0 1rem; }}
.kpi {{ background:#fff; border-radius:14px; padding:.85rem 1rem; border:1px solid #EADFE0; box-shadow:0 4px 16px rgba(30,42,58,.06); }}
.kpi .l {{ color:{MUTED}; font-size:.68rem; letter-spacing:.12em; text-transform:uppercase; font-weight:700; }}
.kpi .v {{ font-size:1.6rem; font-weight:800; color:{INK}; margin-top:.1rem; }}
.kpi .d {{ color:{MUTED}; font-size:.78rem; }}
.kpi.hi {{ background:linear-gradient(135deg, #2F4156, #4E6A87); border:none; }}
.kpi.hi .l, .kpi.hi .d {{ color:#FFE1E6; }} .kpi.hi .v {{ color:#fff; }}
.ex {{ background:#fff; border-radius:12px; padding:.7rem .9rem; margin-bottom:.5rem; border:1px solid #EADFE0; }}
.ex .t {{ font-size:.92rem; color:{INK}; line-height:1.6; }}
.ex .m {{ font-size:.75rem; color:{MUTED}; margin-top:.35rem; }}
.flow {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:.6rem; }}
.step {{ background:#fff; border-radius:12px; padding:.85rem; font-size:.85rem; color:{MUTED}; border-top:4px solid {INDIGO_L}; }}
.step b {{ display:block; color:{INK}; margin-bottom:.25rem; }}
</style>
""", unsafe_allow_html=True)

HERO = f"""
<div class="hero">
  <svg width="190" height="101" viewBox="0 0 150 80">
    <path d="M22 62 Q75 6 128 62" fill="none" stroke="#2F4156" stroke-width="5" stroke-linecap="round"/>
    <rect x="8" y="52" width="34" height="20" rx="7" fill="#C8D9E6"/>
    <text x="25" y="66" text-anchor="middle" font-size="10" font-weight="700" fill="#2F4156">EN</text>
    <rect x="108" y="52" width="34" height="20" rx="7" fill="#F7C9D4"/>
    <text x="125" y="67" text-anchor="middle" font-size="12" font-weight="700" fill="#2F4156" font-family="Noto Sans Arabic">ع</text>
    <circle r="6" fill="#D9738C"><animateMotion dur="3.2s" repeatCount="indefinite" path="M22 62 Q75 6 128 62"
      keyPoints="0.12;0.88;0.12" keyTimes="0;0.5;1" calcMode="spline" keySplines=".45 0 .55 1;.45 0 .55 1"/></circle>
  </svg>
  <div><div class="wm"><span class="a">BI</span><span class="b">LINGO</span></div>
  <p>Two languages, one read on how your customers feel. Arabic and English feedback sorted into positive, mixed or negative.</p></div>
</div>"""


@st.cache_resource(show_spinner="Loading models ...")
def get_service():
    svc = SentimentService()
    _ = svc.baseline      # load both models once, before anyone can click
    svc.warm_up()
    return svc


@st.cache_data(show_spinner=False)
def _load(name, mtime):
    p = REPORTS / name
    return json.loads(p.read_text()) if name.endswith(".json") else pd.read_csv(p)


def load(name):
    p = REPORTS / name
    return _load(name, p.stat().st_mtime) if p.exists() else None


THEMES = {
    "Beige": dict(bg="#F5EFEB", glow1="rgba(200,217,230,.55)", glow2="rgba(247,201,212,.45)", side="#C8D9E6",
                  text="#2F4156", muted="#6B7686", grid="#C9BFC0", dark=False),
    "Sky blue": dict(bg="#C8D9E6", glow1="rgba(255,255,255,.6)", glow2="rgba(247,201,212,.55)", side="#F5EFEB",
                     text="#2F4156", muted="#4E6377", grid="#9DB2C4", dark=False),
    "Navy": dict(bg="#2F4156", glow1="rgba(200,217,230,.20)", glow2="rgba(247,201,212,.16)", side="#26364A",
                 text="#F5EFEB", muted="#C8D9E6", grid="#5D7189", dark=True),
}
T = THEMES["Beige"]


def show_chart(chart, grid=True):
    """Every chart sits on its own white card, so it reads the same on any background."""
    st.altair_chart(chart.configure(background="#FFFFFF", padding=14)
                    .configure_axis(grid=grid, gridColor="#E3D9D6", gridOpacity=1, gridDash=[3, 3], domainColor="#C9BFC0",
                                    labelColor=INK, titleColor=MUTED, labelFontSize=11)
                    .configure_view(strokeWidth=0).configure_legend(labelColor=INK, titleColor=MUTED),
                    width="stretch", theme=None)


def pill(label):
    return f'<span class="pill" style="background:{SENT[label]}">{label}</span>'


def result_card(r, title, extra=""):
    bars = "".join(f'<div class="row"><span>{l}</span><div class="track"><div style="width:{r["scores"][l]*100:.0f}%;'
                   f'background:{SENT[l]}"></div></div><span>{r["scores"][l]:.0%}</span></div>' for l in LABELS)
    return (f'<div class="card"><div class="model">{title}</div>'
            f'<div class="label" style="color:{SENT[r["label"]]}">{r["label"]}</div>'
            f'<div class="conf">{r["confidence"]:.0%} confident</div>{bars}{extra}</div>')


svc = get_service()
EXAMPLES = {
    "Hotel, Arabic, positive": "الفندق رائع جدا، الغرف نظيفة والموظفين في قمة الاحترام وسأعود مرة أخرى",
    "Hotel, Arabic, mixed": "الموقع ممتاز وقريب من كل شيء لكن الغرفة صغيرة والإفطار عادي جدا",
    "Order, Arabic dialect, negative": "الطلب وصل متأخر أسبوعين والمقاس مش مظبوط خالص، مش هطلب تاني",
    "Product, English, positive": "Absolutely love it! Great quality and it arrived two days early.",
    "Product, English, mixed": "It works fine and the price is fair, but the battery barely lasts a day.",
    "Product, English, negative": "Stopped working after a week and customer service never replied.",
    "Tricky: sarcasm": "Oh great, another charger that dies in two days. Exactly what I needed.",
}

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    T = THEMES[st.radio("Background", list(THEMES), index=2, horizontal=True, key="bg_theme")]
    st.markdown(f'<div style="font-weight:800;letter-spacing:.22em;font-size:1.2rem;color:{T["text"]}">BI<span style="color:#D9738C">LINGO</span></div>'
                f'<div style="color:{T["muted"]};font-size:.75rem;letter-spacing:.12em;text-transform:uppercase">Feedback sentiment</div>',
                unsafe_allow_html=True)
    st.caption("UneeQ Interns · ML Task: Arabic/English Sentiment Classifier")
    st.divider()
    st.markdown("**Models**")
    st.markdown(f"- **Fine-tuned XLM-RoBERTa** {'✓' if svc.has_transformer else '(not trained yet)'}\n- **TF-IDF + Logistic Regression** ✓")
    st.caption("Both were trained on the same 30K reviews: 15K Arabic (hotels, books) and 15K English (Amazon products).")
    st.divider()
    st.caption("Labels: positive · mixed · negative. 'Mixed' means the customer liked some things and not others, or gave 3 stars.")

st.markdown(f"""<style>
.stApp {{ background: radial-gradient(900px 380px at 0% -10%, {T['glow1']}, transparent 60%),
                      radial-gradient(700px 300px at 100% 0%, {T['glow2']}, transparent 60%), {T['bg']} !important; }}
[data-testid="stSidebar"] {{ background:{T['side']} !important; border-right:1px solid rgba(0,0,0,.08) !important; }}
.stApp [data-testid="stMarkdownContainer"] p, .stApp [data-testid="stMarkdownContainer"] li,
.stApp [data-testid="stMarkdownContainer"] strong, .stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stMetricValue"], .stApp [data-testid="stMetricLabel"] p,
.stTabs [data-baseweb="tab"] p, .h, .note, .note b {{ color:{T['text']} !important; }}
.h small, .stApp [data-testid="stCaptionContainer"] {{ color:{T['muted']} !important; }}
.stTabs [aria-selected="true"] p {{ font-weight:700; }}
.stTabs [data-baseweb="tab-highlight"] {{ background:{'#F7C9D4' if T['dark'] else '#2F4156'} !important; }}
.card .note, .card .note b {{ color:#6B7686 !important; }}
.stApp .hero p, .stApp [data-testid="stMarkdownContainer"] .hero p {{ color:#6B7686 !important; }}
.stApp [data-testid="stButtonGroup"] button {{ background:#FFFFFF !important; border:1px solid #E3D9D6 !important; }}
.stApp [data-testid="stButtonGroup"] button p {{ color:#2F4156 !important; }}
.stApp [data-testid="stButtonGroup"] button[aria-checked="true"], .stApp [data-testid="stButtonGroup"] button[kind*="Active"] {{
    background:#F7C9D4 !important; border-color:#D9738C !important; }}
.kpi.hi {{ {'background:linear-gradient(135deg, #D9738C, #F0A9BA) !important;' if T['dark'] else ''} }}
.kpi.hi .l, .kpi.hi .d {{ {'color:#FFF5F7 !important;' if T['dark'] else ''} }}
.stApp hr {{ border-color:{T['grid']} !important; }}
</style>""", unsafe_allow_html=True)
st.markdown(HERO, unsafe_allow_html=True)
tab_try, tab_batch, tab_cmp, tab_err, tab_how = st.tabs(
    ["Analyze feedback", "Batch scoring", "Model comparison", "Error analysis", "How it works"])

# ------------------------------------------------------------------ analyze
with tab_try:
    st.markdown('<div class="h">Try a review <small>Arabic or English, any length</small></div>', unsafe_allow_html=True)
    pick = st.pills("Examples", list(EXAMPLES), selection_mode="single", label_visibility="collapsed")
    if pick and st.session_state.get("_last_pick") != pick:
        st.session_state["review"] = EXAMPLES[pick]
        st.session_state["_last_pick"] = pick
    text = st.text_area("Feedback", key="review", height=110, placeholder="اكتب رأيك هنا ... or type a review here",
                        label_visibility="collapsed")
    if text.strip():
        rb = svc.predict([text], model="baseline")[0]
        _, words = svc.explain_baseline(text)
        chips = "".join(f'<span class="chip" dir="auto">{escape(w)}</span>' for w in words)
        extra = f'<div class="note" style="margin-top:.6rem">Words that drove it:<br>{chips}</div>' if words else ""
        lang = "Arabic" if rb["language"] == "ar" else "English"
        st.markdown(f'<div class="note">Detected language: <b>{lang}</b></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        if svc.has_transformer:
            rt = svc.predict([text], model="transformer")[0]
            c1.markdown(result_card(rt, "Fine-tuned XLM-RoBERTa (main model)"), unsafe_allow_html=True)
            agree = "Both models agree." if rt["label"] == rb["label"] else \
                "The models disagree. On the test set the transformer is right more often, but a human should read reviews like this."
        else:
            c1.info("Fine-tuned transformer not found. Run `python -m src.transformer` to train it.")
            agree = ""
        c2.markdown(result_card(rb, "TF-IDF + Logistic Regression (baseline)", extra), unsafe_allow_html=True)
        if agree:
            st.markdown(f'<div class="note" style="margin-top:.8rem">{agree}</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ batch
with tab_batch:
    st.markdown('<div class="h">Score many reviews at once <small>one per line, or upload a CSV</small></div>',
                unsafe_allow_html=True)
    model_choice = st.radio("Model", ["transformer", "baseline"], horizontal=True,
                            format_func=lambda m: "Fine-tuned XLM-RoBERTa" if m == "transformer" else "TF-IDF baseline")
    up = st.file_uploader("CSV with a 'text' column", type="csv")
    pasted = st.text_area("Or paste reviews", height=140, value="\n".join(list(EXAMPLES.values())[:6]))
    texts = []
    if up is not None:
        df_up = pd.read_csv(up)
        col = "text" if "text" in df_up.columns else st.selectbox("Text column", df_up.columns)
        texts = df_up[col].astype(str).tolist()[:2000]
    else:
        texts = [t for t in pasted.splitlines() if t.strip()]
    if texts:
        res = pd.DataFrame(svc.predict(texts, model=model_choice))
        res["language"] = res["language"].map({"ar": "Arabic", "en": "English"})
        k = st.columns(4)
        k[0].metric("Reviews", len(res))
        for i, l in enumerate(LABELS):
            k[i + 1].metric(l.capitalize(), f"{(res['label'] == l).mean():.0%}")
        chart = alt.Chart(res).mark_bar(cornerRadiusEnd=4).encode(
            y=alt.Y("language:N", title=None), x=alt.X("count():Q", title="reviews", stack="normalize", axis=alt.Axis(format="%")),
            color=alt.Color("label:N", scale=alt.Scale(domain=LABELS, range=[SENT[l] for l in LABELS]), title="sentiment"),
            order=alt.Order("label:N")).properties(height=110)
        show_chart(chart)
        st.dataframe(res[["text", "language", "label", "confidence"]], width="stretch", hide_index=True,
                     column_config={"confidence": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1)})
        st.download_button("Download results (CSV)", res.to_csv(index=False).encode("utf-8-sig"), "bilingo_scored.csv")

# ------------------------------------------------------------------ comparison
with tab_cmp:
    results, sig = load("results.csv"), load("significance.json")
    if results is None:
        st.info("Run `python -m src.evaluate` to create the comparison.")
    else:
        g = lambda k, s: results[(results["key"] == k) & (results["subset"] == s)]["macro_f1"].iloc[0]
        lo, hi = sig["all"]["ci95"]
        st.markdown(f"""<div class="kpis">
          <div class="kpi hi"><div class="l">Fine-tuned XLM-RoBERTa</div><div class="v">{g('transformer','All test'):.3f}</div><div class="d">macro-F1 on 6,000 test reviews</div></div>
          <div class="kpi"><div class="l">TF-IDF baseline</div><div class="v">{g('baseline','All test'):.3f}</div><div class="d">macro-F1, same test set</div></div>
          <div class="kpi"><div class="l">Difference</div><div class="v">+{sig['all']['diff']:.3f}</div><div class="d">95% CI {lo:+.3f} to {hi:+.3f} (bootstrap)</div></div>
          <div class="kpi"><div class="l">Pretrained, no fine-tuning</div><div class="v">{g('zero_shot','All test'):.3f}</div><div class="d">tweet sentiment model used as is</div></div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="h">Macro-F1 by language <small>higher is better · 0.33 = random guessing</small></div>',
                    unsafe_allow_html=True)
        sub = results[results["subset"] != "All test"]
        chart = alt.Chart(sub).mark_bar(cornerRadiusEnd=4, height=16).encode(
            x=alt.X("macro_f1:Q", title="macro-F1", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("subset:N", title=None, sort=["Arabic", "English", "Arabic SHEIN (new domain)"], axis=alt.Axis(labelLimit=260)),
            yOffset=alt.YOffset("model:N", sort=list(MODEL_COLORS)),
            color=alt.Color("model:N", scale=alt.Scale(domain=list(MODEL_COLORS), range=list(MODEL_COLORS.values())),
                            legend=alt.Legend(orient="top", title=None, labelLimit=260)),
            tooltip=["model", "subset", alt.Tooltip("macro_f1:Q", format=".3f"), alt.Tooltip("accuracy:Q", format=".3f")]
        ).properties(height=alt.Step(18))
        labels = chart.mark_text(align="left", dx=4, fontSize=11).encode(text=alt.Text("macro_f1:Q", format=".2f"), color=alt.value(INK))
        chart = chart + labels
        show_chart(chart)

        st.markdown('<div class="h">Confusion matrices <small>rows = true label, % of each row</small></div>',
                    unsafe_allow_html=True)
        preds = load("test_predictions.csv")
        lang = st.radio("Language", ["ar", "en"], horizontal=True, format_func={"ar": "Arabic", "en": "English"}.get)
        cols = st.columns(3)
        for col, (key, name) in zip(cols, [("zero_shot", "Pretrained, no fine-tuning"),
                                          ("baseline", "TF-IDF baseline"),
                                          ("transformer", "Fine-tuned XLM-RoBERTa")]):
            d = preds[preds["lang"] == lang]
            cm = pd.crosstab(d["label"], d[key]).reindex(index=LABELS, columns=LABELS, fill_value=0)
            m = cm.reset_index().melt(id_vars="label", var_name="predicted", value_name="n")
            m["pct"] = m["n"] / m.groupby("label")["n"].transform("sum")
            base = alt.Chart(m).encode(x=alt.X("predicted:N", sort=LABELS, title="predicted", axis=alt.Axis(labelAngle=0)),
                                       y=alt.Y("label:N", sort=LABELS, title="true"))
            heat = base.mark_rect(stroke="#F7C9D4", strokeWidth=2).encode(color=alt.Color("pct:Q", scale=alt.Scale(domain=[0, 0.3, 0.6, 1], range=["#C8D9E6", "#8EA8C0", "#4E6A87", "#2F4156"], interpolate="rgb"), legend=None))
            txt = base.mark_text(fontSize=12, fontWeight="bold").encode(
                text=alt.Text("pct:Q", format=".0%"), color=alt.condition("datum.pct > 0.3", alt.value("white"), alt.value(INK)))
            with col:
                st.markdown(f"**{name}**")
                show_chart((heat + txt).properties(height=230), grid=False)
        st.markdown('<div class="note">Most mistakes on every model are between <b>mixed</b> and its neighbours. '
                    'For the two trained models, positive and negative are rarely swapped (4-10% of reviews).</div>', unsafe_allow_html=True)

        st.markdown('<div class="h">Can you trust the confidence? <small>fine-tuned model, test set</small></div>',
                    unsafe_allow_html=True)
        bins = pd.cut(preds["transformer_conf"], [0, .6, .8, .95, 1.0], labels=["under 60%", "60-80%", "80-95%", "over 95%"])
        conf = preds.assign(bin=bins, ok=preds["transformer"] == preds["label"]).groupby("bin", observed=True).agg(
            accuracy=("ok", "mean"), reviews=("ok", "size")).reset_index()
        conf["share"] = conf["reviews"] / conf["reviews"].sum()
        c1, c2 = st.columns([3, 2])
        bars = alt.Chart(conf).mark_bar(cornerRadiusEnd=4, color=INDIGO, height=22).encode(
            y=alt.Y("bin:N", title="model confidence", sort=list(conf["bin"])),
            x=alt.X("accuracy:Q", title="accuracy", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            tooltip=["bin", alt.Tooltip("accuracy:Q", format=".1%"), "reviews"]).properties(height=alt.Step(30))
        with c1:
            show_chart(bars + bars.mark_text(align="left", dx=4, color=INK).encode(text=alt.Text("accuracy:Q", format=".0%")))
        top = conf.iloc[-1]
        c2.markdown(f'<div class="note" style="margin-top:1.5rem">When the model is over 95% sure (<b>{top["share"]:.0%}</b> of reviews) '
                    f'it is right <b>{top["accuracy"]:.0%}</b> of the time. Below 60% it is close to a coin flip, so those '
                    'reviews are the ones to send to a person.</div>', unsafe_allow_html=True)

        st.markdown('<div class="h">Per-class F1</div>', unsafe_allow_html=True)
        tbl = results[["model", "subset", "macro_f1", "accuracy", "f1_negative", "f1_mixed", "f1_positive"]]
        st.dataframe(tbl.style.format({c: "{:.3f}" for c in tbl.columns[2:]}), hide_index=True, width="stretch")

# ------------------------------------------------------------------ errors
with tab_err:
    tags, ex = load("error_tags.csv"), load("error_examples.csv")
    if tags is None:
        st.info("Run `python -m src.evaluate` first.")
    else:
        st.markdown('<div class="h">Where the models fail <small>error rate on the test set</small></div>',
                    unsafe_allow_html=True)
        long = tags.melt(id_vars=["lang", "tag", "n"], var_name="model", value_name="error")
        long["model"] = long["model"].map({"error_zero_shot": "Pretrained, no fine-tuning",
                                           "error_baseline": "TF-IDF + Logistic Regression",
                                           "error_transformer": "Fine-tuned XLM-RoBERTa"})
        long["lang"] = long["lang"].map({"ar": "Arabic", "en": "English"})
        long["tag"] = long["tag"] + " (n=" + long["n"].astype(str) + ")"
        cols = st.columns(2)
        for col, lang_name in zip(cols, ["Arabic", "English"]):
            d = long[long["lang"] == lang_name]
            chart = alt.Chart(d).mark_bar(cornerRadiusEnd=3, height=12).encode(
                x=alt.X("error:Q", title="error rate", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 0.75])),
                y=alt.Y("tag:N", title=None, sort=list(d["tag"].unique()), axis=alt.Axis(labelLimit=220)),
                yOffset=alt.YOffset("model:N", sort=list(MODEL_COLORS)),
                color=alt.Color("model:N", scale=alt.Scale(domain=list(MODEL_COLORS), range=list(MODEL_COLORS.values())),
                                legend=alt.Legend(orient="top", title=None, columns=1, labelLimit=260)),
                tooltip=["tag", "model", alt.Tooltip("error:Q", format=".1%")]).properties(height=alt.Step(14))
            with col:
                st.markdown(f"**{lang_name}**")
                show_chart(chart)
        st.markdown('<div class="note">The transformer makes fewer mistakes in every group, but reviews with <b>contrast</b> '
                    '("good location but dirty room") and <b>long</b> reviews stay the hardest for all models.</div>',
                    unsafe_allow_html=True)

        st.markdown('<div class="h">Real mistakes from the test set</div>', unsafe_allow_html=True)
        st.markdown('<div class="note">Not every "mistake" means the model is wrong. Labels come from star ratings, so a review '
                    'that opens with <b>رائع جدا</b> ("excellent") can still be labelled negative. Many confident errors are like this.</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        kind = c1.selectbox("Kind", ex["kind"].unique())
        lang = c2.radio("Language ", ["ar", "en"], horizontal=True, format_func={"ar": "Arabic", "en": "English"}.get)
        for _, r in ex[(ex["kind"] == kind) & (ex["lang"] == lang)].head(8).iterrows():
            st.markdown(f'<div class="ex"><div class="t" dir="auto">{escape(str(r["text"]))}</div>'
                        f'<div class="m">true {pill(r["label"])} · baseline {pill(r["baseline"])} · '
                        f'transformer {pill(r["transformer"])} ({r["transformer_conf"]:.0%})</div></div>',
                        unsafe_allow_html=True)

# ------------------------------------------------------------------ how it works
with tab_how:
    st.markdown('<div class="h">Pipeline</div>', unsafe_allow_html=True)
    st.markdown("""<div class="flow">
      <div class="step"><b>1 · Data</b>Arabic 100K Reviews (hotels, books) and Amazon reviews (English). Stars mapped to negative / mixed / positive.</div>
      <div class="step"><b>2 · Balance</b>5,000 / 1,000 / 1,000 reviews per class per language for train / validation / test. Duplicates removed.</div>
      <div class="step"><b>3 · Cleaning</b>URLs, HTML, mentions removed. For TF-IDF also: Arabic letter normalisation, diacritics, lowercase, repeated letters. Stopwords kept (not, لا, لكن).</div>
      <div class="step"><b>4 · Baseline</b>TF-IDF on words and character n-grams + logistic regression, tuned on validation.</div>
      <div class="step"><b>5 · Transformer</b>XLM-RoBERTa fine-tuned for 3 epochs on the same data. Best epoch chosen on validation.</div>
      <div class="step"><b>6 · Test once</b>Held-out test set plus a new-domain set (Arabic SHEIN fashion reviews in dialect).</div>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="h" style="margin-top:1.2rem">Limitations</div>', unsafe_allow_html=True)
    st.markdown("""
- **Domain.** Arabic training data is hotels and books, English is Amazon products. Performance drops on new domains (see the SHEIN results).
- **Labels come from star ratings.** A 3-star review is "mixed" even if the text sounds positive, so some "errors" are label noise.
- **Dialects.** Most Arabic training reviews are close to Modern Standard Arabic; Egyptian or Gulf slang is weaker.
- **Sarcasm and long reviews.** Sarcasm is often missed, and the transformer only reads the first ~192 tokens.
- **Use with a human in the loop** for anything that affects a customer. The scores are model confidence, not guarantees.
""")

