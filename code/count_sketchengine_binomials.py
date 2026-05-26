import time
import shutil
import requests
import pandas as pd
import sys
import os

# NOTE: Replace with your own Sketch Engine credentials when running locally
USERNAME = "<YOUR_USERNAME>"
API_KEY  = "<YOUR_API_KEY>"

VIEW_URL = "https://api.sketchengine.eu/bonito/run.cgi/view"
CORP_INFO_URL = "https://api.sketchengine.eu/bonito/run.cgi/corp_info"

WAIT = 1
MAX_BACKOFF = 60

if len(sys.argv) < 2:
    print("Usage: python code/count_sketchengine_binomials.py data/<lang>/input_<lang>.csv")
    sys.exit(1)

INPUT = sys.argv[1]
def make_output_name(path: str) -> str:

    d, base = os.path.split(path)
    if base.startswith("input_"):
        out_base = "output_" + base[len("input_"):]
    else:
        out_base = "output_" + base
    name, ext = os.path.splitext(out_base)
    out_base = f"{name}_sk{ext}"
    return os.path.join(d or ".", out_base)

OUTPUT = make_output_name(INPUT)

name_lower = os.path.basename(INPUT).lower()
if "zh" in name_lower:
    CORPUS = "preloaded/zhtenten17_simplified_stf2"
elif "en" in name_lower:
    CORPUS = "preloaded/ententen21_tt31"
elif "de" in name_lower:
    CORPUS = "preloaded/detenten23_rft3"
elif "ru" in name_lower:
    CORPUS = "preloaded/rutenten20_rft3"
elif "ja" in name_lower:
    CORPUS = "preloaded/jptenten11_suw_comainu_v2"
elif "tr" in name_lower:
    CORPUS = "preloaded/trtenten20_tm2"
elif "id" in name_lower:
    CORPUS = "preloaded/idtenten24_tt2"
elif "ar" in name_lower:
    CORPUS = "preloaded/artenten24_cml1"
else:
    CORPUS = None

def api_get(params: dict, session: requests.Session) -> int:
    backoff = WAIT
    while True:
        r = session.get(VIEW_URL, params=params, auth=(USERNAME, API_KEY), timeout=60)
        if r.status_code == 429:
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)
            continue
        r.raise_for_status()
        j = r.json()
        if isinstance(j, dict) and "error" in j:
            return 0
        return int(j.get("concsize", 0))

def get_tokencount(corpus: str, session: requests.Session) -> int:
    backoff = WAIT
    while True:
        r = session.get(
            CORP_INFO_URL,
            params={"corpname": corpus},
            auth=(USERNAME, API_KEY),
            timeout=60,
        )
        if r.status_code == 429:
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)
            continue
        r.raise_for_status()
        j = r.json()
        try:
            return int(j["sizes"]["tokencount"])
        except Exception:
            raise ValueError(f"Unexpected corp_info response for {corpus}: {j}")

def process_euro(df, sess, lang, conjs):
    mask = df["A_B_count"].isna() | df["B_A_count"].isna()
    total = int(mask.sum())
    done = 0

    def euro_cql(x, conj, y):
        return f'q[(lc="{x}" | lemma_lc="{x}")] [lc="{conj}"] [(lc="{y}" | lemma_lc="{y}")]'

    for i, row in df[mask].iterrows():
        a = row[f"A_{lang}"]
        b = row[f"B_{lang}"]

        ab_total = 0
        ba_total = 0
        ab_by_conj = {}
        ba_by_conj = {}

        for conj in conjs:
            cnt_ab = api_get({
                "corpname": CORPUS, "format": "json", "q": euro_cql(a, conj, b),
                "fromp": 0, "pagesize": 1, "asyn": 0
            }, sess)
            ab_by_conj[conj] = int(cnt_ab)
            ab_total += int(cnt_ab)
            time.sleep(WAIT)

            cnt_ba = api_get({
                "corpname": CORPUS, "format": "json", "q": euro_cql(b, conj, a),
                "fromp": 0, "pagesize": 1, "asyn": 0
            }, sess)
            ba_by_conj[conj] = int(cnt_ba)
            ba_total += int(cnt_ba)
            time.sleep(WAIT)

        df.at[i, "A_B_count"] = ab_total
        df.at[i, "B_A_count"] = ba_total

        if ab_total >= ba_total:
            top_conj = max(conjs, key=lambda c: (ab_by_conj.get(c, 0), -conjs.index(c)))
            pref_str = f"{a} {top_conj} {b}"
        else:
            top_conj = max(conjs, key=lambda c: (ba_by_conj.get(c, 0), -conjs.index(c)))
            pref_str = f"{b} {top_conj} {a}"

        df.at[i, "Order_preference"] = pref_str
        df.at[i, "Preferred_conjunction"] = top_conj

        done += 1
        print(f"[{done}/{total}] {a}/{b} -> AB:{ab_total} BA:{ba_total} | pref:{pref_str} (conj='{top_conj}')")

        df.to_csv(OUTPUT, index=False, encoding="utf-8")

        time.sleep(WAIT)

def process_cjk(df, sess, lang, conjs):
    mask = df["A_B_count"].isna() | df["B_A_count"].isna()
    total = int(mask.sum())
    done = 0

    def cjk_cql_for(x, conj, y):
        xa = str(x).replace('"', r'\"')
        yb = str(y).replace('"', r'\"')
        if conj == "":
            s = f"{xa}{yb}"
            return f'q[word="{s}"]'
        else:
            c = str(conj).replace('"', r'\"')
            return f'q[word="{xa}"] [word="{c}"] [word="{yb}"]'

    for i, row in df[mask].iterrows():
        a = row[f"A_{lang}"]
        b = row[f"B_{lang}"]

        ab_total = 0
        ba_total = 0
        ab_by_conj = {}
        ba_by_conj = {}

        for conj in conjs:
            if conj == "" and lang == "ja":
                a_esc = str(a).replace('"', r'\"')
                b_esc = str(b).replace('"', r'\"')

                q_ab_token = f'q[word="{a_esc}{b_esc}"]'
                cnt_ab_token = api_get({
                    "corpname": CORPUS, "format": "json", "q": q_ab_token,
                    "fromp": 0, "pagesize": 1, "asyn": 0
                }, sess)
                time.sleep(WAIT)

                q_ab_adj = f'q[word="{a_esc}"] [word="{b_esc}"]'
                cnt_ab_adj = api_get({
                    "corpname": CORPUS, "format": "json", "q": q_ab_adj,
                    "fromp": 0, "pagesize": 1, "asyn": 0
                }, sess)
                time.sleep(WAIT)

                cnt_ab = int(cnt_ab_token) + int(cnt_ab_adj)
            else:
                q_ab = cjk_cql_for(a, conj, b)
                cnt_ab = api_get({
                    "corpname": CORPUS, "format": "json", "q": q_ab,
                    "fromp": 0, "pagesize": 1, "asyn": 0
                }, sess)
                time.sleep(WAIT)

            ab_by_conj[conj] = int(cnt_ab)
            ab_total += int(cnt_ab)

            if conj == "" and lang == "ja":
                a_esc = str(a).replace('"', r'\"')
                b_esc = str(b).replace('"', r'\"')

                q_ba_token = f'q[word="{b_esc}{a_esc}"]'
                cnt_ba_token = api_get({
                    "corpname": CORPUS, "format": "json", "q": q_ba_token,
                    "fromp": 0, "pagesize": 1, "asyn": 0
                }, sess)
                time.sleep(WAIT)

                q_ba_adj = f'q[word="{b_esc}"] [word="{a_esc}"]'
                cnt_ba_adj = api_get({
                    "corpname": CORPUS, "format": "json", "q": q_ba_adj,
                    "fromp": 0, "pagesize": 1, "asyn": 0
                }, sess)
                time.sleep(WAIT)

                cnt_ba = int(cnt_ba_token) + int(cnt_ba_adj)
            else:
                q_ba = cjk_cql_for(b, conj, a)
                cnt_ba = api_get({
                    "corpname": CORPUS, "format": "json", "q": q_ba,
                    "fromp": 0, "pagesize": 1, "asyn": 0
                }, sess)
                time.sleep(WAIT)

            ba_by_conj[conj] = int(cnt_ba)
            ba_total += int(cnt_ba)

        df.at[i, "A_B_count"] = ab_total
        df.at[i, "B_A_count"] = ba_total

        if ab_total >= ba_total:
            top_conj = max(conjs, key=lambda c: (ab_by_conj.get(c, 0), -conjs.index(c)))
            pref_str = f"{a}{top_conj}{b}"
        else:
            top_conj = max(conjs, key=lambda c: (ba_by_conj.get(c, 0), -conjs.index(c)))
            pref_str = f"{b}{top_conj}{a}"

        df.at[i, "Order_preference"] = pref_str
        df.at[i, "Preferred_conjunction"] = top_conj

        done += 1
        display = "" if top_conj == "" else top_conj
        print(f"[{done}/{total}] {a}/{b} -> AB:{ab_total} BA:{ba_total} | pref:{pref_str} (conj='{display}')")

        df.to_csv(OUTPUT, index=False, encoding="utf-8")

        time.sleep(WAIT)

def process_turkish(df, sess, lang, _unused):
    mask = df["A_B_count"].isna() | df["B_A_count"].isna()
    total = int(mask.sum())
    done = 0

    def esc_tok(t: str) -> str:
        return str(t).replace('"', r'\"').strip()

    def split_tokens(s: str):
        return [w for w in str(s).split() if w.strip()]

    def seq_lc(tokens):
        return " ".join([f'[lc="{esc_tok(t)}"]' for t in tokens])

    def cql_and_tokenized(a_tokens, conj_tok, b_tokens):
        return f'q{seq_lc(a_tokens)} [lc="{esc_tok(conj_tok)}"] {seq_lc(b_tokens)}'

    def cql_or_yada(a_tokens, b_tokens):
        return f'q{seq_lc(a_tokens)} [lc="ya"] [lc="da"] {seq_lc(b_tokens)}'

    def cql_suffix_ile(a_tokens, b_tokens):
        if not a_tokens:
            return None
        a_pre = a_tokens[:-1]
        a_last = esc_tok(a_tokens[-1])
        a_part = ""
        if a_pre:
            a_part = " " + seq_lc(a_pre)
        last_part = f'[lc="^{a_last}([\'’])?(y)?(la|le)$"]'
        return f'q{a_part} {last_part} {seq_lc(b_tokens)}'.replace("q ", "q")

    def cql_adjacent(a_tokens, b_tokens):
        return f'q{seq_lc(a_tokens)} {seq_lc(b_tokens)}'

    def count(q: str) -> int:
        if not q:
            return 0
        return int(api_get({
            "corpname": CORPUS, "format": "json", "q": q,
            "fromp": 0, "pagesize": 1, "asyn": 0
        }, sess))

    for i, row in df[mask].iterrows():
        a_raw = row.get(f"A_{lang}", "")
        b_raw = row.get(f"B_{lang}", "")
        a_tokens = split_tokens(a_raw)
        b_tokens = split_tokens(b_raw)

        forms = [
            ("ve",      lambda A, B: cql_and_tokenized(A, "ve", B),
                               lambda A, B: cql_and_tokenized(B, "ve", A)),
            ("ile",     lambda A, B: cql_and_tokenized(A, "ile", B),
                               lambda A, B: cql_and_tokenized(B, "ile", A)),
            ("-(y)la/le", lambda A, B: cql_suffix_ile(A, B),
                               lambda A, B: cql_suffix_ile(B, A)),
            ("veya",    lambda A, B: cql_and_tokenized(A, "veya", B),
                               lambda A, B: cql_and_tokenized(B, "veya", A)),
            ("ya da",   lambda A, B: cql_or_yada(A, B),
                               lambda A, B: cql_or_yada(B, A)),
            ("",        lambda A, B: cql_adjacent(A, B),
                               lambda A, B: cql_adjacent(B, A)),
        ]

        ab_total = 0
        ba_total = 0
        ab_by_form = {}
        ba_by_form = {}

        for label, mk_ab, mk_ba in forms:
            q_ab = mk_ab(a_tokens, b_tokens)
            cnt_ab = count(q_ab)
            ab_by_form[label] = cnt_ab
            ab_total += cnt_ab
            time.sleep(WAIT)

            q_ba = mk_ba(a_tokens, b_tokens)
            cnt_ba = count(q_ba)
            ba_by_form[label] = cnt_ba
            ba_total += cnt_ba
            time.sleep(WAIT)

        df.at[i, "A_B_count"] = int(ab_total)
        df.at[i, "B_A_count"] = int(ba_total)

        if ab_total >= ba_total:
            top_label = max(forms, key=lambda f: (ab_by_form.get(f[0], 0), -forms.index(f)))[0]
            if top_label == "-(y)la/le":
                pref_str = f"{a_raw} +{top_label} {b_raw}"
            elif top_label == "ya da":
                pref_str = f"{a_raw} ya da {b_raw}"
            elif top_label == "":
                pref_str = f"{a_raw} {b_raw}"
            else:
                pref_str = f"{a_raw} {top_label} {b_raw}"
        else:
            top_label = max(forms, key=lambda f: (ba_by_form.get(f[0], 0), -forms.index(f)))[0]
            if top_label == "-(y)la/le":
                pref_str = f"{b_raw} +{top_label} {a_raw}"
            elif top_label == "ya da":
                pref_str = f"{b_raw} ya da {a_raw}"
            elif top_label == "":
                pref_str = f"{b_raw} {a_raw}"
            else:
                pref_str = f"{b_raw} {top_label} {a_raw}"

        df.at[i, "Order_preference"] = pref_str
        df.at[i, "Preferred_conjunction"] = top_label

        done += 1
        print(f"[{done}/{total}] {a_raw}/{b_raw} -> AB:{ab_total} BA:{ba_total} | "
              f"pref:{pref_str} (conj='{top_label}')")

        df.to_csv(OUTPUT, index=False, encoding="utf-8")

def process_indonesian(df, sess, lang, conjs):
    mask = df["A_B_count"].isna() | df["B_A_count"].isna()
    total = int(mask.sum())
    done = 0

    def esc(t: str) -> str:
        return str(t).replace('"', r'\"').strip()

    def toks(s: str):
        return [w for w in str(s).split() if w.strip()]

    def seq(tokens):
        return " ".join([f'[lc="{esc(t)}"]' for t in tokens])

    def with_conj(a_tokens, conj, b_tokens):
        return f'q{seq(a_tokens)} [lc="{esc(conj)}"] {seq(b_tokens)}'

    def adjacent(a_tokens, b_tokens):
        return f'q{seq(a_tokens)} {seq(b_tokens)}'

    def count(q: str) -> int:
        return int(api_get({
            "corpname": CORPUS, "format": "json", "q": q,
            "fromp": 0, "pagesize": 1, "asyn": 0
        }, sess))

    for i, row in df[mask].iterrows():
        a_raw = row.get(f"A_{lang}", "")
        b_raw = row.get(f"B_{lang}", "")
        a = toks(a_raw)
        b = toks(b_raw)

        forms = [
            ("dan",     lambda A, B: with_conj(A, "dan", B),    lambda A, B: with_conj(B, "dan", A)),
            ("atau",    lambda A, B: with_conj(A, "atau", B),   lambda A, B: with_conj(B, "atau", A)),
            ("",        lambda A, B: adjacent(A, B),            lambda A, B: adjacent(B, A)),
        ]

        ab_total = 0
        ba_total = 0
        ab_by = {}
        ba_by = {}

        for label, mk_ab, mk_ba in forms:
            q_ab = mk_ab(a, b)
            cab = count(q_ab)
            ab_by[label] = cab
            ab_total += cab
            time.sleep(WAIT)

            q_ba = mk_ba(a, b)
            cba = count(q_ba)
            ba_by[label] = cba
            ba_total += cba
            time.sleep(WAIT)

        df.at[i, "A_B_count"] = int(ab_total)
        df.at[i, "B_A_count"] = int(ba_total)

        if ab_total >= ba_total:
            top = max(forms, key=lambda f: (ab_by.get(f[0], 0), -forms.index(f)))[0]
            pref = f"{a_raw} {('' if top == '' else top + ' ')}{b_raw}".strip()
        else:
            top = max(forms, key=lambda f: (ba_by.get(f[0], 0), -forms.index(f)))[0]
            pref = f"{b_raw} {('' if top == '' else top + ' ')}{a_raw}".strip()

        df.at[i, "Order_preference"] = pref
        df.at[i, "Preferred_conjunction"] = top

        done += 1
        print(f"[{done}/{total}] {a_raw}/{b_raw} -> AB:{ab_total} BA:{ba_total} | "
              f"pref:{pref} (conj='{top}')")

        df.to_csv(OUTPUT, index=False, encoding="utf-8")

def process_arabic(df, sess, lang):
    mask = df["A_B_count"].isna() | df["B_A_count"].isna()
    total = int(mask.sum())
    done = 0

    def esc_word(t: str) -> str:
        return str(t).replace('"', r'\"').strip()

    def toks(s: str):
        return [w for w in str(s).split() if w.strip()]

    def seq_with_al(tokens):
        return " ".join(f'[word="^(?:ال)?{esc_word(t)}$"]' for t in tokens)

    def seq_b_with_attached_waw(b_tokens):
        if not b_tokens:
            return ""
        head = f'[word="^و(?:ال)?{esc_word(b_tokens[0])}$"]'
        if len(b_tokens) == 1:
            return head
        tail = " ".join(f'[word="^(?:ال)?{esc_word(t)}$"]' for t in b_tokens[1:])
        return f"{head} {tail}"

    def seq_b_without_waw(b_tokens):
        if not b_tokens:
            return ""
        head = f'[word="^(?:ال)?{esc_word(b_tokens[0])}$"]'
        if len(b_tokens) == 1:
            return head
        tail = " ".join(f'[word="^(?:ال)?{esc_word(t)}$"]' for t in b_tokens[1:])
        return f"{head} {tail}"

    def render_waw_attached(a_raw: str, b_raw: str) -> str:
        bt = toks(b_raw)
        if not bt:
            return a_raw
        attached_b = "و" + bt[0]
        if len(bt) > 1:
            attached_b += " " + " ".join(bt[1:])
        return f"{a_raw} {attached_b}"

    def render_waw_attached_rev(a_raw: str, b_raw: str) -> str:
        at = toks(a_raw)
        if not at:
            return b_raw
        attached_a = "و" + at[0]
        if len(at) > 1:
            attached_a += " " + " ".join(at[1:])
        return f"{b_raw} {attached_a}"

    def q_and_separate(a_tokens, b_tokens):
        return f'q{seq_with_al(a_tokens)} [word="^و$"] {seq_b_without_waw(b_tokens)}'

    def q_and_attached(a_tokens, b_tokens):
        return f'q{seq_with_al(a_tokens)} {seq_b_with_attached_waw(b_tokens)}'

    def q_or(a_tokens, b_tokens):
        return f'q{seq_with_al(a_tokens)} [word="^(?:أو|او)$"] {seq_with_al(b_tokens)}'

    def q_adjacent(a_tokens, b_tokens):
        return f'q{seq_with_al(a_tokens)} {seq_with_al(b_tokens)}'

    def count(q: str) -> int:
        return int(api_get({
            "corpname": CORPUS, "format": "json", "q": q,
            "fromp": 0, "pagesize": 1, "asyn": 0
        }, sess))

    forms = [
        ('و',          q_and_separate,  lambda A,B: q_and_separate(B,A)),
        ('و_attached', q_and_attached,  lambda A,B: q_and_attached(B,A)),
        ('او',      q_or,            lambda A,B: q_or(B,A)),
        ('',           q_adjacent,      lambda A,B: q_adjacent(B,A)),
    ]

    for i, row in df[mask].iterrows():
        a_raw = row.get(f"A_{lang}", "")
        b_raw = row.get(f"B_{lang}", "")
        A = toks(a_raw)
        B = toks(b_raw)

        ab_total = 0
        ba_total = 0
        ab_by = {}
        ba_by = {}

        for label, mk_ab, mk_ba in forms:
            q_ab = mk_ab(A, B)
            cab  = count(q_ab)
            ab_by[label] = cab
            ab_total += cab
            time.sleep(WAIT)

            q_ba = mk_ba(A, B)
            cba  = count(q_ba)
            ba_by[label] = cba
            ba_total += cba
            time.sleep(WAIT)

        df.at[i, "A_B_count"] = int(ab_total)
        df.at[i, "B_A_count"] = int(ba_total)

        if ab_total >= ba_total:
            top = max(forms, key=lambda f: (ab_by.get(f[0], 0), -forms.index(f)))[0]
            if top == 'و_attached':
                pref = render_waw_attached(a_raw, b_raw)
            elif top == 'و':
                pref = f"{a_raw} و {b_raw}"
            elif top == 'أو/او':
                pref = f"{a_raw} أو/او {b_raw}"
            else:
                pref = f"{a_raw} {b_raw}"
        else:
            top = max(forms, key=lambda f: (ba_by.get(f[0], 0), -forms.index(f)))[0]
            if top == 'و_attached':
                pref = render_waw_attached_rev(a_raw, b_raw)
            elif top == 'و':
                pref = f"{b_raw} و {a_raw}"
            elif top == 'أو/او':
                pref = f"{b_raw} أو/او {a_raw}"
            else:
                pref = f"{b_raw} {a_raw}"

        df.at[i, "Order_preference"] = pref
        df.at[i, "Preferred_conjunction"] = top

        done += 1
        print(f"[{done}/{total}] {a_raw}/{b_raw} -> AB:{ab_total} BA:{ba_total} | "
              f"pref:{pref} (conj='{top}')")

        df.to_csv(OUTPUT, index=False, encoding="utf-8")



def main():

    if CORPUS is None:
        print(f"Language not supported for file {INPUT}")
        return
    if not os.path.exists(OUTPUT):
        shutil.copyfile(INPUT, OUTPUT)
        print(f"Created {OUTPUT} from {INPUT}")
    else:
        print(f"Resuming with existing {OUTPUT}")

    try:
        df = pd.read_csv(OUTPUT, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(OUTPUT, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(OUTPUT, encoding="latin1")

    lang = os.path.basename(INPUT).split("_")[1].split(".")[0]

    sess = requests.Session()

    if "en" in name_lower:
        conjs = ["and", "or"]
        process_euro(df, sess, lang, conjs)
    elif "de" in name_lower:
        conjs = ["und", "oder"]
        process_euro(df, sess, lang, conjs)
    elif "ru" in name_lower:
        conjs = ["и", "или"]
        process_euro(df, sess, lang, conjs)
    elif "zh" in name_lower:
        conjs = ["和", "与", "或", ""]
        process_cjk(df, sess, lang, conjs)
    elif "ja" in name_lower:
        conjs = ["と", "または", ""]
        process_cjk(df, sess, lang, conjs)
    elif "tr" in name_lower:
        process_turkish(df, sess, lang, None)
    elif "id" in name_lower:
        conjs = ["dan", "atau", ""]
        process_indonesian(df, sess, lang, conjs)
    elif "ar" in name_lower:
        process_arabic(df, sess, lang)
    else:
        print(f"Language not supported for file {INPUT}")
        return


    df.to_csv(OUTPUT, index=False, encoding="utf-8")

    print(f"Done. Updated {OUTPUT}. (Input left unchanged: {INPUT})")
    print(f"Corpus used: {CORPUS}")
    print(f"Output file: {OUTPUT}")

if __name__ == "__main__":
    main()