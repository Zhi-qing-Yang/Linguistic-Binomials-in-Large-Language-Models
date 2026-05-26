import pandas as pd

CJK_LANGS = {"zh", "ja"}


def build_binomial_surface(first, conj, second, lang):
    first = str(first).strip()
    second = str(second).strip()
    conj = "" if pd.isna(conj) else str(conj).strip()

    if lang == "ar" and conj == "و_attached":
        return f"{first} و{second}"

    if lang == "ar" and conj == "او":
        conj = "أو"

    if conj:
        if lang in CJK_LANGS:
            return f"{first}{conj}{second}"
        return f"{first} {conj} {second}"

    if lang in CJK_LANGS:
        return f"{first}{second}"
    return f"{first} {second}"
