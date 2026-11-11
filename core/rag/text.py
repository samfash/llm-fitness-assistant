"""Tokenisation shared by indexing and querying (lowercase, stop-words, light stemming)."""
import re

_WORD = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
STOP = frozenset("""
a an the and or but if then so of to in on at by for with from as is are was were be been being am do does did
doing have has had i me my we our you your it its this that these those what which who whom how why when where
can could should would will shall may might must not no yes about into over under than too very just also any
some more most other such only own same up down out off again further once there here all both each few
get got make made want need like really much many okay ok please tell help
""".split())


def stem(w: str) -> str:
    """Crude suffix stripper: muscle/muscles, lift/lifting/lifted, run/running, body/bodies all collapse."""
    if len(w) <= 3 or w.isdigit():
        return w
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("s") and not w.endswith(("ss", "us")):
        w = w[:-1]
    for suf in ("ing", "ed"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)]
            if len(w) > 3 and w[-1] == w[-2] and w[-1] in "bdgmnprt":
                w = w[:-1]
            break
    if w.endswith("e") and len(w) > 3:
        w = w[:-1]
    return w


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in _WORD.findall(text.lower()) if t not in STOP]
