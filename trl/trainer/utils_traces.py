import re
def _toks(s): return [t.lower() for t in re.findall(r"\w+", s)]
def _ngrams(xs, n): return set(tuple(xs[i:i+n]) for i in range(max(0, len(xs)-n+1)))
def jaccard_ngrams(a, b, n=3):
    A, B = _ngrams(_toks(a), n), _ngrams(_toks(b), n)
    return 1.0 if not A and not B else len(A & B)/max(1, len(A | B))

def mmr_select(items, sim_fn, top_m, alpha=0.7):
    # items: list of (text, utility, payload)
    sel, pool = [], list(items)
    if not pool: return sel
    pool.sort(key=lambda x: x[1], reverse=True)
    sel.append(pool.pop(0))
    while pool and len(sel) < top_m:
        best, best_score = None, -1e9
        for z in pool:
            sim = max(sim_fn(z[0], s[0]) for s in sel)
            score = alpha*z[1] - (1-alpha)*sim
            if score > best_score: best, best_score = z, score
        pool.remove(best); sel.append(best)
    return sel
