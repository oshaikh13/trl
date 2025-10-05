import math, re
from collections import Counter
from typing import Optional, Iterable

class InMemoryBM25Temporal:
    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self._tok = re.compile(r"\w+")
        self.docs, self.df, self.N, self.avgdl = [], Counter(), 0, 0.0

    def _toks(self, s): return [t.lower() for t in self._tok.findall(s)]

    def add(self, text: str, *, event_ts: int, visible_after_ts: Optional[int] = None,
            namespace: str = "train", metadata: Optional[dict] = None):
        if visible_after_ts is None: visible_after_ts = event_ts
        toks = self._toks(text)
        self.docs.append({
            "text": text, "toks": toks, "len": len(toks),
            "event_ts": int(event_ts), "visible_after_ts": int(visible_after_ts),
            "namespace": namespace, "meta": metadata or {},
        })
        for w in set(toks): self.df[w] += 1
        self.N = len(self.docs)
        self.avgdl = sum(d["len"] for d in self.docs)/max(1, self.N)

    def query(self, text: str, *, k: int, cutoff_ts: int,
              namespaces: Optional[Iterable[str]] = None,
              time_decay_lambda: Optional[float] = None):
        q = self._toks(text)
        ns = set(namespaces) if namespaces else None
        scores = []
        for i, d in enumerate(self.docs):
            if d["event_ts"] > cutoff_ts: continue
            if d["visible_after_ts"] > cutoff_ts: continue
            if ns and d["namespace"] not in ns: continue
            tf = Counter(d["toks"])
            s = 0.0
            for w in q:
                dfw = self.df.get(w, 0)
                if dfw == 0: continue
                idf = math.log(1 + (self.N - dfw + 0.5)/(dfw + 0.5))
                denom = tf[w] + self.k1*(1 - self.b + self.b*d["len"]/max(1, self.avgdl))
                s += idf * (tf[w]*(self.k1+1))/max(1e-6, denom)
            if time_decay_lambda:
                age = max(0, cutoff_ts - d["event_ts"])
                s *= math.exp(-time_decay_lambda * age)
            if s > 0.0: scores.append((s, i))
        scores.sort(reverse=True)
        out = []
        for s, idx in scores[:k]:
            di = self.docs[idx]
            out.append({"text": di["text"], "meta": di["meta"], "score": s, "event_ts": di["event_ts"]})
        return out
