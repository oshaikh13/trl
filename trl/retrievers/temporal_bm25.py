import math, re
from collections import Counter
from typing import Optional, Iterable, Callable

class InMemoryBM25Temporal:
    def __init__(self, k1=1.5, b=0.75, dedup_threshold=0.8, dedup_sim_fn=None):
        self.k1, self.b = k1, b
        self._tok = re.compile(r"\w+")
        self.docs, self.df, self.N, self.avgdl = [], Counter(), 0, 0.0
        self.dedup_threshold = dedup_threshold
        self.dedup_sim_fn = dedup_sim_fn  # function(text1, text2) -> float in [0,1]

    def _toks(self, s): return [t.lower() for t in self._tok.findall(s)]

    def reset(self):
        self.docs = []
        self.df = Counter()
        self.N = 0
        self.avgdl = 0.0

    def add(self, text: str, *, event_ts: int, visible_after_ts: Optional[int] = None,
            namespace: str = "train", metadata: Optional[dict] = None):
        if visible_after_ts is None: visible_after_ts = event_ts
        toks = self._toks(text)
        
        # DATABASE-level deduplication: check if similar doc already exists
        if self.dedup_sim_fn is not None:
            for i, existing_doc in enumerate(self.docs):
                # Only compare within same namespace
                if existing_doc["namespace"] != namespace:
                    continue
                
                sim = self.dedup_sim_fn(text, existing_doc["text"])
                if sim >= self.dedup_threshold:
                    # Found a close match! Keep the latest version
                    if event_ts >= existing_doc["event_ts"]:
                        # New doc is newer or same age - replace the old one
                        # First, update document frequencies by removing old doc's contribution
                        for w in set(existing_doc["toks"]):
                            self.df[w] -= 1
                            if self.df[w] <= 0:
                                del self.df[w]
                        
                        # Replace with new doc
                        self.docs[i] = {
                            "text": text, "toks": toks, "len": len(toks),
                            "event_ts": int(event_ts), "visible_after_ts": int(visible_after_ts),
                            "namespace": namespace, "meta": metadata or {},
                        }
                        
                        # Update document frequencies with new doc
                        for w in set(toks): self.df[w] += 1
                        # N stays the same, recompute avgdl
                        self.avgdl = sum(d["len"] for d in self.docs)/max(1, self.N)
                    # else: existing doc is newer, skip adding this one
                    return  # Either way, don't add as a new document
        
        # No similar doc found, add as new document
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
