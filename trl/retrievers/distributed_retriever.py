import itertools
import torch
from accelerate.utils import gather_object, broadcast_object_list, gather

class DistributedRetriever:
    def __init__(self, retriever, accelerator, namespace: str):
        self.retriever = retriever      # rank 0 is authoritative
        self.acc = accelerator
        self.namespace = namespace

    def query_batch(self, queries, cutoff_ts_list, top_k, time_decay_lambda=None, namespaces=None):
        assert len(queries) == len(cutoff_ts_list)
        acc = self.acc
        local_n = len(queries)
        counts = gather(torch.tensor([local_n], device=acc.device)).cpu().tolist()
        starts = list(itertools.accumulate([0] + counts))
        all_queries  = gather_object(queries)
        all_cutoffs  = gather_object(cutoff_ts_list)

        if acc.is_main_process:
            all_hits = []
            for q, cut in zip(all_queries, all_cutoffs):
                hits = self.retriever.query(
                    q, k=top_k, cutoff_ts=int(cut),
                    namespaces=namespaces or [self.namespace],
                    time_decay_lambda=time_decay_lambda
                )
                all_hits.append(hits)
        else:
            all_hits = None

        payload = [all_hits]
        broadcast_object_list(payload, from_process=0)
        all_hits = payload[0]
        if local_n == 0: return []
        rank = acc.process_index
        start, end = starts[rank], starts[rank+1]
        return all_hits[start:end]

    def query_single(self, query, cutoff_ts, top_k, time_decay_lambda=None, namespaces=None):
        return self.query_batch([query], [cutoff_ts], top_k, time_decay_lambda, namespaces)[0]

    def reset(self):
        if self.acc.is_main_process:
            if hasattr(self.retriever, "reset"):
                self.retriever.reset()
        self.acc.wait_for_everyone()

    def add_candidates_parsimonious(self, local_rows, *, dedup_sim_fn, mmr_select_fn, top_m, alpha, visible_delay=1):
        acc = self.acc
        all_rows = gather_object(local_rows)
        if acc.is_main_process:
            from collections import defaultdict
            buckets = defaultdict(list)
            for r in all_rows:
                buckets[r["bucket_key"]].append(r)
            for _, rows in buckets.items():
                items = [(r["text"], float(r["utility"]), r) for r in rows]
                selected = mmr_select_fn(items, sim_fn=dedup_sim_fn, top_m=top_m, alpha=alpha)
                for text, util, r in selected:
                    self.retriever.add(
                        text=text,
                        event_ts=int(r["now_ts"]),
                        visible_after_ts=int(r["now_ts"]) + visible_delay,
                        namespace=self.namespace,
                        metadata={"origin": "revise", "utility": util},
                    )
        acc.wait_for_everyone()
