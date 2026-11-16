"""Run end-to-end Day 07 experiments: 3 Retrieval Methods, 2 Improvement Configurations, Agent Evaluation."""
import json
import time
import subprocess
import sys
from pathlib import Path

from src.retrievers import DenseRetriever, LexicalRetriever, HybridRetriever
from src.controller import GeminiController
from starter.baseline import pack_context, hit
from starter.agentic import run_agent

base_dir = Path(__file__).resolve().parents[1]
data_dir = base_dir / "data"
results_dir = base_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)

# 1. Load data
chunks = json.loads((data_dir / "chunks.json").read_text(encoding="utf-8"))
questions = json.loads((data_dir / "questions.json").read_text(encoding="utf-8"))
cases = json.loads((data_dir / "cases.json").read_text(encoding="utf-8"))

dev_questions = [q for q in questions if q["split"] == "dev"]
test_questions = [q for q in questions if q["split"] == "test"]
dev_cases = [c for c in cases if c["split"] == "dev"]
test_cases = [c for c in cases if c["split"] == "test"]

print(f"Loaded {len(chunks)} chunks, {len(questions)} answerable questions, {len(cases)} total cases.")
print(f"Dev: {len(dev_questions)} questions ({len(dev_cases)} cases), Test: {len(test_questions)} questions ({len(test_cases)} cases)")

# Initialize retrievers
dense = DenseRetriever(storage_path=str(data_dir / "qdrant_db"), collection_name="strategy_c")
lexical = LexicalRetriever(chunks)
hybrid = HybridRetriever(dense, lexical)

retrievers = {
    "dense": dense,
    "lexical": lexical,
    "hybrid": hybrid
}

def evaluate_retrieval(retriever_name, q_list, budget=800, top_k=3):
    r = retrievers[retriever_name]
    traces = []
    t0 = time.perf_counter()
    for q in q_list:
        start = time.perf_counter()
        flt = q.get("filter")
        hits = r.search(q["question"], filters=flt, top_k=top_k)
        context, ids = pack_context(hits, budget=budget)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        marked = [{**c, "hit": hit(c, q["gold"])} for c in hits]
        is_hit1 = bool(marked and marked[0]["hit"])
        is_hit3 = any(c["hit"] for c in marked)
        is_ctx_hit = any(c["hit"] and c["id"] in ids for c in marked)
        
        traces.append({
            "query_id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "gold": q["gold"],
            "results": marked,
            "context": context,
            "context_ids": ids,
            "hit1": is_hit1,
            "hit3": is_hit3,
            "context_hit": is_ctx_hit,
            "elapsed_ms": elapsed_ms
        })
    total_time = time.perf_counter() - t0
    n = len(q_list)
    h1 = sum(1 for t in traces if t["hit1"])
    h3 = sum(1 for t in traces if t["hit3"])
    ctx_h = sum(1 for t in traces if t["context_hit"])
    return {
        "n": n,
        "hit1": h1,
        "hit1_ratio": f"{h1}/{n}",
        "hit1_pct": round(h1 / n * 100, 1),
        "hit3": h3,
        "hit3_ratio": f"{h3}/{n}",
        "hit3_pct": round(h3 / n * 100, 1),
        "context_hit": ctx_h,
        "context_hit_ratio": f"{ctx_h}/{n}",
        "context_hit_pct": round(ctx_h / n * 100, 1),
        "total_time_sec": round(total_time, 2),
        "avg_ms": round(total_time * 1000 / n, 2),
        "traces": traces
    }

print("\n" + "="*70)
print("PART 1: RETRIEVAL METHODS COMPARISON ON DEV (n=15)")
print("="*70)
dev_retrieval_results = {}
for name in ["lexical", "dense", "hybrid"]:
    res = evaluate_retrieval(name, dev_questions, budget=800, top_k=3)
    dev_retrieval_results[name] = res
    print(f"Method: {name:<8} | Hit@1: {res['hit1_ratio']:<6} ({res['hit1_pct']}%) | Hit@3: {res['hit3_ratio']:<6} ({res['hit3_pct']}%) | CtxHit: {res['context_hit_ratio']:<6} ({res['context_hit_pct']}%) | Avg ms: {res['avg_ms']}")

print("\n" + "="*70)
print("PART 2: RETRIEVAL IMPROVEMENT EXPERIMENTS (DEV)")
print("="*70)
# Cải thiện 1: Tăng budget từ 800 lên 1500 chars (giải quyết lỗi rớt chunk pháp lý lớn)
res_imp1 = evaluate_retrieval("dense", dev_questions, budget=1500, top_k=3)
print(f"Imp 1 (Dense budget=1500):  Hit@1: {res_imp1['hit1_ratio']} | Hit@3: {res_imp1['hit3_ratio']} | CtxHit: {res_imp1['context_hit_ratio']} ({res_imp1['context_hit_pct']}%)")

# Cải thiện 2: Hybrid RRF + budget=1500 (kết hợp cả từ khóa chính xác và ngữ nghĩa)
res_imp2 = evaluate_retrieval("hybrid", dev_questions, budget=1500, top_k=3)
print(f"Imp 2 (Hybrid budget=1500): Hit@1: {res_imp2['hit1_ratio']} | Hit@3: {res_imp2['hit3_ratio']} | CtxHit: {res_imp2['context_hit_ratio']} ({res_imp2['context_hit_pct']}%)")

# Test evaluation for baseline vs chosen best configuration
test_retrieval_baseline = evaluate_retrieval("dense", test_questions, budget=800, top_k=3)
test_retrieval_chosen = evaluate_retrieval("hybrid", test_questions, budget=1500, top_k=3)
print(f"Test Baseline (Dense 800):  Hit@1: {test_retrieval_baseline['hit1_ratio']} | Hit@3: {test_retrieval_baseline['hit3_ratio']} | CtxHit: {test_retrieval_baseline['context_hit_ratio']}")
print(f"Test Chosen (Hybrid 1500):  Hit@1: {test_retrieval_chosen['hit1_ratio']} | Hit@3: {test_retrieval_chosen['hit3_ratio']} | CtxHit: {test_retrieval_chosen['context_hit_ratio']}")

# Save retrieval summary
retrieval_summary = {
    "dev_methods": {
        "lexical": {k: v for k, v in dev_retrieval_results["lexical"].items() if k != "traces"},
        "dense": {k: v for k, v in dev_retrieval_results["dense"].items() if k != "traces"},
        "hybrid": {k: v for k, v in dev_retrieval_results["hybrid"].items() if k != "traces"},
        "imp1_dense_1500": {k: v for k, v in res_imp1.items() if k != "traces"},
        "imp2_hybrid_1500": {k: v for k, v in res_imp2.items() if k != "traces"}
    },
    "test": {
        "baseline_dense_800": {k: v for k, v in test_retrieval_baseline.items() if k != "traces"},
        "chosen_hybrid_1500": {k: v for k, v in test_retrieval_chosen.items() if k != "traces"}
    }
}
with open(results_dir / "retrieval_summary.json", "w", encoding="utf-8") as f:
    json.dump(retrieval_summary, f, ensure_ascii=False, indent=2)

print("\n" + "="*70, flush=True)
print("PART 3: AGENT EVALUATION (DEV & TEST)", flush=True)
print("="*70, flush=True)
controller = GeminiController()

# Pre-cache unanswerable questions to avoid rate limit spikes
for c in cases:
    if not c["answerable"]:
        dense.embed_query(c["question"])

def run_agent_suite(case_list, retriever_obj, budget=1500, max_searches=3, top_k=3):
    traces = []
    reviews = []
    q_map = {q["id"]: q for q in questions}

    for idx, c in enumerate(case_list, start=1):
        cid = c["id"]
        q_text = c["question"]
        is_answerable = c["answerable"]
        flt = q_map.get(cid, {}).get("filter")

        print(f"  [{idx}/{len(case_list)}] Running Agent on {cid} ({'Answerable' if is_answerable else 'Unanswerable'})...", flush=True)
        res = run_agent(
            question=q_text,
            controller=controller,
            retriever=retriever_obj,
            filters=flt,
            max_searches=max_searches,
            top_k=top_k,
            budget=budget
        )
        
        status = res["status"]
        answer = res["answer"]
        citations = res["citations"]
        events = res["events"]

        trace_entry = {
            "query_id": cid,
            "status": status,
            "answer": answer,
            "citations": citations,
            "events": events
        }
        traces.append(trace_entry)

        # Build audit review
        if is_answerable:
            q_gold = q_map[cid]["gold"]
            all_retrieved = {}
            for e in events:
                if e.get("event") == "search":
                    for chk in e.get("results", []):
                        all_retrieved[chk["id"]] = chk
            
            c_supports = []
            for cit_id in citations:
                chk = all_retrieved.get(cit_id)
                if chk and hit(chk, q_gold):
                    c_supports.append(True)
                else:
                    c_supports.append(False)

            if status == "answered" and any(c_supports):
                grade = 2
            elif status == "answered":
                grade = 1
            else:
                grade = 0

            reviews.append({
                "query_id": cid,
                "answer_grade": grade,
                "citation_support": c_supports
            })
        else:
            is_correct_abstain = (status == "abstain")
            reviews.append({
                "query_id": cid,
                "abstention_correct": is_correct_abstain,
                "citation_support": []
            })
            
        time.sleep(0.5)

    return traces, reviews

print("\n--- Running Agent on DEV set (17 cases) with Chosen Config (Hybrid RRF + budget=1500) ---", flush=True)
dev_agent_traces, dev_agent_reviews = run_agent_suite(dev_cases, hybrid, budget=1500, max_searches=3, top_k=3)

with open(results_dir / "agent_dev_traces.json", "w", encoding="utf-8") as f:
    json.dump(dev_agent_traces, f, ensure_ascii=False, indent=2)
with open(results_dir / "agent_dev_reviews.json", "w", encoding="utf-8") as f:
    json.dump(dev_agent_reviews, f, ensure_ascii=False, indent=2)

print("\n--- Running Agent on TEST set (7 cases) with Chosen Config (Hybrid RRF + budget=1500) ---", flush=True)
test_agent_traces, test_agent_reviews = run_agent_suite(test_cases, hybrid, budget=1500, max_searches=3, top_k=3)

with open(results_dir / "agent_test_traces.json", "w", encoding="utf-8") as f:
    json.dump(test_agent_traces, f, ensure_ascii=False, indent=2)
with open(results_dir / "agent_test_reviews.json", "w", encoding="utf-8") as f:
    json.dump(test_agent_reviews, f, ensure_ascii=False, indent=2)

print("\n--- Running starter/agent_metrics.py to aggregate metrics ---", flush=True)
# Dev metrics
subprocess.run([
    sys.executable, "starter/agent_metrics.py",
    "--cases", str(data_dir / "cases.json"),
    "--traces", str(results_dir / "agent_dev_traces.json"),
    "--reviews", str(results_dir / "agent_dev_reviews.json"),
    "--split", "dev",
    "--output", str(results_dir / "agent_dev_summary.json")
], check=True)

# Test metrics
subprocess.run([
    sys.executable, "starter/agent_metrics.py",
    "--cases", str(data_dir / "cases.json"),
    "--traces", str(results_dir / "agent_test_traces.json"),
    "--reviews", str(results_dir / "agent_test_reviews.json"),
    "--split", "test",
    "--output", str(results_dir / "agent_test_summary.json")
], check=True)

dense.close()
print("\nALL EXPERIMENTS COMPLETED SUCCESSFULLY!", flush=True)
