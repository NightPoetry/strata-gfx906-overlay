#!/usr/bin/env python3
# measure.py <port> <tag> — 速度块: 健康/短prompt 3x256/长prompt prefill+decode+draft → results/<tag>-speed.json
import json, os, sys, time, urllib.request

port, tag = sys.argv[1], sys.argv[2]
BASE = "/data/ai/quant-probe"
os.makedirs(f"{BASE}/results", exist_ok=True)

def call(body, timeout=900):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

health = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10))

SHORT = "请用三句话介绍一下长江和黄河的发源地与流向。"
rounds = []
for r in range(3):
    t0 = time.time()
    resp = call({"model": "probe", "messages": [{"role": "user", "content": SHORT}],
                 "max_tokens": 256, "temperature": 0.7})
    dt = time.time() - t0
    ct = resp["usage"]["completion_tokens"]
    t = resp.get("timings", {})
    rounds.append({"s": round(dt, 2), "tok": ct, "toks": round(ct / dt, 1),
                   "draft": [t.get("draft_n"), t.get("draft_n_accepted")]})
med = sorted(r["toks"] for r in rounds)[1]

para = ("长江发源于青藏高原唐古拉山脉，黄河发源于巴颜喀拉山脉，两条大河都自西向东流入大海，"
        "孕育了中华文明的农耕与航运传统，沿岸生态与水利工程建设深刻影响了区域经济发展。")
long_body = {"model": "probe",
             "messages": [{"role": "user", "content": para * 320 + " 请用一句话总结。"}],
             "max_tokens": 400, "temperature": 0.7}
lr = call(long_body, timeout=1200)
lt = lr.get("timings", {})
out = {"tag": tag, "health_max_context": health.get("max_context"),
       "short_rounds": rounds, "short_median_toks": med,
       "long_prompt_n": lt.get("prompt_n"), "long_prefill_toks": lt.get("prompt_per_second"),
       "long_decode_toks": lt.get("predicted_per_second"),
       "long_draft": [lt.get("draft_n"), lt.get("draft_n_accepted")]}
with open(f"{BASE}/results/{tag}-speed.json", "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f"{tag} speed: short_med={med} long_prefill={out['long_prefill_toks']} "
      f"long_decode={out['long_decode_toks']} draft={out['long_draft']}")
