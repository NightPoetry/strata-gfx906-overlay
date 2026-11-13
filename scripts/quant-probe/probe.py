#!/usr/bin/env python3
# quant-probe: 量化差距探测套件 — 四室程序化判分（无裁判模型）
# think budget pinned at 8192, uncapped generation (2026-11-01 protocol): per-task
# think chars / answer chars / total generated tokens are recorded,
# 思考消耗本身作为跨量化档指标（同 tokenizer 可比）
# 用法: python3 probe.py <port> <tag> [suite...]  输出: <BASE>/results/<tag>-<suite>.json
import json, gzip, os, random, re, subprocess, sys, time, urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8099"
TAG = sys.argv[2] if len(sys.argv) > 2 else "run"
ONLY = set(sys.argv[3:])
BASE = "/path/to/quant-probe"
OUT = os.path.join(BASE, "results")
os.makedirs(OUT, exist_ok=True)
random.seed(42)  # 全局定种子: 各量化档抽到同一批题
BUDGET = 8192    # 思考不设上限(引擎服务端无钳, 8192=网关同级上限)

def ask(prompt, max_tokens=BUDGET):
    body = json.dumps({"model": "probe", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    for attempt in range(2):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=1200))
            m = r["choices"][0]["message"]
            u = r.get("usage", {})
            met = {"tok_total": u.get("completion_tokens", 0),
                   "reason_chars": len(m.get("reasoning_content") or ""),
                   "content_chars": len(m.get("content") or ""),
                   "finish": r["choices"][0].get("finish_reason")}
            return m.get("content") or "", met
        except Exception as e:
            if attempt == 1:
                return "", {"tok_total": 0, "reason_chars": 0, "content_chars": 0, "finish": "error"}
            time.sleep(3)

def seeded(items, n):
    return random.sample(items, min(n, len(items)))

# ---------- 室1 GSM8K ----------
def gsm8k_room():
    rows = [json.loads(l) for l in open(f"{BASE}/data/gsm8k_test.parquet.jsonl")]
    cases = seeded(rows, 30)
    out = []
    for c in cases:
        gold = c["answer"].split("####")[-1].strip().replace(",", "").replace("$", "")
        prompt = c["question"].strip() + "\n\n请一步步推理，最后一行单独输出最终数值答案，格式: #### 数字"
        txt, met = ask(prompt)
        nums = re.findall(r"-?[\d,]*\d", (txt.split("####")[-1] if "####" in txt else txt).replace("$", ""))
        got = nums[-1].replace(",", "") if nums else ""
        out.append(dict(ok=got == gold, gold=gold, got=got, **met))
    return out

# ---------- 室2 HumanEval ----------
def humaneval_room():
    rows = [json.loads(l) for l in gzip.open(f"{BASE}/data/HumanEval.jsonl.gz")]
    cases = seeded(rows, 20)
    out = []
    for c in cases:
        prompt = (c["prompt"].strip() +
                  "\n\n只补全这个 Python 函数的实现，输出一个 ```python 代码块，不要解释。")
        txt, met = ask(prompt)
        m = re.search(r"```(?:python)?\s*\n(.*?)```", txt, re.S)
        code = m.group(1) if m else txt
        full = c["prompt"] + "\n" + code + "\n\n" + c["test"]
        try:
            r = subprocess.run(["python3", "-c", full], capture_output=True, timeout=15, text=True)
            ok = r.returncode == 0
        except Exception:
            ok = False
        out.append(dict(ok=ok, **met))
    return out

# ---------- 室3 IFEval（可验证指令子集） ----------
def ifeval_checks(inst, kw, txt):
    t = txt.strip()
    if inst == "min_words": return len(t.split()) >= int(kw["num_words"]) * 0.95
    if inst == "max_words": return len(t.split()) <= int(kw["num_words"]) * 1.05
    if inst == "json_format":
        s = t[t.find("{"):t.rfind("}") + 1]
        try: json.loads(s); return True
        except Exception: return False
    if inst == "lowercase": return t == t.lower()
    if inst == "uppercase": return t == t.upper()
    if inst == "end_phrase": return t.lower().endswith(kw["end_phrase"].strip().lower())
    if inst == "keywords:existence": return all(k.lower() in t.lower() for k in kw.get("keywords", []))
    if inst == "keywords:frequency":
        w = re.findall(r"\b" + re.escape(kw["keyword"]) + r"\b", t, re.I)
        rel = kw.get("relation", "at least")
        n = int(kw.get("frequency", 0))
        return {"at least": len(w) >= n, "less than": len(w) < n}.get(rel, False)
    if inst == "forbidden_words":
        return not any(w.lower() in t.lower() for w in kw.get("forbidden_words", []))
    if inst == "placeholders":
        return not re.search(r"\[.*?\]", t)
    return None

SUPPORTED = {"min_words", "max_words", "json_format", "lowercase", "uppercase", "end_phrase",
             "keywords:existence", "keywords:frequency", "forbidden_words", "placeholders"}

def ifeval_room():
    rows = [json.loads(l) for l in open(f"{BASE}/data/ifeval_data.jsonl")]
    usable = [r for r in rows if all(i in SUPPORTED for i in r["instruction_id_list"])]
    cases = seeded(usable, 15)
    out = []
    for c in cases:
        txt, met = ask(c["prompt"])
        oks = [ifeval_checks(i, kw, txt) for i, kw in zip(c["instruction_id_list"], c["kwargs"])]
        out.append(dict(ok=all(oks), n_checks=len(oks), **met))
    return out

# ---------- 室4 JSON 结构化 ----------
SCHEMAS = [
    (" Extract the event into JSON with keys: name(string), year(integer), city(string). Output ONLY the JSON object.", 1),
    (" Given a person's age and name, output JSON {\"name\": string, \"age\": integer, \"adult\": boolean}. ONLY JSON.", 2),
    (" Output a JSON object {\"items\": [array of 3 strings], \"total\": integer equals array length}. ONLY JSON.", 3),
    (" Classify the sentiment into JSON {\"label\": one of [\"pos\",\"neg\",\"neutral\"], \"confidence\": number 0-1}. ONLY JSON.", 4),
]
JSON_TOPICS = ["the launch of the Voyager 1 probe in 1977 from Cape Canaveral",
               "Marie Curie winning the Nobel Prize in Physics in 1903 in Stockholm",
               "the premiere of Beethoven's Ninth Symphony in 1824 in Vienna",
               "the founding of Singapore as a trading post in 1819 by Raffles",
               "the first Moon landing by Apollo 11 in 1969 near Mare Tranquillitatis"]

def json_room():
    out = []
    for i in range(20):
        spec, kind = SCHEMAS[i % len(SCHEMAS)]
        topic = JSON_TOPICS[i % len(JSON_TOPICS)]
        txt, met = ask("关于 " + topic + "，" + spec)
        s = txt[txt.find("{"):txt.rfind("}") + 1]
        try:
            j = json.loads(s)
            ok = isinstance(j, dict)
            if kind == 1: ok &= all(k in j for k in ("name", "year", "city"))
            if kind == 2: ok &= isinstance(j.get("age"), int) and isinstance(j.get("adult"), bool)
            if kind == 3: ok &= isinstance(j.get("items"), list) and len(j["items"]) == 3 and j.get("total") == 3
            if kind == 4: ok &= j.get("label") in ("pos", "neg", "neutral") and isinstance(j.get("confidence"), (int, float))
        except Exception:
            ok = False
        out.append(dict(ok=bool(ok), kind=kind, **met))
    return out

ROOMS = {"gsm8k": gsm8k_room, "humaneval": humaneval_room, "ifeval": ifeval_room, "json": json_room}

for name, fn in ROOMS.items():
    if ONLY and name not in ONLY: continue
    t0 = time.time()
    res = fn()
    score = sum(1 for r in res if r["ok"])
    think = [r.get("tok_total", 0) - round(r.get("content_chars", 0) / 3.2) for r in res]
    rep = {"tag": TAG, "suite": name, "score": score, "n": len(res),
           "think_tokens_avg": round(sum(max(t, 0) for t in think) / max(len(res), 1)),
           "finish_breakdown": {f: sum(1 for r in res if r.get("finish") == f)
                                 for f in {r.get("finish") for r in res}},
           "seconds": round(time.time() - t0), "cases": res}
    with open(f"{OUT}/{TAG}-{name}.json", "w") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(f"{TAG} {name}: {score}/{len(res)} think_avg={rep['think_tokens_avg']} ({rep['seconds']}s)")
