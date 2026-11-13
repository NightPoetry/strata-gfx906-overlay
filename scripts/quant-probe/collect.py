import json, glob, os
os.chdir("/path/to/quant-probe/results")
rows = {}
for f in sorted(glob.glob("*-speed.json")):
    d = json.load(open(f))
    t = d["tag"]
    rows.setdefault(t, {}).update(short=d["short_median_toks"], prefill=d["long_prefill_toks"],
                                  ldecode=d["long_decode_toks"], draft=d["long_draft"],
                                  plen=d["long_prompt_n"])
for f in sorted(glob.glob("*-bootline.txt")):
    t = f.split("-")[0]
    rows.setdefault(t, {})["boot"] = open(f).read().strip()[:110]
for f in sorted(glob.glob("mtx*-*.json")):
    if "speed" in f:
        continue
    d = json.load(open(f))
    t, s = d["tag"], d["suite"]
    rows.setdefault(t, {})[s] = "{}/{}".format(d["score"], d["n"])
    rows[t][s + "_think"] = d.get("think_tokens_avg")
for t in ["q2", "q4", "q6", "q6draft"]:
    if t in rows:
        print(t, json.dumps(rows[t], ensure_ascii=False))
