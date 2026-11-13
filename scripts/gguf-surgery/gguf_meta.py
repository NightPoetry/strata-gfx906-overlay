import sys
from gguf import GGUFReader

path = sys.argv[1]
r = GGUFReader(path)
for k in sorted(r.fields):
    if any(s in k.lower() for s in ("nextn", "block_count", "expert_count", "context")):
        f = r.fields[k]
        try:
            v = f.parts[-1]
            print(" ", k, "=", list(v)[:4])
        except Exception as e:
            print(" ", k, "<err>", e)
names = [t.name for t in r.tensors]
nxt = [n for n in names if "nextn" in n.lower() or ".40." in n or n.startswith("blk.40")]
print("tensors:", len(names), "| blk40/nextn:", nxt[:6])
print("last tensors:", names[-3:])
