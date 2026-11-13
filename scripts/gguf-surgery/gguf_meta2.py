import struct, sys

path = sys.argv[1]
f = open(path, "rb")
assert f.read(4) == b"GGUF"
ver, n_t, n_kv = struct.unpack("<IQQ", f.read(20))

def rstr():
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")

TYPE_SIZE = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}

def skip_payload(t):
    if t == 8: rstr(); return
    f.seek(TYPE_SIZE[t], 1)

def rval(t):
    if t == 8: return rstr()
    if t == 9:
        it = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        if it == 8:
            out = []
            for _ in range(min(n, 2)):
                out.append(rstr())
            for _ in range(n - min(n, 2)):
                ln = struct.unpack("<Q", f.read(8))[0]
                f.seek(ln, 1)
            return out + [f"({n})"] if n > 2 else out
        f.seek(TYPE_SIZE[it] * n, 1)
        return f"<arr {n} x t{it}>"
    fmt = {0:"<B",1:"<b",2:"<H",3:"<h",4:"<I",5:"<i",6:"<f",7:"<?",10:"<Q",11:"<q",12:"<d"}[t]
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

for _ in range(n_kv):
    k = rstr(); t = struct.unpack("<I", f.read(4))[0]
    try:
        v = rval(t)
    except Exception as e:
        print("STOP", k, e); sys.exit(0)
    if any(s in k.lower() for s in ("nextn", "block_count", "file_type", "general.name")):
        print(k, "=", v)
print("meta OK; meta_end_offset:", f.tell())

# tensor infos: name + type only
TENSOR_TYPES = {}
for i in range(100):
    try:
        n = rstr()
    except Exception:
        break
print("(skip tensor detail walk)")
