import struct, sys

path = sys.argv[1]
f = open(path, "rb")
assert f.read(4) == b"GGUF"
ver, n_t, n_kv = struct.unpack("<IQQ", f.read(20))

def rstr():
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")

TYPE_SIZE = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}

def skip_val(t):
    if t == 8: rstr(); return
    if t == 9:
        it = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        if it == 8:
            for _ in range(n):
                ln = struct.unpack("<Q", f.read(8))[0]
                f.seek(ln, 1)
        else:
            f.seek(TYPE_SIZE.get(it,8) * n, 1)
        return
    f.seek(TYPE_SIZE.get(t,8), 1)

for _ in range(n_kv):
    k = rstr(); t = struct.unpack("<I", f.read(4))[0]
    skip_val(t)

TYPES = {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",8:"Q8_0",10:"Q2_K",11:"Q3_K_S",12:"Q3_K_M",13:"Q3_K_L",14:"Q4_K_S",15:"Q4_K",16:"Q5_K",17:"Q6_K",18:"Q8_K",19:"IQ2_XXS",20:"IQ2_XS",21:"IQ2_S",22:"IQ3_XXS",23:"IQ1_S",24:"IQ4_NL",25:"IQ3_S",26:"IQ2_M",27:"IQ4_S",28:"IQ1_M",29:"BF16",30:"?30",40:"NVFP4?",42:"?42"}

def rdim():
    return struct.unpack("<Q", f.read(8))[0]

shown = 0
import sys as _s
for i in range(n_t):
    name = rstr()
    nd = struct.unpack("<I", f.read(4))[0]
    dims = [rdim() for _ in range(nd)]
    ttype = struct.unpack("<I", f.read(4))[0]
    f.seek(8, 1)  # offset (u64) — skip
    if "exp" in name and ("blk.0." in name or "blk.40." in name):
        print(name, "type:", TYPES.get(ttype, ttype), "dims:", dims)
        shown += 1
print("done")
