import sys
from gguf import GGUFReader, GGUFWriter, GGUFValueType

SRC = "/data/ai/LoadedModel/occamy-1.0/UD-Q2_K_XL-pinned.gguf"
OUT = "/data/ai/LoadedModel/occamy-1.0/UD-Q2_K_XL-nodraft.gguf"

r = GGUFReader(SRC)
w = GGUFWriter(OUT, "qwen35moe")

for k, f in r.fields.items():
    if k == "general.architecture" or k.startswith("GGUF."):
        continue
    if k == "qwen35moe.block_count":
        w.add_key_value(k, 40, GGUFValueType.UINT32)
        continue
    pass
    w.add_key_value(k, f.contents(), f.types[0])

w.add_key_value("qwen35moe.nextn_predict_layers", 1, GGUFValueType.UINT32)

info = []
for t in r.tensors:
    if t.name.startswith("blk.40."):
        continue
    w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    info.append(t.name)
print("tensors kept:", len(info))

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_ti_data_to_file()
for t in r.tensors:
    if t.name.startswith("blk.40."):
        continue
    w.write_tensor_data(t.data, tensor_endianess=r.endianess)
w.close()
print("STRIP_DONE")
