import sys
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType, GGMLQuantizationType
from gguf.quants import dequantize

MAIN = "/data/ai/LoadedModel/occamy-1.0/occamy-iq3s.gguf"
DRAFT = "/data/ai/LoadedModel/occamy-1.0/occamy-mtp-q4.gguf"
OUT = "/data/ai/LoadedModel/occamy-1.0/occamy-iq3s-mtp.gguf"

# The engine uploads these as F32: they must be F32/BF16/F16/Q8_0 in the file
F32_REQUIRED = {
    "attn_k_norm.weight", "attn_norm.weight", "attn_q_norm.weight",
    "ffn_gate_inp_shexp.weight", "ffn_gate_inp.weight",
    "nextn.enorm.weight", "nextn.hnorm.weight", "nextn.shared_head_norm.weight",
    "post_attention_norm.weight", "ssm_alpha.weight", "ssm_beta.weight",
    "ssm_conv1d.weight", "ssm_dt.bias", "ssm_norm.weight",
}
OK_TYPES = {GGMLQuantizationType.F32, GGMLQuantizationType.F16,
            GGMLQuantizationType.BF16, GGMLQuantizationType.Q8_0}

rm = GGUFReader(MAIN)
rd = GGUFReader(DRAFT)

w = GGUFWriter(OUT, "qwen35moe")

for k, f in rm.fields.items():
    if k == "general.architecture" or k.startswith("GGUF."):
        continue
    if k == "qwen35moe.block_count":
        w.add_key_value(k, 41, GGUFValueType.UINT32)
        continue
    if k == "qwen35moe.nextn_predict_layers":
        continue
    w.add_key_value(k, f.contents(), f.types[0])

w.add_key_value("qwen35moe.nextn_predict_layers", 1, GGUFValueType.UINT32)

def tensor_stream():
    for t in rm.tensors:
        yield t
    for t in rd.tensors:
        if t.name.startswith("blk.40."):
            yield t

info = []
payload = []
req_deq = 0
for t in tensor_stream():
    needs_f32 = any(t.name.endswith(n) for n in F32_REQUIRED)
    if needs_f32 and t.tensor_type not in OK_TYPES:
        f32 = dequantize(t.data, t.tensor_type).astype(np.float32)
        w.add_tensor_info(t.name, f32.shape, np.dtype(np.float32), f32.nbytes, GGMLQuantizationType.F32)
        payload.append(f32)
        req_deq += 1
    else:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
        payload.append(t.data)
    info.append(t.name)
print("tensors:", len(info), "| dequantized to F32:", req_deq)

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_ti_data_to_file()

for arr in payload:
    w.write_tensor_data(arr, tensor_endianess=rm.endianess)

w.close()
print("MERGE_DONE deq=", req_deq)
