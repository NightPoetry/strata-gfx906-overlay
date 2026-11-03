# 使用指南——在自己的电脑上跑起来（zh-CN）

从零开始：一张 AMD MI50 16G 显卡，到跑通 262K 上下文的推理服务。全程约一小时（大部分时间是编译和模型下载）。

## 0. 硬件与系统要求

| 项 | 要求 | 说明 |
|---|---|---|
| 显卡 | AMD MI50 / MI50 II 16G（gfx906） | 16G 版是本方案的适用对象 |
| ROCm | **6.3.x** | ROCm 6.4 起官方移除了 gfx906，不要用 6.4+ |
| 内存 | 建议 32G | kv-whole 预划 + 检查点 + 会话泊车都在主机内存 |
| 磁盘 | ≥30G 空闲 | Q4 主文件约 21.5G + MTP 草稿 1.2G |
| 系统 | Linux（x86-64 主机） | 参考环境：X99 平台 + MI50 |

## 1. 安装 ROCm 6.3

按 AMD 官方文档安装 ROCm 6.3.x。装完验证两项：

```bash
rocminfo | grep gfx906          # 能看到 gfx906
ls /sys/class/drm/card0/device/gpu_busy_percent   # 占用率接口存在
```

## 2. 获取源码并应用 overlay

```bash
git clone https://github.com/Niko1221/Strata strata
cp -r <本仓库>/overlay/* strata/          # 整目录覆盖
sha256sum -c <(sed 's|overlay/|strata/|' <本仓库>/MANIFEST.sha256)   # 校验一致性
cd strata
cmake -B build -DSTRATA_HIP=ON -DSTRATA_HIP_GFX906=ON <你常用的其他开关>
cmake --build build -j$(nproc)
```

产物 `strata-poly-rich` 是全功能富服务器（build/ 下）。

## 3. 准备模型

1. 主模型：Qwen3.6-35B-A3B 的 **UD-Q4_K_M** 动态量化件（Q4 是实测甜点档，见 docs/PERF.md）。
2. MTP 草稿：`occamy-mtp-q4.gguf` 一类的独立草稿件（约 1.2G），从原版权重提取打包。
3. **检查社区量化件的草稿层**：社区转换管线会静默裁掉或损坏 nextn 草稿层。用 gguf 元数据自查——`qwen35moe.block_count` 应为 41 且有 nextn 键；若为 40 且无 nextn 键=草稿被裁（旁挂独立草稿即可工作）；若张量声明尺寸与文件不符=损坏（需手术摘掉 blk.40，再旁挂）。
4. 把模型放到如 `/data/ai/LoadedModel/occamy-1.0/`。

## 4. 配置与启动

`configs/occamy-main-q4.json` 就是生产配置，逐项含义：

| 参数 | 含义 |
|---|---|
| `--kv q8_0` + `--kv-whole` | KV 用 q8_0 减半显存，且装载时一次给足全窗——热专家驻留从此固定不被挤（本方案的核心规矩） |
| `--max-context 262144` | 满窗 26 万 token |
| `--threads 8` | 等于物理核数（不是超线程数） |
| `--expert-profile` | 热专家画像暖启动：装载即把常用专家驻进显存（约 49%） |
| `--prompt-cache 24 --prompt-cache-every 4096` | 检查点存主机内存，切回旧会话快速恢复 |
| `--conversation-cache-slots 8 --conversation-cache-mib auto` | 会话泊车（内存），多会话秒回 |

启动：

```bash
cd /data/ai/poly
.venv/bin/python -m serve.server --engine strata \
  --config occamy-main-q4.json --port 8099
```

首次装载约 100 秒（kv-whole 预划全窗 KV 属正常），之后常驻。

## 5. 验证与调用

```bash
curl http://127.0.0.1:8099/health
# {"status":"ok","max_context":262144,...,"loaded":true}

curl http://127.0.0.1:8099/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"occamy","messages":[{"role":"user","content":"你好"}],"max_tokens":100}'
```

OpenAI 兼容接口，任何标准客户端都能直连。响应里的 `timings` 字段带服务端计时（prompt_per_second / predicted_per_second / draft 接受率）。

## 6. 预期性能（MI50 16G 实测）

| 场景 | 数值 |
|---|---|
| 解码（短 prompt，含 MTP） | 65.5 tok/s |
| prefill（14.7K prompt） | 814 tok/s |
| 满窗 prefill（253K） | 265 tok/s |
| 热专家驻留 | 49%（固定） |

## 7. 常见问题

- **闲置后第一发略慢**：冷专家需要从内存搬进显存，属结构性现象；热专家驻留已固定，只发生一次，之后正常。
- **装载比别的引擎慢**：kv-whole 预划全窗 KV，一次性代价换运行期稳定。
- **prefill 有锯齿**：同上，每批之间搬冷专家，占空比实测 83%（大灌入）/26%（小请求），都正常。
- **想用 Q6**：16G 上无可用草稿，诚实口径是无草稿 20.9 tok/s；质量与 Q4 几乎打平，不建议。
- **显存不够 / 专家被挤**：检查是否漏了 `--kv-whole`；`--prompt-batch` 别超过 1536（再大会挤专家工作区）。
