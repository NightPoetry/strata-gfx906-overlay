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

> **先澄清一个容易踩的坑**：主模型是**基座 Qwen3.6-35B-A3B** 的 Unsloth UD 动态量化件——不是 Accio-Lab 的 Occamy-1.0 微调件。生产环境的存放目录恰好叫 `occamy-1.0/`（本地路径习惯），但文件元数据（`general.basename` = `Qwen3.6-35B-A3B`，`quantized_by` = `Unsloth`，`base_model` = `Qwen/Qwen3.6-35B-A3B`）才是身份凭证。下错微调件的话架构相同能跑，但复现的就不再是本文档的性能与配置。

### 3.1 主模型（约 20.6 GB）

| 项 | 值 |
|---|---|
| 文件 | `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` |
| 尺寸钉版 | 22,134,528,992 字节 |
| 来源 | [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF)（Q4 是实测甜点档，依据见 docs/PERF.md） |
| 国内镜像 | `https://hf-mirror.com/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` |

```bash
curl -L -C - -o Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  "https://hf-mirror.com/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
```

### 3.2 MTP 草稿（旁挂，约 1.2 GB）

投机解码需要一个独立 nextn 草稿文件（`--draft-model`）。三条路任选：

1. **社区现成件**：[Accio-Lab/occamy-1.0-MTP](https://huggingface.co/Accio-Lab/occamy-1.0-MTP)（官方）或 [fetchingcat/occamy-1.0-mtp-gguf](https://huggingface.co/fetchingcat/occamy-1.0-mtp-gguf)——qwen35moe 的 nextn 形状通用，q4/q8 档均可；
2. **从主文件自提**：若你的主文件带草稿层（见 3.3 自检），用 `scripts/gguf-surgery/` 工具摘出 blk.40 并按 q4 打包；
3. **我们生产用的确切文件**：`occamy-mtp-q4.gguf`，1,260,897,824 字节，sha256 见 MANIFEST.modelhashes（不在本仓库分发，20GB 级模型不入 git）。

### 3.3 下载后自检（必做）

镜像 LFS 偶发飘移（同一路径服务非钉版对象，实测踩过），下载完先验身再跑：

```bash
# ① 字节数与 sha256 对表（主模型 22,134,528,992 字节；sha256 见 MANIFEST.modelhashes）
stat -c %s Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
sha256sum Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
# ② 元数据自检（scripts/gguf-surgery/gguf_meta.py）
python3 gguf_meta.py Qwen3.6-35B-A3B-UD-Q4_K_M.gguf | grep -E 'general.basename|block_count'
```

判读口径：

- `general.basename` = `Qwen3.6-35B-A3B`——是本文档的模型；若是 `occamy-1.0` 则是 Accio-Lab 微调件（能跑但不是本档）；
- `qwen35moe.block_count` = 41 且有 nextn 键——自带草稿层，可旁挂也可直用；
- `block_count` = 40 且无 nextn 键——草稿层被裁（Unsloth 现行版如此），**必须旁挂 3.2 的草稿**；
- 张量声明尺寸与文件不符（报 `does not fit its file`）——镜像飘移件，换源重下，或用 `strip40.py` 手术摘掉坏层后旁挂（我们生产走过的路）。

### 3.4 备选量化（不推荐但可用）

- [mradermacher/occamy-1.0-GGUF](https://huggingface.co/mradermacher/occamy-1.0-GGUF) Q6_K：16G 上无可用草稿、无投机（诚实口径 20.9 tok/s），质量与 Q4 几乎打平，不建议；
- [Accio-Lab/occamy-1.0-GGUF](https://huggingface.co/Accio-Lab/occamy-1.0-GGUF) 官方平量化：同样无 nextn 草稿层，需旁挂。

### 3.5 放置

放到如 `/data/ai/LoadedModel/occamy-1.0/`（目录名随意，与 config 同步即可）。

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
