# strata-gfx906-overlay — 在 AMD MI50（gfx906）上运行 Strata 富服务器的移植套件

这个仓库提供一套 overlay（覆盖分发包）。它的作用是：把 [Strata](https://github.com/Niko1221/Strata)（MIT 许可）的全功能 qwen35moe 服务路径（PolyStrata 富服务器）带到 AMD Instinct MI50 16G 显卡上，在 ROCm 6.3 下编译并运行。这项移植是必要的，原因有两点：上游从未为 MI50 编译过 HIP 版本；并且 ROCm 6.4 起官方删除了 gfx906 编译目标。

版权归属：上游 Strata 采用 MIT 许可，版权归 Niko1221 与 Strata 贡献者所有。本仓库在其之上新增了 HIP/CUDA 兼容层和编译、路由开关；新增部分由 FasFox 以 MIT 许可发布。详见 LICENSE 与 NOTICE。

## 仓库里有什么

```
overlay/    上游被改动文件的完整替换副本，以及新增的 include/strata/hip_compat 等
            头文件。用法：把 overlay/ 的内容复制到一份干净的 Strata 源码树上。
diffs/      两个未保留原始副本的文件的统一 diff（CMakeLists.txt 与 generate.cpp），
            便于逐行审阅改动。
configs/    MI50 16G 的生产配置：UD-Q4_K_M 主模型加独立 MTP 草稿，262K 上下文，
            KV 用 q8_0，热专家驻留用 --kv-whole 固定。
scripts/    启动脚本、量化基准测试套件，以及 gguf-surgery 工具（检查 gguf 元
            数据、摘除或合并草稿层）。
docs/       移植笔记与性能实测。
MANIFEST.sha256        overlay 每个文件的 sha256 校验清单。
MANIFEST.modelhashes   生产模型文件的字节数与 sha256 钉版。
```

## 为什么上游跑不了 gfx906，而这个 overlay 可以

上游有两处限制。第一处是构建脚本：CMakeLists.txt 在启用 STRATA_HIP 时会把富服务器目标排除掉（共三段，约在 2050、2073、2081 行附近）。第二处是路由代码：generate.cpp 里的 qwen35moe 家族路由会把 qwen35moe 模型派给一个功能缩水的侧引擎。本移植做了三件事来打开这条路：打开上述开关；补一层 HIP/CUDA 兼容层（驱动 VMM shim、cublas 映射、fp16 与内联函数、数学常量头文件）；用 HIP 语言重编 18 个 MMQ 量化内核。另外，MXFP4 路径被主动排除，因为它是 Blackwell FP4 MMA 单元专用路径，gfx906 没有这个单元。完整移植经过见 docs/PORT-NOTES-gfx906.md。

## 在你自己的机器上跑起来

完整的分步指南（ROCm 6.3 安装、模型下载地址与校验、配置逐项解释、启动、API 调用、常见问题）见 **[docs/USAGE.zh-CN.md](docs/USAGE.zh-CN.md)**。最短路径如下：

```bash
cp -r overlay/* /path/to/strata/          # 应用 overlay
cmake -B build -DSTRATA_HIP=ON -DSTRATA_HIP_GFX906=ON && cmake --build build -j
# 按 USAGE §3 下载主模型（unsloth UD-Q4_K_M，字节与 sha256 已钉版）和 MTP 草稿
python -m serve.server --engine strata --config occamy-main-q4.json --port 8099
```

## 应用 overlay 的步骤

```bash
# 上游源码，例如：git clone https://github.com/Niko1221/Strata
cp -r overlay/* /path/to/strata/
# 用 MANIFEST.sha256 校验覆盖结果，然后编译：
cmake -B build -DSTRATA_HIP=ON -DSTRATA_HIP_GFX906=ON <你常用的其他开关>
cmake --build build -j$(nproc)
```

ROCm 必须用 6.3（6.4 起没有 gfx906）。本套件在 MI50 16G 上编译并用 `strata-poly-rich` 实测通过；生产配置在 configs/。

## 实测结果（MI50 16G，262K 上下文）

| 指标 | 数值 |
|---|---|
| 解码（UD-Q4_K_M + MTP 草稿，短 prompt） | 65.5 tok/s |
| prefill（14.7K prompt，staging 调优后） | 814.6 tok/s |
| 热专家驻留（--kv-whole 固定） | 49% |
| 量化三档质量（Q2/Q4/Q6，85 题） | 82/84/83——噪声带内 |
| 满窗 prefill（253K token） | 265 tok/s |

测试方法和由此形成配置选择的几个失败模式（kv_grow 缓存挤压、Q6 草稿饥荒、staging 空窗）见 docs/PERF.md。
