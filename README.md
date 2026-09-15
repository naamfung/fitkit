# fitkit — FIT-GGUF 量化工具集（纯 Go 移植版）

[fitkit](https://github.com/naamfung/fitkit) 是以 [FIT-GGUF](https://github.com/Scorp1o117/FIT-GGUF) 算法为核心的量化工具集；本仓库是其**纯 Go 移植版**，命令名 `fitting`：在 Go 内完成 **analyze → plan → quantize** 全流程，只调用 llama.cpp 运行时（`llama-quantize` / `llama-imatrix` / `llama-perplexity`），**不依赖 Python**。

移植基于上游 0.2.0 算法，并吸收 0.3.x 的行为改进（KL-only 保真门限、可命名 preset 的产物命名、按模型的 reference manifest 发现、calibrate/registry 产线等），同时保留 Go 版自有的扩展（up/down 双向优化、多级阶梯候选、并行评估）。

## 特性

- **analyze → plan → quantize 一键量化**：冻结 analysis.json，按目标字节精确规划（G2 字节精确门限，`actual == expected`）
- **双向优化**：`-mode up`（下界为基线、升格高价值张量）/ `-mode down`（上界为基线、从高品质向下削减）
- **多级阶梯候选**：默认沿 curated ladder 逐档过渡张量（`-no-ladder` 可回退为二进位直接转换）
- **down 模式逐档回滚**：以阶梯单步粒度回滚最有价值张量，充分利用剩余预算（避免整张量粒度过粗造成的预算浪费）
- **Fidelity Contract v2（KL-only 门限）**：`PASS = macro KL ≤ 全局锚点`；guard profile 可选，仅作为 same-top 参考
- **`fit calibrate` 校准产线**：imatrix 生成 → 5 域参考构建 → 标准阶梯 → gap probes → P5 floor 推导 → Calibration Bundle（guard profile、reference manifest、seed material、registry entry）
- **Fidelity Registry v1**：按 source weights SHA-256 键控的信任根，`list/show/verify/validate` 全量结构 + 哈希 + 交叉引用校验
- **Refine Profile**：C_role / band-conditional 效用加权，可调候选排序
- **产物命名规则**：后缀保证是可识别的 GGUF preset（无裸 `Q3_K/Q4_K/Q5_K` 张量类型）
- **平台感知**：Windows `.exe/.cmd/.bat` 二进制解析、CUDA runtime 兄弟目录自动发现（避免静默 CPU 评估）

## 命令

| 命令 | 说明 |
| --- | --- |
| `fitting` | analyze → plan → quantize 主入口（含 `-mode`、`-fit` 比例目标、`-fidelity-tier`） |
| `fitfidelity` | fidelity-tier 产品路径：搜索满足 KL 锚点的最小通过产物并生成 |
| `fitcalibrate` | 校准产线：生成 Calibration Bundle（`--replay-existing` 支持零评估重推导） |
| `fitregistry` | Fidelity Registry v1 只读工具：`list` / `show` / `verify` / `validate` |
| `fitdry` / `fitpoc` / `plancheck` | 内部验证工具（provenance 检查、字节预测校验、plan 对拍） |

## 快速上手

```powershell
# 编译
./build.sh          # Git Bash / WSL（POSIX shell）；或逐个：go build -o bin/fitting ./cmd/fitting

# 一键量化：源 BF16 → 目标 4.88 GiB，Q3_K_L→F16 窗口，向下优化
fitting -source model-bf16.gguf -imatrix im.gguf `
        -target 4.88GiB -lower Q3_K_L -upper F16 -mode down

# 比例目标（窗口间距的 1/3）
fitting -source model-bf16.gguf -imatrix im.gguf -fit 1/3 -upper BF16

# 只规划、看类型份额
fitting -source model-bf16.gguf -imatrix im.gguf -target 4.88GiB -plan-only

# 保真产品路径（Quality 档）
fitfidelity -source model-bf16.gguf -imatrix im.gguf -runtime <llama-bin> `
            -refs-dir <bf16 refs> -eval-data-dir <slices> `
            -tier quality -preset-ladder IQ4_XS,Q6_K,Q8_0 -out-dir out -work-dir work `
            -manifest manifest.txt -logs-dir logs -freeze <FREEZE.json>

# 校准产线
fitcalibrate -source model-bf16.gguf -imatrix-corpus corpus.txt `
             -runtime <llama-bin> -eval-data <slices> -out-dir bundle -model-id <id>

# Registry 验证
fitregistry -root <package-dir> verify
```

完整参数见 `fitting -h` 及各命令帮助；`fitting` 命令的详细使用说明见 [cmd/fitting/README.md](cmd/fitting/README.md)。

## 目录结构

```
fitting/
  cmd/            # 各命令入口（fitting、fitfidelity、fitcalibrate、fitregistry、…）
  fidelity/       # 保真搜索状态机、guard profile、eval 执行器、产物路径
  pipeline/       # analyze/plan/quantize、候选生成、优化器、refine、契约输出
  gguf/           # GGUF 布局读取与尺寸预测
  calibration/    # 校准契约（fidelity-calibration-v1）与执行产线
  registry/       # Fidelity Registry v1（digest、verify、validate）
```

## 与 Python 原版的关系

- 移植基线为上游 **0.2.0** 算法，并吸收 **0.3.x** 的改进：KL-only 保真门限、`primary_type_from_plan` 命名规则、窗口边界选择、按模型 manifest 发现、无条件 source↔reference 绑定、`fit calibrate` / `fit registry` 产线。
- Go 版扩展：`-mode up|down` 双向优化、多级阶梯候选（`-no-ladder` 可关）、`-eval-parallel` 并行域评估、float baseline 的 `Q8_0` 名义 ftype + 全量张量覆盖方案。
- 构建：`go build ./...`，Go 1.21+；唯一第三方依赖 `gopkg.in/yaml.v3`。

## 参考

- 原始项目介绍（中文）：[FIT-GGUF README.zh-CN.md](https://github.com/Scorp1o117/FIT-GGUF/blob/main/README.zh-CN.md)
