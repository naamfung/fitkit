# fitting — 纯 Golang 版 FIT-GGUF

`fitting` 在纯 Go 内完成 **analyze → plan → quantize** 全流程，仅调用 `llama-quantize`，**不依赖 Python**。

## 用法

```
fitting -source <BF16.gguf> -imatrix <im.gguf> -target <size> [options]
```

## 参数

| 参数                  | 必填     | 默认值             | 说明                                               |
| ------------------- | ------ | --------------- | ------------------------------------------------ |
| `-source`           | ✅      | —               | BF16/F16 源 GGUF 路径                               |
| `-imatrix`          | ✅      | —               | 重要性矩阵 .gguf 路径                                   |
| `-target`           | ✅      | —               | 目标体积，见"体积格式"                                     |
| `-lower`            | <br /> | `Q3_K_M`        | 下界预设（窗口最小类型）                                     |
| `-upper`            | <br /> | `Q8_0`          | 上界预设（窗口最大类型）                                     |
| `-policy`           | <br /> | `balanced`      | 规划策略：`balanced` / `original`(greedy) / `random`  |
| `-mode`             | <br /> | `up`            | 优化方向：`up`（下界为基线、升格高价值张量）/ `down`（上界为基线、从高品质向下削减） |
| `-no-ladder`        | <br /> | `false`         | 禁用多级阶梯候选生成（仅使用二进位候选：upper→lower直接转换）             |
| `-out`              | <br /> | 自动              | 输出路径；省略则自动命名                                     |
| `-runtime`          | <br /> | 本机 build-v17 路径 | llama.cpp 二进制目录                                  |
| `-plan-only`        | <br /> | `false`         | 只规划并打印类型份额，不做量化                                  |
| `-allow-requantize` | <br /> | `false`         | 允许以已量化母本（如 Q8\_0）作为 `-source` 继续量化               |
| `-keep`             | <br /> | `false`         | 保留临时工作目录                                         |

## 体积格式（`-target`）

- 后缀单位：`5GiB`、`5GB`、`512MiB`、`512MB`（`GiB`=1024³，`GB`=1000³）

- 裸 `G`：`5G` = 5×1024³

- 裸数字：视为字节，如 `5368709120`

## 输出自动命名

`<模型名>-FITKIT-<体积>.2fG-<主类型>-lower<下界>-upper<上界>.gguf`

例：`Qwen3.5-9B-...-FITKIT-5.50G-Q4_K-lowerQ4_K_S-upperF16.gguf`

## 产物目录结构

GGUF 与全部产物统一放入以输出名（去 `.gguf`）命名的子目录，避免散落：

```
<stem>/
  <stem>.gguf
  <stem>-analysis.json
  <stem>-profile.json
  <stem>-plan.json
  <stem>-recipe.json
  <stem>-tensor-types.txt
  <stem>.gguf.quantize-record.json
```

可用 `-no-artifacts` 关闭结构产物（仅输出 GGUF）。

## 可选类型（`-lower` / `-upper`）

已完整对齐 llama-quantize：

- **浮点**：`F32` `F16` `BF16`

- **标量量化**：`Q4_0` `Q4_1` `Q5_0` `Q5_1` `Q8_0`

- **新型**：`Q1_0` `Q2_0` `MXFP4_MOE`

- **TQ/三元**：已全部停用（`TQ1_0` `TQ2_0` `TQ3_1S` `TQ4_1S`）

- **K 系列**：`Q2_K` `Q2_K_S` `Q3_K_S` `Q3_K_M` `Q3_K_L` `Q4_K_S` `Q4_K_M` `Q5_K_S` `Q5_K_M` `Q6_K`（`Q3_K`/`Q4_K`/`Q5_K` 为别名→M）

- **IQ 系列**：`IQ1_S` `IQ1_M` `IQ2_XXS` `IQ2_XS` `IQ2_S` `IQ2_M` `IQ3_XXS` `IQ3_XS` `IQ3_S` `IQ3_M` `IQ4_NL` `IQ4_XS`

## 示例

```powershell
# 出 4.88GiB，Q3_K_M→BF16 窗口，向上取值（默认 mode=up）
fitting -source 9b-model.gguf -imatrix im.gguf -target 4.88GiB -upper BF16

# 出 4.88GiB，向上取值模式（下界为基线，升格高价值张量）
fitting -source 9b-model.gguf -imatrix im.gguf -target 4.88GiB -lower IQ4_XS -upper F16 -mode up

# 出 4.88GiB，向下取值模式（上界 F16 为基线，从高品质向下削减）
fitting -source 9b-model.gguf -imatrix im.gguf -target 4.88GiB -lower IQ4_XS -upper F16 -mode down

# 只生成计划、看类型份额，不量化
fitting -source 9b-model.gguf -imatrix im.gguf -target 4.88GiB -lower IQ4_XS -upper F16 -mode down -plan-only

# 禁用多级阶梯候选生成（仅使用二进位候选：upper→lower直接转换）
fitting -source 9b-model.gguf -imatrix im.gguf -target 4.88GiB -lower IQ1_S -upper BF16 -mode down -no-ladder

# 指定输出路径
fitting -source 9b-model.gguf -imatrix im.gguf -target 6GiB -out D:\out\m.gguf
```

## 退出码

- `0`：G2 字节精确 PASS

- `1`：G2 FAIL（`actual != expected`）或运行错误

- `2`：参数缺失 / 目标超出窗口 `[下界, 上界]`

运行 `fitting -h` 可查看内嵌帮助。
