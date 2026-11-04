# KIMI 小工具

Kimi 模型训练工程工具台。一个仓库、一个入口，按工作面拆成独立模块。

在线入口：[kimi显存分析](https://phillss.github.io/kimi-tools/)
源码：[kimi显存源码](https://github.com/Phillss/kimi-tools)

## 模块

| 编号 | 工作面 | 目录 | 状态 |
|---|---|---|---|
| 01 | 显存分析 | [`memory/`](memory/) | 已上线：FSDPTurbo 训练 / 推理 |
| 02 | 数据 | `data/` | 空 |
| 03 | 性能 | `perf/` | 空 |
| 04 | 技术报告 | `reports/` | 空 |

## 仓库结构

```text
kimi-tools/
├── kimi_memory.py      # Kimi-K3 激活值逐算子清单（Python 公式参考）
├── index.html          # 总入口
├── README.md           # 本文件：模块地图
└── memory/             # 01 显存分析
    ├── README.md
    ├── fsdp.html       # FSDPTurbo 训练显存建模
    └── infer.html      # 推理（iframe 嵌入 Eco-Sphere）
```

## Kimi-K3 模型概览

| 项 | 值 |
|---|---|
| 架构 | MoE（Stable LatentMoE）+ KDA + Gated MLA |
| 总参数 | 2.8T |
| 激活参数 | 104B |
| 层数 | 93（69 KDA + 24 Gated MLA + 1 Dense FFN） |
| hidden | 7168 |
| experts | 896（topk=16，shared=2） |
| 量化 | MXFP4-W / MXFP8-A（QAT） |
| 上下文 | 1M token |
| 视觉 | MoonViT-V2（401M） |
| 训练框架 | FSDPTurbo（FSDP2 + TP + EP + Ulysses CP） |

详见 [Kimi-K3 README](https://github.com/moonshotai/Kimi-K3)。

## 本地预览

```bash
cd kimi-tools
python3 -m http.server 8876 --bind 127.0.0.1
```

打开 `http://127.0.0.1:8876/`。

## 后续迭代

- Kimi-K2 / K2.5 显存分析支持
- safetensors header 逐张量求和，验证参数估算
