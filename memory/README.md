# 显存分析

Kimi-K3 单卡显存估算：FSDPTurbo 训练，以及 MoE 推理。

## 页面

- [FSDPTurbo 显存建模](fsdp.html) —— FSDP2 + TP + EP + Ulysses CP 全量微调
- [推理显存建模](infer.html) —— MoE 推理（嵌入 Eco-Sphere）

源码目录即本文件夹。GitHub：[kimi-tools](https://github.com/zyang6/qwen-tools)（参考 qwen-tools 架构）。

## 选哪一页

| 页面 | 场景 | 静态怎么切 | 高峰 |
|---|---|---|---|
| [`fsdp.html`](fsdp.html) | FSDPTurbo 全量微调 | `N/P × 12`（fp32 权重 + Adam；不含梯度）。Adam 在 CPU 则优化器为 0 | 梯度 `N/P×4` + 动态 + 激活 |
| [`infer.html`](infer.html) | MoE 推理部署 | 权重按 EP/TP 切 | KV / 激活 / HCCL / ACLGraph / CANN / OS |

## Kimi-K3 模型参数

来自 [Kimi-K3 README](https://github.com/moonshotai/Kimi-K3) + `configuration_kimi_k3.py` + `kimi_memory.py`：

| 参数 | 值 | 说明 |
|---|---|---|
| 总参数 | 2.8T | Total Parameters |
| 激活参数 | 104B | Activated Parameters |
| 层数 | 93 | 69 KDA + 24 Gated MLA |
| Dense 层数 | 1 | 前 1 层 dense FFN |
| hidden_size | 7168 | D |
| vocab_size | 163840 | ~160K |
| attention heads | 96 | nH |
| KDA head_dim | 128 | kdaHeadDim |
| KDA chunk_size | 64 | kdaChunkSize |
| q_lora_rank | 1536 | MLA |
| kv_lora_rank | 512 | MLA |
| qk_nope_head_dim | 128 | MLA |
| qk_rope_head_dim | 64 | MLA |
| v_head_dim | 128 | MLA |
| experts | 896 | 路由专家数 |
| topk | 16 | 每 token 选专家数 |
| shared experts | 2 | 共享专家数 |
| moe_intermediate | 3072 | D_MOE，专家中间维度 |
| latent_moe_dim | 3584 | D_PART，Stable LatentMoE 路由维度 |
| dense_intermediate | 33792 | D_DENSE，第 1 层 dense FFN |
| activation | SiTU-GLU | 激活函数 |
| quantization | MXFP4-W / MXFP8-A | QAT 量化感知训练 |
| context length | 1048576 | 1M token |
| vision encoder | MoonViT-V2 | 401M 参数 |

## FSDPTurbo 口径

```text
静态 = N/P × 12          # fp32 参数(4) + Adam(8)；梯度不算进静态
高峰再叠：梯度 N/P×4 + 动态 + 激活 + HCCL/碎片/PTA/OS

N = Text + ViT
N_expert = L_MOE × (E × 3 × D_PART × D_MOE)   # 路由专家
N_dense  = N − N_expert

动态内存（wrap=experts）：
  U = (E × 3 × D_PART × D_MOE) × 2 / EP      # bf16 all-gather 单位
  动态 = all_gather(U) + copy_out(U) + prefetch×U + foreach_reduce(2U) + GMM(U)

激活值（kimi_memory.py 逐算子清单）：
  recompute/offload 开启 → 只留 1 层（取 max(M_KDA, M_MLA, M_MLP, M_MOE)）+ Embed + LMHead
  关闭 → 全层累加：
    M_list = L_KDA×M_KDA + L_MLA×M_MLA + L_DENSE×M_MLP + L_MOE×M_MOE
    M_total = M_list + M_Embed + M_LMHead

  M_KDA = T × (4D + 22×nH×kdaHD + 4×nH×chunk + 4×nH)
  M_MLA = T × (4D + 4×qLora + 4×kvLora + 2×qkRope + 8×nH×(qkNope+qkRope) + 6×nH×vHead + 8×nH×faStats)
  M_MLP = T × (4D + 6×denseInter)
  M_MOE = T × (8D + (8×topk+4)×D_PART + 6×(topk+shared)×D_MOE)

  T = mbs × (seq / CP)    # CP 切分后每 rank token 数

chunk_loss 开启时 lm_head 物化从 [seq, vocab] 降至 [chunk, vocab]
mm_merge 开启时图文合并按块搬运，否则全量拷贝 ~seq×H×2

HCCL = n_comm × BUFFSIZE × 2 / 1024
n_comm = 1 + [EP>1] + [CP>1] + [TP>1]

整卡 = 静态 + 梯度 + 动态 + 激活 + HCCL + 碎片1 + PTA1 + Device OS 3.5
```

## 并行度约束（FSDPTurbo）

```text
world_size % (fully_shard_parallel_size × tensor_parallel_size) == 0
world_size % (expert_fully_shard_parallel_size × expert_parallel_size) == 0
world_size % (ulysses_parallel_size × tensor_parallel_size) == 0
```

FSDP、EP、CP 使用独立 device mesh，三组并行度不需彼此相乘。

默认配置（参考 `examples/kimi_k3/config.yaml`）：

```yaml
distributed:
  fully_shard_parallel_size: 16   # P
  tensor_parallel_size: 1         # TP
  ulysses_parallel_size: 2        # CP
  expert_parallel_size: 16        # EP
  expert_fully_shard_parallel_size: 1  # EFSDP
```

## 推理口径

嵌入 Eco-Sphere 工具，计算在对方页面完成：

```text
单卡 = 权重 + KV/Index Cache + 激活 + HCCL
     + ACLGraph + CANN/PTA/算子 + Device OS
```

Kimi-K3 推理引擎：[vLLM](https://github.com/vllm-project/vllm) / [SGLang](https://github.com/sgl-project/sglang) / [TokenSpeed](https://lightseek.org/tokenspeed)。

## 本目录文件

```text
memory/
├── README.md        # 本文件
├── fsdp.html        # FSDPTurbo 训练显存建模
└── infer.html       # 推理（iframe 嵌入 Eco-Sphere）
```

## 后续迭代

- Kimi-K2 / K2.5 支持：复用框架，调整 `countParams` 与激活清单（去 KDA / 去 AttnRes / 标准 MoE）。
- safetensors header 逐张量求和：验证 N 估算精度。

## 反馈

发现公式、切分或模型解析有问题，请开 [Issue](https://github.com/zyang6/qwen-tools/issues/new)，附：

- 页面（FSDP / 推理）
- 模型与切分
- 工具结果 / 预期 / 日志
