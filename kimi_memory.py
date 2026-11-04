"""按 Kimi 显存分析文档计算清单累计量；直接修改顶部全局变量即可。"""
import unicodedata
import textwrap
# 输入与模型参数。B/S 是分析设置，其余维度来自 Kimi-K3 text_config。
B = 1
S = 4096
D = 7168
VOCAB_SIZE = 163840
N_HEADS = 96
KDA_HEAD_DIM = 128
KDA_CHUNK_SIZE = 64
Q_LORA_RANK = 1536
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_NOPE_HEAD_DIM = 128
V_HEAD_DIM = 128
FA_STATS_WIDTH = 8
D_DENSE = 33792
D_PART = 3584
D_MOE = 3072
N_ROUTED_EXPERTS = 896
N_SHARED_EXPERTS = 2
TOP_K = 16
L_TOTAL = 93
L_KDA = 69
L_MLA = 24
L_DENSE = 1
L_MOE = 92
BF16_BYTES = 2
FP32_BYTES = 4
MIB = 2**20
GIB = 2**30
def build_modules():
    """每行对应文档一行，dtype 字节数位于表达式末尾。"""
    t = B * S
    def bf(op, module, expression, elements):
        return (op, module, "bf16", expression + " * 2", elements * BF16_BYTES)
    def fp(op, module, expression, elements):
        return (op, module, "fp32", expression + " * 4", elements * FP32_BYTES)
    td = t * D
    hd = t * N_HEADS * KDA_HEAD_DIM
    chunk = t * N_HEADS * KDA_CHUNK_SIZE
    kda = [
        bf("RmsNorm", "Decoder.input_layernorm", "T * D", td),
        bf("matmul", "KDA.q_proj", "T * N_HEADS * KDA_HEAD_DIM", hd),
        bf("matmul", "KDA.k_proj", "T * N_HEADS * KDA_HEAD_DIM", hd),
        bf("matmul", "KDA.v_proj", "T * N_HEADS * KDA_HEAD_DIM", hd),
        bf("conv1d", "KDA.q/k/v_conv1d", "T * N_HEADS * KDA_HEAD_DIM", hd),
        bf("matmul", "KDA.f_b_proj", "T * N_HEADS * KDA_HEAD_DIM", hd),
        fp("sigmoid", "KDA.b_proj/chunk_kda", "T * N_HEADS", t * N_HEADS),
        bf("l2norm", "KDA.chunk_kda (q)", "T * N_HEADS * KDA_HEAD_DIM", hd),
        bf("l2norm", "KDA.chunk_kda (k)", "T * N_HEADS * KDA_HEAD_DIM", hd),
        bf("chunk_kda", "KDA.chunk_kda", "T * N_HEADS * KDA_HEAD_DIM", hd),
        bf("transpose/contiguous", "KDA.Aqk (候选)", "T * N_HEADS * KDA_CHUNK_SIZE", chunk),
        bf("transpose/contiguous", "KDA.Akk (候选)", "T * N_HEADS * KDA_CHUNK_SIZE", chunk),
        bf("RmsNorm", "KDA.o_norm", "T * N_HEADS * KDA_HEAD_DIM", hd),
        bf("sigmoid", "KDA.o_norm", "T * N_HEADS * KDA_HEAD_DIM", hd),
        bf("mul", "KDA.o_norm", "T * N_HEADS * KDA_HEAD_DIM", hd),
        bf("add", "Decoder.attention_residual", "T * D", td),
    ]
    qk = t * N_HEADS * (QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM)
    v = t * N_HEADS * V_HEAD_DIM
    stats = t * N_HEADS * FA_STATS_WIDTH
    qk_expr = "T * N_HEADS * (QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM)"
    v_expr = "T * N_HEADS * V_HEAD_DIM"
    mla = [
        bf("RmsNorm", "Decoder.input_layernorm", "T * D", td),
        bf("matmul", "MLA.q_a_proj", "T * Q_LORA_RANK", t * Q_LORA_RANK),
        bf("rmsnorm", "MLA.q_a_layernorm", "T * Q_LORA_RANK", t * Q_LORA_RANK),
        bf("matmul", "MLA.kv_a_proj_with_mqa", "T * (KV_LORA_RANK + QK_ROPE_HEAD_DIM)", t * (KV_LORA_RANK + QK_ROPE_HEAD_DIM)),
        bf("rmsnorm", "MLA.kv_a_layernorm", "T * KV_LORA_RANK", t * KV_LORA_RANK),
        bf("cat", "MLA.cat (q)", qk_expr, qk),
        bf("cat", "MLA.cat (k)", qk_expr, qk),
        bf("pad", "MLA.F.pad", qk_expr, qk),
        ("flash_attn", "MLA.attention_interface", "bf16 / fp32 / fp32",
         qk_expr + " * 2 + T * N_HEADS * FA_STATS_WIDTH * 4 + T * N_HEADS * FA_STATS_WIDTH * 4",
         qk * BF16_BYTES + stats * FP32_BYTES + stats * FP32_BYTES),
        bf("copy_slice", "MLA.slice/contiguous", v_expr, v),
        bf("sigmoid", "MLA.g_proj", v_expr, v),
        bf("mul", "MLA.output_gate", v_expr, v),
        bf("add", "Decoder.attention_residual", "T * D", td),
    ]
    mlp = [
        bf("rmsnorm", "Decoder.post_attention_layernorm", "T * D", td),
        bf("cat", "MLP.cat", "2 * T * D_DENSE", 2 * t * D_DENSE),
        bf("situglu", "MLP.act_fn/SituAndMul", "T * D_DENSE", t * D_DENSE),
        bf("add", "Decoder.ffn_residual", "T * D", td),
    ]
    routed = t * TOP_K
    moe = [
        bf("rmsnorm", "Decoder.post_attention_layernorm", "T * D", td),
        fp("copy_cast", "MoE.gate/KimiMoEGate", "T * D", td),
        bf("permute", "MoE.experts.permute", "T * TOP_K * D_PART", routed * D_PART),
        bf("gmm", "MoE.experts.gate_up_proj", "2 * T * TOP_K * D_MOE", 2 * routed * D_MOE),
        bf("situglu", "MoE.experts._apply_activation", "T * TOP_K * D_MOE", routed * D_MOE),
        bf("gmm", "MoE.experts.down_proj", "T * TOP_K * D_PART", routed * D_PART),
        fp("copy_cast", "MoE.unpermute", "T * TOP_K * D_PART", routed * D_PART),
        bf("copy_cast", "MoE.unpermute", "T * D_PART", t * D_PART),
        bf("rmsnorm", "MoE.routed_expert_norm", "T * D_PART", t * D_PART),
        bf("cat", "MoE.shared_experts.cat", "2 * T * N_SHARED_EXPERTS * D_MOE", 2 * t * N_SHARED_EXPERTS * D_MOE),
        bf("situglu", "MoE.shared_experts.act_fn", "T * N_SHARED_EXPERTS * D_MOE", t * N_SHARED_EXPERTS * D_MOE),
        bf("add", "Decoder.ffn_residual", "T * D", td),
    ]
    embedding = [bf("embedding", "Model.embed_tokens", "T * D", td)]
    lm_head = [
        bf("rmsnorm", "Model.norm", "T * D", td),
        bf("matmul", "LMHead.lm_head", "T * VOCAB_SIZE", t * VOCAB_SIZE),
    ]
    return {"KDA": kda, "MLA": mla, "MLP": mlp, "MoE": moe,
            "Embedding": embedding, "LMHead": lm_head}
def width(text):
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
def print_table(name, rows):
    print(f"\n{name}：单模块清单累计量")
    headers = ["编号", "算子", "模型对应", "dtype", "显存表达式 (bytes)", "MiB"]
    body = [[str(i), *row[:4], f"{row[4] / MIB:,.3f}"] for i, row in enumerate(rows, 1)]
    total = sum(row[4] for row in rows)
    body.append(["合计", "", "", "", "", f"{total / MIB:,.3f}"])
    physical_rows = []
    for row in body:
        formulas = textwrap.wrap(row[4], width=56, break_long_words=False) or [""]
        physical_rows.append([*row[:4], formulas[0], row[5]])
        physical_rows.extend(["", "", "", "", part, ""] for part in formulas[1:])
    sizes = [max(width(row[j]) for row in [headers, *physical_rows]) for j in range(len(headers))]
    def render(row):
        cells = []
        for j, cell in enumerate(row):
            padding = " " * (sizes[j] - width(cell))
            cells.append(padding + cell if j in (0, 5) else cell + padding)
        return "| " + " | ".join(cells) + " |"
    print(render(headers))
    print("| " + " | ".join("-" * n for n in sizes) + " |")
    for row in physical_rows:
        print(render(row))
    return total
def main():
    if L_KDA + L_MLA != L_TOTAL or L_DENSE + L_MOE != L_TOTAL:
        raise ValueError("注意力与前馈层数必须分别等于 L_TOTAL")
    if not 0 < TOP_K <= N_ROUTED_EXPERTS:
        raise ValueError("请检查 TOP_K")
    if (BF16_BYTES, FP32_BYTES) != (2, 4):
        raise ValueError("本脚本的公式展示约定 bf16=2 bytes、fp32=4 bytes")
    print("Kimi-K3 训练显存公式计算（不分配模型或设备张量）")
    print(f"B={B}, S={S}, T=B*S={B*S}, D={D}, N_HEADS={N_HEADS}")
    print(f"层数：KDA={L_KDA}, MLA={L_MLA}, MLP={L_DENSE}, MoE={L_MOE}")
    totals = {name: print_table(name, rows) for name, rows in build_modules().items()}
   # 独立合并公式与逐项表格交叉检查。
    t = B * S
    closed = {
        "KDA": t * (4*D + 22*N_HEADS*KDA_HEAD_DIM + 4*N_HEADS*KDA_CHUNK_SIZE + 4*N_HEADS),
        "MLA": t * (4*D + 4*Q_LORA_RANK + 4*KV_LORA_RANK + 2*QK_ROPE_HEAD_DIM
                    + 8*N_HEADS*(QK_NOPE_HEAD_DIM+QK_ROPE_HEAD_DIM) + 6*N_HEADS*V_HEAD_DIM + 8*N_HEADS*FA_STATS_WIDTH),
        "Embedding": t * D * BF16_BYTES,
        "LMHead": t * D * BF16_BYTES + t * VOCAB_SIZE * BF16_BYTES,
        "MLP": t * (4*D + 6*D_DENSE),
        "MoE": t * (8*D + (8*TOP_K+4)*D_PART + 6*(TOP_K+N_SHARED_EXPERTS)*D_MOE),
    }
    if totals != closed:
        raise RuntimeError("逐项表格与合并公式不一致")
    activation = (L_KDA*totals["KDA"] + L_MLA*totals["MLA"]
                  + L_DENSE*totals["MLP"] + L_MOE*totals["MoE"])
    print("\n整模型清单累计量公式（MiB）")
    print("M_list = L_KDA*M_KDA + L_MLA*M_MLA + L_DENSE*M_MLP + L_MOE*M_MOE")
    print(f"       = {L_KDA}*{totals['KDA']/MIB:g} + {L_MLA}*{totals['MLA']/MIB:g}"
          f" + {L_DENSE}*{totals['MLP']/MIB:g} + {L_MOE}*{totals['MoE']/MIB:g}")
    print(f"       = {activation/MIB:,.3f} MiB = {activation/GIB:,.3f} GiB")
    total = activation + totals["Embedding"] + totals["LMHead"]
    print("\n加入输入输出层的理论累计量（MiB）")
    print("M_total = M_list + M_Embedding + M_LMHead")
    print(f"        = {activation/MIB:.3f} + {totals['Embedding']/MIB:.3f} + {totals['LMHead']/MIB:.3f}")
    print(f"        = {total/MIB:,.3f} MiB = {total/GIB:,.3f} GiB")
    print("\n静态状态公式（bytes）")
    print("Adam: M_static = 16 * P / D_shard")
    print("Muon: M_static = 12 * P / D_shard")

if __name__ == "__main__":
    main()