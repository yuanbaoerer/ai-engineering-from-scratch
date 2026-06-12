"""托管 LLM 平台比较器 — 纯 Python 标准库实现。

模拟三个云 LLM 平台（Bedrock 按需、Azure PTU、Vertex 按需）在相同工作负载下的表现。
报告每日成本、TTFT 中位数/P99 和成本归属能力，帮助你选择合适的平台。

教育目的：价格和延迟数据是 2026 年的近似值。

演示的核心概念：
1. 成本对比：按需 vs 预留容量（PTU）
2. 延迟建模：不同容量模型下的中位数 TTFT vs P99
3. SLA 验证：检查平台是否满足 P99 延迟要求
4. 盈亏平衡分析：PTU 何时比按需更便宜
5. 多提供商冗余：避免厂商锁定的成本
"""

from __future__ import annotations

from dataclasses import dataclass
import random
import statistics


@dataclass
class Platform:
    """
    云 LLM 平台的定价和性能特征数据类。
    
    每个平台有不同的定价模型、延迟特性和成本归属机制。
    这个类封装了比较所需的所有变量。
    
    属性说明：
        name: 平台名称（如 "Bedrock 按需"）
        per_mtok_input: 按需输入 token 成本（$/百万 token）
        per_mtok_output: 按需输出 token 成本（$/百万 token）
        ptu_hourly: 一个 PTU 预留单位的每小时成本（None 表示不提供 PTU）
        ptu_tokens_per_hour: 一个 PTU 每小时能处理的 token 数
        ttft_median_ms: 共享/按需容量的中位数首 token 延迟（毫秒）
        ttft_p99_ms: 共享容量的 P99 延迟 - 最坏情况延迟
        ttft_median_ptu_ms: 使用专用 PTU 容量时的中位数 TTFT（更低更稳定）
        attribution: 成本归属能力等级（FinOps 评估）
    """
    name: str
    per_mtok_input: float        # $/百万输入 token（按需）
    per_mtok_output: float       # $/百万输出 token（按需）
    ptu_hourly: float | None     # $/小时（一个预留单位，None = 不提供）
    ptu_tokens_per_hour: int     # 一个 PTU 每小时处理的 token 数
    ttft_median_ms: float        # 共享容量中位数 TTFT
    ttft_p99_ms: float           # 共享容量 P99 TTFT
    ttft_median_ptu_ms: float    # 专用 PTU 中位数 TTFT
    attribution: str             # FinOps 归属能力定性评级


# 平台定义，包含 2026 年的近似定价和性能数据
# 数据来源：AWS/Azure/GCP 定价页面，Artificial Analysis 基准测试
#
# Bedrock："市场模式" - 通过一个 API 访问多种模型（Claude、Llama、Titan）
#   - 每 token 成本较高但模型多样性强
#   - Application Inference Profiles 提供清晰的成本归属
#   - Provisioned Throughput 可用但起价 $21/小时
#
# Azure OpenAI："独家合作" - 只有 OpenAI 模型（GPT-4/5）
#   - 每 token 成本较低，PTU 提供专用容量
#   - 基于作用域的归属（订阅/资源组）
#   - PTU 延迟最低（约 50ms 中位数）
#
# Vertex AI："Gemini 优先" - Google 模型 + Model Garden
#   - 每 token 成本最低（Gemini 效率高）
#   - 无 PTU 等效产品（预留容量按 SKU 销售）
#   - 基于 BigQuery 的归属（灵活但需手动配置）
#
PLATFORMS = [
    Platform("Bedrock 按需",    3.00, 15.00, 21.0, 1_200_000, 75, 180, 55, "A（Application Inference Profiles）"),
    Platform("Azure OpenAI (PTU)",    2.50, 10.00, 10.0, 2_000_000, 50, 140, 38, "B（作用域 + 标签 + PTU 对象）"),
    Platform("Vertex AI Gemini",     1.25,  5.00, None,          0, 60, 160,  0, "B+（BigQuery 账单导出）"),
]


def simulate(tokens_in_per_day: int, tokens_out_per_day: int, sla_ttft_ms: float, use_ptu: bool) -> None:
    """
    模拟给定工作负载配置下各平台的成本和延迟。
    
    在相同工作负载条件下比较三个平台，显示哪个最便宜以及是否满足延迟 SLA。
    
    成本计算逻辑：
    1. 按需成本：(tokens_in * input_price + tokens_out * output_price) / 1M
    2. PTU 成本：计算所需 PTU 数量，然后 hourly_rate * 24 小时
    3. 实际成本：按需和 PTU 的最小值（取更便宜的）
    
    延迟逻辑：
    - 按需：使用共享容量的中位数/P99（方差较大）
    - PTU：使用专用容量延迟（更低更稳定）
    - PTU 的 P99 估算为中位数的 1.5 倍（分布更集中）
    
    参数：
        tokens_in_per_day: 每日处理的输入 token 总数
        tokens_out_per_day: 每日生成的输出 token 总数
        sla_ttft_ms: 可接受的最大 P99 首 token 延迟（你的 SLA）
        use_ptu: 是否考虑 PTU 定价（True）还是仅按需（False）
    """
    print(f"\nWorkload: {tokens_in_per_day/1e6:.1f}M input, {tokens_out_per_day/1e6:.1f}M output per day")
    print(f"SLA: TTFT P99 < {sla_ttft_ms:.0f} ms   |   PTU path: {'enabled' if use_ptu else 'off'}\n")
    
    # 表格标题
    header = f"{'Platform':25}  {'$/day':>9}  {'TTFT P50':>10}  {'TTFT P99':>10}  {'SLA':>6}  Attribution"
    print(header)
    print("-" * len(header))

    for p in PLATFORMS:
        # 计算按需成本（无预留容量）
        # 公式：(input_tokens / 1M) * input_price + (output_tokens / 1M) * output_price
        cost_ondemand = (tokens_in_per_day / 1e6) * p.per_mtok_input + \
                        (tokens_out_per_day / 1e6) * p.per_mtok_output

        if use_ptu and p.ptu_hourly is not None:
            # PTU 路径：计算预留容量成本
            # 总 token = input + output（PTU 按总吞吐量计费）
            total_tokens = tokens_in_per_day + tokens_out_per_day
            
            # 一个 PTU 每天能处理多少 token？
            daily_capacity_per_ptu = p.ptu_tokens_per_hour * 24
            
            # 计算需要多少 PTU（向上取整）
            # 示例：50M token/天，容量 28.8M，需要 2 个 PTU
            ptu_count = max(1, (total_tokens + daily_capacity_per_ptu - 1) // daily_capacity_per_ptu)
            
            # PTU 成本 = 单位数 * 小时费率 * 24 小时（全天候预留，即使空闲）
            cost_ptu = ptu_count * p.ptu_hourly * 24
            
            # 选择更便宜的选项（按需或 PTU）
            cost = min(cost_ondemand, cost_ptu)
            
            # 延迟取决于使用的路径
            # PTU：使用专用容量延迟（更低更稳定）
            # 按需：使用共享容量延迟（更高更波动）
            ttft_p50 = p.ttft_median_ptu_ms if cost == cost_ptu else p.ttft_median_ms
            ttft_p99 = ttft_p50 * 1.5 if cost == cost_ptu else p.ttft_p99_ms
            path = "PTU" if cost == cost_ptu else "on-demand"
        else:
            # 仅按需路径（无 PTU 或 PTU 不可用）
            cost = cost_ondemand
            ttft_p50 = p.ttft_median_ms
            ttft_p99 = p.ttft_p99_ms
            path = "on-demand"

        # 检查平台是否满足延迟 SLA（P99 必须低于阈值）
        sla_ok = "PASS" if ttft_p99 < sla_ttft_ms else "FAIL"
        print(f"{p.name:25}  ${cost:8.2f}  {ttft_p50:7.0f} ms  {ttft_p99:7.0f} ms  {sla_ok:>6}  {p.attribution}  [{path}]")


def break_even_demo() -> None:
    """
    演示 Azure OpenAI PTU vs 按需定价的盈亏平衡分析。
    
    展示在什么利用率下 PTU 比按需更便宜。
    
    核心洞察：PTU 是固定成本（$10/小时 = $240/天），而按需是可变成本。
    低利用率时，你在为空闲容量付费；高利用率时，PTU 胜出。
    
    盈亏平衡点通常在 40-60% 利用率。
    
    业务含义：
    - 可预测的高流量工作负载：使用 PTU（节省高达 70%）
    - 突发或低流量工作负载：使用按需（避免为空闲容量付费）
    """
    print("\n" + "=" * 80)
    print("PTU 盈亏平衡分析 — Azure OpenAI, GPT-4o 级别")
    print("=" * 80)
    p = PLATFORMS[1]  # Azure（PLATFORMS 列表索引 1）
    print(f"按需费率：${p.per_mtok_output:.2f}/百万输出 token  |  PTU：${p.ptu_hourly:.0f}/小时, {p.ptu_tokens_per_hour/1e6:.1f}M token/小时\n")
    print(f"{'利用率':>8}  {'按需 $/天':>18}  {'PTU $/天':>12}  胜出者")
    
    # 遍历利用率从 10% 到 100%
    # 在每个点计算哪个选项更便宜
    for util_pct in (10, 20, 30, 40, 50, 60, 70, 80, 90, 100):
        # 计算在这个利用率水平下处理多少 token
        tokens_per_day = int(p.ptu_tokens_per_hour * 24 * (util_pct / 100.0))
        
        # 按需成本：可变的，随使用量增长
        ondemand = (tokens_per_day / 1e6) * p.per_mtok_output
        
        # PTU 成本：固定的，无论使用多少都要付全额
        ptu = 24 * p.ptu_hourly
        
        # 判断哪个更便宜
        winner = "PTU" if ptu < ondemand else "on-demand"
        print(f"{util_pct:>7}%  ${ondemand:>16.2f}  ${ptu:>10.2f}  {winner}")


def lock_in_cost() -> None:
    """
    计算实施双提供商最低要求策略的冗余成本。
    
    为什么要用两个提供商？
    - 单一提供商 = 厂商锁定 = 被排除在 2/3 的前沿模型之外
    - 云服务故障时有发生（如 2025 年 1 月 Azure OpenAI 事件、AWS us-east-1 故障）
    - 模型季度轮换：Claude 3.7 → Gemini 2.5 → GPT-5
    
    "双提供商最低要求"策略意味着：
    1. 每个关键 LLM 路径在 ≥2 个提供商上运行
    2. 网关在它们之间路由（如 LiteLLM、自定义代理）
    3. 一个提供商宕机时自动故障转移
    
    成本构成：
    - 网关开销：约 3%（路由逻辑、健康检查、重试逻辑）
    - 空闲备用余量：约 10%（保持备用容量随时可用）
    - 总增加：比单一提供商高约 13%
    
    业务案例：13% 的成本增加是防止以下情况的保险：
    - 客户因故障流失
    - SLA 赔偿支付
    - 故障期间的应急响应时间
    """
    print("\n" + "=" * 80)
    print("双提供商最低要求 — 冗余成本增加")
    print("=" * 80)
    
    # 示例工作负载：单提供商 5M token/天
    tokens_per_day = 5_000_000
    primary_cost = (tokens_per_day / 1e6) * 10.00  # 假设 $10/百万输出 token
    
    # 网关开销：路由、健康检查、负载均衡
    # 这是运行代理层的操作成本
    gateway_overhead_pct = 3.0
    
    # 空闲备用余量：保持备用提供商随时可用
    # 即使不服务流量，也要为故障转移做好准备
    failover_headroom_pct = 10.0
    
    # 总增加 = 网关 + 余量
    uplift = primary_cost * (gateway_overhead_pct + failover_headroom_pct) / 100
    
    print(f"主提供商每日支出：${primary_cost:.2f}")
    print(f"网关开销（{gateway_overhead_pct:.0f}%）：${primary_cost * gateway_overhead_pct / 100:.2f}/天")
    print(f"空闲备用余量（{failover_headroom_pct:.0f}%）：${primary_cost * failover_headroom_pct / 100:.2f}/天")
    print(f"总增加：${uplift:.2f}/天")
    print(f"月增加：${uplift * 30:.2f}")
    print("无冗余时一次多小时区域故障的成本：客户流失、SLA 赔偿、应急响应时间")


def main() -> None:
    """
    主入口：运行三个模拟场景演示平台权衡。
    
    场景 1：低流量工作负载（3M 输入，1M 输出 token/天）
    - 宽松 SLA：P99 TTFT < 200ms（容易满足）
    - 无需 PTU（成本不划算）
    - 结果：Vertex 最便宜，Bedrock 最贵，全部通过 SLA
    
    场景 2：高流量工作负载（30M 输入，15M 输出 token/天）
    - 严格 SLA：P99 TTFT < 100ms（需要专用容量）
    - 启用 PTU：展示何时预留容量更划算
    - 结果：Azure/Bedrock 的 PTU 可能更便宜，Vertex 仍然最便宜
    
    场景 3：盈亏平衡分析
    - 展示什么利用率下 PTU 比按需更便宜
    - 核心洞察：盈亏平衡点在 40-60% 利用率
    
    场景 4：多提供商成本
    - 计算 13% 的冗余成本增加
    - 理由：防止故障和厂商锁定的保险
    """
    print("=" * 80)
    print("托管 LLM 平台比较器 — 2026 年近似值")
    print("=" * 80)

    # 场景 1：低流量，宽松 SLA - 无需 PTU
    simulate(tokens_in_per_day=3_000_000, tokens_out_per_day=1_000_000, sla_ttft_ms=200, use_ptu=False)
    
    # 场景 2：高流量，严格 SLA - PTU 可能划算
    simulate(tokens_in_per_day=30_000_000, tokens_out_per_day=15_000_000, sla_ttft_ms=100, use_ptu=True)

    # 场景 3：PTU 盈亏平衡分析（Azure OpenAI）
    break_even_demo()
    
    # 场景 4：多提供商冗余成本
    lock_in_cost()


if __name__ == "__main__":
    main()
