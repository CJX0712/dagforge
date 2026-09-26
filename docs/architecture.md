# DagForge 架构设计文档

> 作者：晨星 · 版本 v0.1.0 · 单向无环调用 `cli → pipeline → {data, discovery, eval} → core`

## 1. 设计原则

1. **确定性优先**：任何两次同 seed 运行必须产出逐位一致的核心指标。
2. **离线兜底**：SOTA 第三方后端（causal-learn / gcastle）缺失时，纯 numpy/scipy 仍能完整运行。
3. **单向无环依赖**：上层只依赖下层，禁止反向或横切引用，保证可测试、可替换。
4. **配置即代码**：所有超参集中在 `core/config.py`，支持 `DAGFORGE_*` 环境变量覆盖。

## 2. 分层结构

```
┌──────────────────────────────────────────────────────────┐
│  cli.py  ·  examples/run_demo.py   (入口 / 演示)            │
└───────────────────────────┬──────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────┐
│  pipeline/discovery_pipeline.py  (编排：数据→发现→评测→汇总) │
└─────────┬───────────────┬───────────────┬────────────────┘
          ▼               ▼               ▼
   ┌────────────┐  ┌──────────────┐  ┌────────────┐
   │ data/      │  │ discovery/   │  │ eval/      │
   │ synthetic  │  │ notears(pc)  │  │ metrics    │
   │ loader     │  │ corr, sota   │  │            │
   └─────┬──────┘  └──────┬───────┘  └─────┬──────┘
         └────────────────┴────────────────┘
                          ▼
                 ┌────────────────┐
                 │ core/          │
                 │ seed config    │
                 │ types errors   │
                 │ interfaces     │
                 └────────────────┘
```

## 3. 关键模块说明

### core/ — 地基
- `seed.py`：`set_all(seed)` 设齐随机源；`rng()` 派生不污染全局的 `Generator`。
- `config.py`：`DagForgeConfig` 数据类 + `from_env()`（`DAGFORGE_*` 覆盖）+ `validate()`。
- `types.py`：`Dataset` / `DAGResult` / `Metrics` 命名元组，明确数据契约。
- `errors.py`：分层异常 `DagForgeError(E000)` + 各域子类（E100~E500）。
- `interfaces.py`：`StructureLearner` 协议 + `BackendProbe`（可用性探测）。

### data/ — 数据与加载
- `synthetic.py`：
  - `generate_dag`：Erdős–Rényi 随机边，约束 `i<j` 保证生成即无环（DAG）；
  - `generate_dataset`：在 DAG 上跑**线性高斯 SEM** 采样，`X = W·X + noise`，标注真值邻接；
  - `generate_benchmark_suite`：批量生成可复现基准数据集。
- `loader.py`：`load_csv(path, has_header, true_adj_path)` → `Dataset`，支持从真实 CSV 学习。

### discovery/ — 结构学习方法
- `notears.py`（**旗舰**）：
  - `_h(W) = tr(e^{W∘W}) − d` 无环约束；
  - `_grad_h(W) = expm(W∘W)ᵀ · (2W)`；
  - `_ensure_acyclic`：DFS 找环，移去环上最小 `|W|` 边直到 DAG；
  - `notears_linear`：增广拉格朗日外层 + 内层 **L-BFGS-B**（平滑 L1 + 变量界 `±5`）。
- `pc.py`：约束式 PC——Fisher-Z 偏相关条件独立性检验 + v-结构 + Meek R1 定向，纯 numpy/scipy。
- `corr.py`：`corr_threshold`（强基线，朴素相关阈值，无向）、`empty_graph`（下界基线）。
- `sota.py`：`available_causallearn()` / `available_gcastle()` 实测探针 + 对应封装；不可用时返回 `None`，由 pipeline 剔除。

### eval/ — 评测
- `metrics.py`：
  - `structure_metrics`：骨架 SHD / precision / recall / F1 / FDR；
  - `shd_directed`：含方向的结构汉明距离；
  - `aggregate`：多数据集指标聚合（均值/标准差）。

### pipeline/ — 编排
- `discovery_pipeline.py`：`DagForgePipeline.run()`：
  1. `set_all(seed)` → 生成基准数据集；
  2. `_method_candidates()` 列出候选方法（含 SOTA 后端）；
  3. **可用性探测**：首数据集试跑，返回 `None` 的后端直接剔除；
  4. 逐数据集评测，聚合；
  5. `_headline()` 计算 SHD 下降% 与 F1 增益，`_cross_validation()` 独立交叉验证；
  6. 产出 `benchmark.json`。

## 4. 确定性实现要点

| 风险来源 | 对策 |
|----------|------|
| 多线程 BLAS 浮点舍入不确定 | 入口/测试/conftest 强制 `OMP_NUM_THREADS=1` 等 5 个变量单线程 |
| 随机种子分散 | 统一 `set_all(seed)`；方法实现本身无随机性 |
| 墙钟耗时不可比 | 校验时剔除 `elapsed*` 字段，仅比算法指标 |

## 5. 离线兜底流程

```
method_candidates()
   ├─ 纯 numpy/scipy：notears_lin / pc / corr_threshold / empty   （永远可用）
   └─ SOTA 后端（use_sota=True 时）：
        available_causallearn()? → cl_pc / cl_notears
        available_gcastle()?     → gcastle_notears
        探测失败（如缺 torch）→ 返回 None → pipeline 剔除，不影响其他结果
```

## 6. 可扩展性

新增一个结构学习方法只需：
1. 在 `discovery/` 下实现 `fn(dataset) -> DAGResult`；
2. 在 `pipeline._method_candidates()` 注册名称与调用；
3. 如是对标 SOTA，在 `sota.py` 加 `available_*` 探针。

无需改动评测、数据、核心层。
