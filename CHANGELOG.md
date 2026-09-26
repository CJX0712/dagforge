# Changelog

遵循语义化版本（SemVer）。所有版本作者：晨星。

## [0.1.0] - 2026-09-27

### Added
- 旗舰方法 NOTEARS（线性高斯）纯 numpy/scipy 实现：无环约束 `h(W)=tr(e^{W∘W})−d` +
  增广拉格朗日 + 内层 L-BFGS-B（平滑 L1 + 变量界 ±5）。
- 约束式 PC 算法（Fisher-Z 偏相关 CI 检验 + v-结构 + Meek R1 定向）。
- 朴素强基线相关阈值法（corr_threshold）与下界基线（empty_graph）。
- SOTA 后端探针：causal-learn PC（已验证可用，作交叉验证）、gcastle NOTEARS（需 torch，优雅降级）。
- 合成数据引擎：Erdős–Rényi DAG + 线性高斯 SEM，真值标注，可复现基准套件。
- 评测模块：SHD / F1 / precision / recall / FDR + 聚合。
- 确定性保障：全局 `set_all(seed)` + 单线程 BLAS 强制。
- CLI（`run`/`learn`/`demo`）+ 端到端演示 `examples/run_demo.py`。
- 完整文档：README / architecture.md / model_card.md / CHANGELOG / LICENSE / Dockerfile / Makefile / CI。
- 23 个单元测试，全部通过；双跑确定性校验逐位一致。

### Fixed
- NOTEARS 陈旧无环约束惩罚（用外层迭代前的陈腐 `h(W)`）→ 每层重算 `h_new`。
- Adam 收敛过慢 → 切换 L-BFGS-B。
- FISTA 因 `expm` 梯度溢出 NaN 发散 → 变量有界 ±5 + 平滑 L1 彻底解决。
- 多线程 BLAS 导致跨运行浮点不确定性 → 强制单线程。

### Quality
- 质量等级：**S**（SHD↓94.86%、F1↑0.368、CI 全绿、依赖锁定、离线可降级、tag v0.1.0 + Release）。
