# 投稿前终核清单（每次更新稿件后过一遍）

## 引用与授权
- [ ] ref 38（Sun, Front Oncol 2021）：标题/卷期/DOI 以 PubMed PMID 34395236 为准终核
- [ ] 10x 各数据集授权文字：已在官网确认 CC BY 4.0（2026-08-14 核实，多页面原文 "This dataset is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license"）；Fig 4 图注署名已加 ✅
- [ ] spatialLIBD / HumanPilot 数据引用格式终核（Maynard 2021 已引 ✅）
- [ ] EMBL spatialdata 注释（Rep1）与 10x xlsx 注释（Rep2）的引用/致谢方式

## 数字与文件真实性（纪律：声称必须可复核）
- [ ] scripts/audit_results.py 全绿（最近一次 15/18 + 3 项文档级说明）
- [ ] 所有"已生成"文件以**无沙箱环境 ls 复核**（沙箱 overlay 幽灵文件教训，2026-08-14）
- [ ] 图中硬编码数字与最新实验 stdout/npz 逐项一致（Fig 3d-f、Fig 4d-i、Fig 5 Rep1 行）
- [ ] manuscript 正文数字与 audit/多宿主种子结果一致（CXCL12 −1.07/−38.9、PTN 三队列、12 切片 marker 命中数 PCP4 12/12 + KRT17 11/12）

## 投稿工程
- [ ] GitHub 公开仓库（用户账号）→ manuscript/cover letter 三处 [TBD] 回填
- [ ] PyPI 发布（用户 token）
- [ ] Zenodo DOI 归档
- [ ] 作者列表/单位/ORCID/Author contributions
- [ ] 推荐审稿人 3–5 名（cover letter 占位）
- [ ] Reporting Summary 誊入官方 Word 模板（draft 在 paper/reporting_summary_draft.md）
- [ ] PDF 合成 ≤30MB 预检；图注与最终图版逐字对齐
- [ ] Extended Data 补充图（消融/敏感性，约 14 张）
