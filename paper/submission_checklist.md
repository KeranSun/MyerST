# Nature Communications 投稿系统操作清单（照着点）

入口：https://mts-ncomms.nature.com/ （Editorial Manager 系统，先注册账号，建议用通讯邮箱 sunkeran@stu.hebmu.edu.cn 注册并绑定 ORCID 0000-0001-6065-8945）

## Step 1 — Article type
- 选 **Article**
- 如有 collection 选项，勾选对应的 "AI in spatial omics" 专题（没有就跳过，cover letter 里已说明）

## Step 2 — Title & Abstract
- Title 粘贴：`Topology-constrained attribution reveals verifiable explanations for spatial omics graph neural networks`
- Abstract 粘贴 manuscript_v1.md 的 Abstract 段（147 词）

## Step 3 — Authors
- 按 manuscript 顺序逐一添加 4 位作者（姓名/单位/邮箱）
- Keran Sun、Fei Yin 勾 **Corresponding author**；Keran Sun 填 ORCID 0000-0001-6065-8945（Fei Yin ORCID: 0000-0002-1075-0951）
- 每位作者需要邮箱验证链接（系统会发邮件）

## Step 4 — Files（上传入口对照）

| 系统入口 | 传什么 |
|---|---|
| Manuscript | manuscript_v1 导出的 PDF 或 Word（正文+图注；LaTeX 亦可，≤30MB）|
| Figures | fig1_v3_schematic、fig2_v3、fig3_dlpfc、fig4_v3_he、fig5_multi_cohort 的 PNG（每张单独传，系统里标注 Fig 1–5）|
| Supplementary Information | 合并 PDF：Extended Data 1–3 + supp_table_s1 + Box 1（标注为 Supplementary Information）|
| Reporting Summary | reporting_summary_draft 誊入官方模板后的 Word（模板在投稿系统下载）|
| Cover letter | cover_letter.md 内容粘贴或导出 PDF |
| Related manuscripts | 无（首次投稿）|

## Step 5 — Reviewers
- Suggested reviewers：从候选名单选 3–5（张世华/聂青/Lakkaraju/Agarwal/Lundeberg/樊荣，先查冲突）
- Opposed reviewers：按需

## Step 6 — Declarations
- Competing interests: 选 No
- Data availability / Code availability：粘贴 manuscript 对应两段（已含 GitHub + Zenodo DOI）
- Ethics: 选 "not applicable"（纯公共数据二次分析，无人体/动物实验）

## Step 7 — Final check & submit
- 系统生成合并 PDF 后**逐页检查**（重点：图是否清晰、图注是否完整、作者顺序、单位上标）
- 点 Submit，记下稿件编号（NCOMMS-XX-XXXXX）

## 投稿前最后一遍（我方）
- [ ] ref 38（Sun, Front Oncol）完整引用终核
- [ ] Methods 数字与审计脚本对账终跑一遍
- [ ] 确认 GitHub 仓库 public 且 README 正常渲染
