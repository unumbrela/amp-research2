# AMP 数据集构建报告

## 一、最终数据集概览

| 指标 | 数值 |
|------|------|
| 总序列数 | 25,622 |
| AMP 正样本 | 15,404 (60.1%) |
| 非 AMP 负样本 | 10,218 (39.9%) |
| 序列长度范围 | 5-50 AA (平均 22.5) |
| 训练集 | 20,497 |
| 验证集 | 2,562 |
| 测试集 | 2,563 |

### 属性覆盖率

| 属性 | 有标注数量 | 覆盖率 |
|------|-----------|--------|
| mic_value (MIC 值, log10 ug/ml) | 5,296 | 20.7% |
| is_hemolytic (溶血性) | 4,621 | 18.0% |
| is_toxic (细胞毒性) | 2,285 | 8.9% |

---

## 二、数据来源明细

### 已自动获取的数据源

| 数据源 | 获取方式 | 原始条数 | 去重后贡献 | 属性 |
|--------|---------|---------|-----------|------|
| **DRAMP general_amps** | HTTP 下载 | 8,782 | ~6,000+ | MIC (2,996), 溶血 (4,885), 毒性 (2,476) |
| **DRAMP synthetic_amps** | HTTP 下载 | 6,320 | ~4,000+ | 仅序列 + AMP 标签 |
| **UniProt AMP** (KW-0929) | REST API | 1,823 | ~1,500+ | 仅序列 + AMP 标签 |
| **UniProt non-AMP** | REST API | 3,750 | ~3,700 | 仅序列 (负样本) |
| **Diff-AMP** (本地) | 本地 CSV | 12,310 | ~4,000+ (大量与 DRAMP 重叠) | AMP 标签 |
| **AMPainter** (本地) | 本地 TXT | 6,524 | ~2,500+ | MIC-like scores (3,259) |

### MIC 值提取说明

DRAMP `Target_Organism` 字段包含自由文本描述，使用正则表达式提取 MIC/IC50/EC50 值：
- 支持单位: ug/ml, uM, mM, mg/ml
- uM 转 ug/ml: MW = 序列长度 * 110 Da
- 多个 MIC 值取几何平均
- 最终存储为 log10(ug/ml)

### 溶血性/毒性标签提取说明

从 DRAMP 的 `Hemolytic_activity` 和 `Cytotoxicity` 自由文本字段中，使用规则匹配分类：
- **非溶血**: "no hemolytic", "< 5% hemolysis", "HC50 > 200" 等
- **溶血**: "hemolytic at", "HC50 < 50", "> 20% hemolysis" 等
- **非毒性**: "no cytotoxic", "IC50 > 200" 等
- **毒性**: "cytotoxic", "IC50 < 50" 等

---

## 三、未能自动获取的数据源

以下数据源因技术原因无法自动爬取，需要手动获取：

### 1. DBAASP (Database of Antimicrobial Activity and Structure of Peptides)

**失败原因**: API 返回空响应 (content-length: 0)，疑似需要浏览器会话/CAPTCHA 验证。

**手动获取步骤**:
1. 打开浏览器访问 https://dbaasp.org/search
2. 设置筛选条件:
   - Peptide Type: Monomer
   - Sequence Length: 5 - 50
3. 点击 "Search" 搜索
4. 在结果页面点击 "Export" 按钮，选择 CSV 格式下载
5. 将下载的文件保存到 `esm_diffvae/data/raw/dbaasp_export.csv`

**下载后解析**: DBAASP 的 CSV 包含结构化的 MIC 数据 (organism, MIC value, unit)，解析脚本待实现。将该文件放入 `data/raw/` 后重新运行 `merge_and_clean.py` 即可整合。

**预期产出**: ~15,000-20,000 条 AMP，大量带有结构化 MIC 值。

### 2. APD3 (Antimicrobial Peptide Database)

**失败原因**: HTTPS SSL 证书错误，无法建立安全连接。

**手动获取步骤**:
1. 浏览器访问 https://aps.unmc.edu/
2. 进入 "Database" → "Download" 页面
3. 下载完整数据库 (FASTA 或 CSV 格式)
4. 保存到 `esm_diffvae/data/raw/apd3.csv` 或 `apd3.fasta`

**预期产出**: ~3,300 条经实验验证的 AMP。

### 3. HemoPI (Hemolytic Peptide Database)

**失败原因**: 服务器拒绝连接 (IIITD 服务器目前宕机)。

**手动获取步骤**:
1. 浏览器访问 https://webs.iiitd.edu.in/raghava/hemopi/
2. 进入 "Download" 页面
3. 下载 hemolytic 和 non-hemolytic 肽数据集
4. 保存到 `esm_diffvae/data/raw/hemopi_hemolytic.csv` 和 `hemopi_nonhemolytic.csv`

**预期产出**: ~1,000+ 条带溶血性标签的肽，可显著提升 `is_hemolytic` 属性覆盖率。

### 4. ToxinPred / ToxinDB

**失败原因**: 服务器不可靠，连接超时。

**手动获取步骤**:
1. 浏览器访问 https://webs.iiitd.edu.in/raghava/toxinpred/
2. 下载训练数据集 (包含 toxic 和 non-toxic 肽)
3. 保存到 `esm_diffvae/data/raw/toxinpred.csv`

**预期产出**: ~3,000 条带毒性标签的肽，可提升 `is_toxic` 属性覆盖率。

### 5. CAMP3 (Collection of Anti-Microbial Peptides)

**失败原因**: SSL 证书问题。

**手动获取步骤**:
1. 浏览器访问 http://www.camp3.bicnirrh.res.in/
2. 进入 "Download" 页面
3. 下载 AMP 序列数据
4. 保存到 `esm_diffvae/data/raw/camp3.csv`

---

## 四、数据整合说明

手动下载的数据放入 `esm_diffvae/data/raw/` 后，需要：

1. **编写对应解析器** (如 `parse_dbaasp.py`)，将原始格式转为统一 CSV:
   - 列: `sequence, is_amp, source, mic_value, is_toxic, is_hemolytic`
2. **重新运行合并脚本**:
   ```bash
   python esm_diffvae/data/crawl/merge_and_clean.py
   ```
3. **重新计算 PLM 嵌入**:
   ```bash
   python esm_diffvae/data/compute_embeddings.py --backend prot_t5 --model prot_t5_xl_half
   ```

---

## 五、文件清单

```
esm_diffvae/data/
├── raw/                          # 原始数据 (各来源)
│   ├── dramp_general.csv         # DRAMP 天然 AMP (8,782 条)
│   ├── dramp_synthetic.csv       # DRAMP 合成 AMP (6,320 条)
│   ├── uniprot_amp.csv           # UniProt AMP (1,823 条)
│   ├── uniprot_nonamp.csv        # UniProt 非 AMP (3,750 条)
│   ├── diffamp.csv               # Diff-AMP 本地数据 (12,310 条)
│   ├── ampainter.csv             # AMPainter 本地数据 (6,524 条)
│   ├── dramp_general_raw.txt     # DRAMP 原始文件 (备份)
│   └── dramp_synthetic_raw.txt   # DRAMP 原始文件 (备份)
├── processed/                    # 处理后数据 (最终输出)
│   ├── train.csv                 # 训练集 (20,497 条)
│   ├── val.csv                   # 验证集 (2,562 条)
│   ├── test.csv                  # 测试集 (2,563 条)
│   ├── all.csv                   # 全部数据 (25,622 条)
│   └── stats.json                # 数据集统计信息
└── crawl/                        # 爬取/解析脚本
    ├── crawl_dramp.py            # DRAMP 爬取
    ├── crawl_uniprot.py          # UniProt 爬取
    ├── parse_local_sources.py    # 本地数据解析
    └── merge_and_clean.py        # 合并去重主脚本
```
