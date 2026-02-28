// 抗菌肽生成项目调研数据
// Academic Journal Style - Data Layer

export interface GenerationModel {
  name: string;
  year: number;
  architecture: string;
  journal: string;
  citations: number;
  github: string | null;
  stars: number | null;
  features: string[];
  description: string;
  doi: string;
}

export interface EvaluationTool {
  name: string;
  category: string;
  year: number;
  description: string;
  github: string | null;
  features: string[];
}

export interface TimelineEvent {
  year: number;
  title: string;
  description: string;
  type: "model" | "tool" | "database";
}

export interface GithubRepo {
  name: string;
  fullName: string;
  url: string;
  stars: number;
  description: string;
  language: string;
  features: string[];
}

// 生成模型数据
export const generationModels: GenerationModel[] = [
  {
    name: "HydrAMP",
    year: 2023,
    architecture: "条件变分自编码器 (cVAE)",
    journal: "Nature Communications",
    citations: 177,
    github: "https://github.com/szczurek-lab/hydramp",
    stars: 56,
    features: ["无条件生成", "类似物生成", "AMP分类", "MIC分类"],
    description: "基于条件VAE的成熟模型，提供无约束生成和类似物生成两种模式。将序列编码到连续的低维潜在空间，再从该空间解码生成新序列。支持AMP/非AMP条件生成，内置MIC分类器和AMP分类器。",
    doi: "10.1038/s41467-023-36994-z",
  },
  {
    name: "Diff-AMP",
    year: 2024,
    architecture: "扩散+GAN+强化学习",
    journal: "Briefings in Bioinformatics",
    citations: 80,
    github: "https://github.com/wrab12/diff-amp",
    stars: 17,
    features: ["序列生成", "AMP识别", "多属性预测", "迭代优化"],
    description: "创新的四合一集成框架，将扩散模型和GAN结合用于生成，整合了基于预训练模型的分类器和基于CNN的属性预测器，最后通过强化学习进行优化。提供完整的生成-评估-优化闭环。",
    doi: "10.1093/bib/bbae078",
  },
  {
    name: "AMP-Designer",
    year: 2025,
    architecture: "大型语言模型 (LLM)",
    journal: "Science Advances",
    citations: 82,
    github: null,
    stars: null,
    features: ["从头设计", "广谱抗菌", "快速验证", "多条件生成"],
    description: "基于LLM的基础模型，11天内实现18个广谱抗革兰氏阴性菌AMP的从头设计，体外验证成功率94.4%。从设计到验证仅需48天，展示了LLM在AMP设计中的巨大潜力。",
    doi: "10.1126/sciadv.ads8932",
  },
  {
    name: "AMPGen",
    year: 2025,
    architecture: "进化信息+扩散模型",
    journal: "Communications Biology",
    citations: 17,
    github: null,
    stars: null,
    features: ["靶标特异性", "进化信息保留", "从头设计"],
    description: "利用多序列比对(MSA)中的进化信息，结合条件扩散模型进行靶标特异性AMP的从头设计。创新性地将进化信息融入生成过程。",
    doi: "10.1038/s42003-025-08282-7",
  },
  {
    name: "BroadAMP-GPT",
    year: 2025,
    architecture: "GPT模型",
    journal: "Gut Microbes",
    citations: 9,
    github: null,
    stars: null,
    features: ["广谱抗菌", "多层筛选", "ESKAPE病原体"],
    description: "集成计算-实验框架，使用GPT模型生成广谱AMP，配合多层筛选策略，专门针对多重耐药ESKAPE病原体。避免了GAN训练不稳定和VAE空间分布约束问题。",
    doi: "10.1080/19490976.2025.2523811",
  },
  {
    name: "潜在扩散模型",
    year: 2025,
    architecture: "VAE+潜在扩散+条件生成",
    journal: "Science Advances",
    citations: 52,
    github: null,
    stars: null,
    features: ["多样性生成", "高效设计", "分子动力学验证"],
    description: "结合VAE和潜在扩散模型进行条件生成，实现多样且高效的AMP从头设计。通过分子动力学模拟验证生成肽的结构和功能。",
    doi: "10.1126/sciadv.xxx",
  },
];

// 模型架构对比数据
export const architectureComparison = [
  {
    type: "变分自编码器 (VAE)",
    representatives: "HydrAMP, PepCVAE",
    principle: "将序列编码到连续的低维潜在空间，再从该空间解码生成新序列",
    pros: "能够学习平滑的潜在空间，适合生成与已知序列相似的变体",
    cons: "生成序列的多样性可能受限",
    diversityScore: 65,
    qualityScore: 75,
    speedScore: 85,
    stabilityScore: 90,
  },
  {
    type: "生成对抗网络 (GAN)",
    representatives: "AMPGAN, Diff-AMP",
    principle: "通过生成器和判别器的对抗性训练，生成高质量序列",
    pros: "能够生成高质量、高真实感的序列",
    cons: "训练过程不稳定，容易出现模式崩溃",
    diversityScore: 70,
    qualityScore: 85,
    speedScore: 80,
    stabilityScore: 55,
  },
  {
    type: "扩散模型 (Diffusion)",
    representatives: "AMPGen, ProT-Diff",
    principle: "从随机噪声开始，通过学习到的去噪过程逐步生成结构化的序列",
    pros: "生成质量和多样性均很高",
    cons: "推理速度相对较慢，计算成本较高",
    diversityScore: 90,
    qualityScore: 90,
    speedScore: 45,
    stabilityScore: 80,
  },
  {
    type: "大型语言模型 (LLM)",
    representatives: "AMP-Designer, BroadAMP-GPT",
    principle: "将肽序列视为语言，利用预训练语言模型的强大能力进行生成",
    pros: "能够利用海量预训练知识，支持复杂的条件生成",
    cons: "需要大量计算资源，对小样本任务可能过拟合",
    diversityScore: 85,
    qualityScore: 88,
    speedScore: 70,
    stabilityScore: 75,
  },
];

// 评估工具数据
export const evaluationTools: EvaluationTool[] = [
  {
    name: "LLAMP",
    category: "MIC预测",
    year: 2025,
    description: "基于ESM-2语言模型微调，可预测针对特定菌种的最小抑菌浓度(MIC)值",
    github: "https://github.com/GIST-CSBL/LLAMP",
    features: ["物种感知", "ESM-2微调", "回归预测"],
  },
  {
    name: "ANIA",
    category: "MIC预测",
    year: 2026,
    description: "基于Inception-Attention网络的MIC预测模型，性能优越",
    github: null,
    features: ["Inception架构", "注意力机制", "高精度"],
  },
  {
    name: "MBC-Attention",
    category: "MIC预测",
    year: 2023,
    description: "深度学习回归模型，利用注意力机制预测抗菌肽的MIC值",
    github: null,
    features: ["注意力机制", "回归模型", "多菌种"],
  },
  {
    name: "HAPPENN",
    category: "溶血性预测",
    year: 2020,
    description: "经典的基于神经网络的溶血活性分类器，被广泛引用",
    github: null,
    features: ["神经网络", "二分类", "高引用"],
  },
  {
    name: "AmpLyze",
    category: "溶血性预测",
    year: 2025,
    description: "能够直接预测溶血性指标HC50的具体数值，而非仅分类",
    github: null,
    features: ["HC50预测", "回归模型", "定量评估"],
  },
  {
    name: "Diff-AMP (预测模块)",
    category: "多属性预测",
    year: 2024,
    description: "基于CNN的多属性预测模块，可同时预测抗革兰氏阳性/阴性菌、抗真菌等20种活性",
    github: "https://github.com/wrab12/diff-amp",
    features: ["多标签分类", "CNN架构", "20种属性"],
  },
  {
    name: "PyAMPA / AMPSolve",
    category: "综合评估",
    year: 2024,
    description: "高通量预测和优化平台，涵盖溶血性、毒性、半衰期和抗菌谱等多维度评估",
    github: null,
    features: ["高通量", "多维评估", "优化建议"],
  },
  {
    name: "modlAMP",
    category: "物理化学性质",
    year: 2017,
    description: "功能强大的Python包，支持计算疏水性、疏水矩、净电荷、等电点等多种关键物理化学性质",
    github: "https://github.com/alexarnimueller/modlAMP",
    features: ["描述符计算", "序列分析", "可视化"],
  },
  {
    name: "DBAASP",
    category: "数据库+预测",
    year: 2021,
    description: "最全面的AMP数据库之一，提供在线预测工具和API，手动整理了大量肽的序列、结构和活性数据",
    github: null,
    features: ["手动策划", "API接口", "在线预测"],
  },
  {
    name: "CalcAMP",
    category: "活性预测",
    year: 2023,
    description: "机器学习模型，可预测AMP对革兰氏阳性和阴性菌的活性",
    github: "https://github.com/CDDLeiden/CalcAMP",
    features: ["革兰氏阳性", "革兰氏阴性", "机器学习"],
  },
];

// GitHub仓库数据
export const githubRepos: GithubRepo[] = [
  {
    name: "hydramp",
    fullName: "szczurek-lab/hydramp",
    url: "https://github.com/szczurek-lab/hydramp",
    stars: 56,
    description: "HydrAMP: 基于条件VAE的抗菌肽生成与评估模型",
    language: "Python (TensorFlow)",
    features: ["无条件生成", "类似物生成", "AMP分类", "MIC分类"],
  },
  {
    name: "diff-amp",
    fullName: "wrab12/diff-amp",
    url: "https://github.com/wrab12/diff-amp",
    stars: 17,
    description: "Diff-AMP: 四合一AMP框架（生成+识别+预测+优化）",
    language: "Python (PyTorch)",
    features: ["扩散生成", "AMP识别", "多属性预测", "强化学习优化"],
  },
  {
    name: "LLAMP",
    fullName: "GIST-CSBL/LLAMP",
    url: "https://github.com/GIST-CSBL/LLAMP",
    stars: 1,
    description: "LLAMP: 基于ESM-2的物种感知MIC预测模型",
    language: "Python (PyTorch)",
    features: ["MIC预测", "ESM-2微调", "物种感知"],
  },
  {
    name: "Antimicrobial-peptide-generation",
    fullName: "gc-js/Antimicrobial-peptide-generation",
    url: "https://github.com/gc-js/Antimicrobial-peptide-generation",
    stars: 18,
    description: "AMP生成、分类和回归的完整工具集",
    language: "Python",
    features: ["GAN生成", "分类", "回归预测", "HuggingFace部署"],
  },
  {
    name: "modlAMP",
    fullName: "alexarnimueller/modlAMP",
    url: "https://github.com/alexarnimueller/modlAMP",
    stars: 60,
    description: "肽序列生成、描述符计算和序列分析的Python包",
    language: "Python",
    features: ["描述符计算", "序列生成", "数据库查询", "可视化"],
  },
  {
    name: "CalcAMP",
    fullName: "CDDLeiden/CalcAMP",
    url: "https://github.com/CDDLeiden/CalcAMP",
    stars: 5,
    description: "机器学习预测AMP对革兰氏阳性和阴性菌的活性",
    language: "Python",
    features: ["活性预测", "革兰氏分类", "机器学习"],
  },
  {
    name: "Awesome-AMP-Design",
    fullName: "ruihan-dong/Awesome-AMP-Design",
    url: "https://github.com/ruihan-dong/Awesome-AMP-Design",
    stars: 26,
    description: "AI驱动的抗菌肽设计论文精选列表",
    language: "Markdown",
    features: ["论文列表", "分类整理", "持续更新"],
  },
];

// 时间线数据
export const timeline: TimelineEvent[] = [
  { year: 2017, title: "modlAMP发布", description: "首个专门用于AMP描述符计算和分析的Python包", type: "tool" },
  { year: 2020, title: "HAPPENN发布", description: "基于神经网络的溶血活性预测工具，成为该领域标杆", type: "tool" },
  { year: 2021, title: "AMPGAN v2", description: "GAN方法在AMP生成中的重要应用，引入条件生成", type: "model" },
  { year: 2021, title: "DBAASP v3", description: "AMP数据库重大更新，新增预测工具和API", type: "database" },
  { year: 2023, title: "HydrAMP发表", description: "Nature Communications发表，cVAE方法的里程碑，开源代码", type: "model" },
  { year: 2023, title: "MBC-Attention", description: "注意力机制首次用于AMP的MIC值回归预测", type: "tool" },
  { year: 2024, title: "Diff-AMP发表", description: "首个四合一集成框架，整合生成、识别、预测和优化", type: "model" },
  { year: 2024, title: "PyAMPA发布", description: "高通量AMP预测和优化平台，涵盖多维度评估", type: "tool" },
  { year: 2025, title: "AMP-Designer", description: "Science Advances发表，LLM方法实现94.4%验证成功率", type: "model" },
  { year: 2025, title: "AMPGen", description: "进化信息+扩散模型实现靶标特异性AMP设计", type: "model" },
  { year: 2025, title: "BroadAMP-GPT", description: "GPT模型针对ESKAPE病原体生成广谱AMP", type: "model" },
  { year: 2025, title: "LLAMP", description: "基于ESM-2的物种感知MIC预测模型", type: "tool" },
];

// 实现方案数据
export const implementationSteps = [
  {
    step: 1,
    title: "环境搭建与数据准备",
    description: "搭建Python开发环境，安装核心依赖库，从DBAASP和APD数据库下载AMP序列及活性数据，构建训练集和测试集。",
    tools: ["Python 3.8+", "PyTorch / TensorFlow", "Biopython", "modlAMP", "pandas"],
    details: "建议使用conda创建独立环境。从DBAASP下载包含序列、MIC值、靶标信息的完整数据集。数据清洗包括去除冗余序列、统一序列长度范围（通常10-50个氨基酸）、标注分类标签。",
  },
  {
    step: 2,
    title: "无条件序列生成",
    description: "参考HydrAMP的无条件生成模式，训练一个VAE模型，从潜在空间随机采样并解码生成全新的肽序列。",
    tools: ["HydrAMP代码库", "TensorFlow/Keras", "条件VAE架构"],
    details: "核心思路：在大量肽序列上训练VAE，学习序列的潜在表示。生成时从标准正态分布中采样潜在向量，通过解码器生成新序列。可以加入AMP条件标签，使生成偏向AMP序列。",
  },
  {
    step: 3,
    title: "条件变体序列生成",
    description: "参考HydrAMP的类似物生成模式，将给定种子序列编码到潜在空间，对潜在向量进行微小扰动后解码，生成变体序列。",
    tools: ["HydrAMP代码库", "潜在空间操作", "高斯噪声扰动"],
    details: "将输入序列通过编码器映射到潜在空间，对潜在向量添加可控的高斯噪声（噪声幅度控制变异程度），然后通过解码器生成新序列。还可以在潜在空间中进行插值，生成两个序列之间的过渡变体。",
  },
  {
    step: 4,
    title: "基本性质评估模块",
    description: "使用modlAMP和Biopython库，构建物理化学性质计算模块，输出分子量、长度、电荷、疏水性、等电点等指标。",
    tools: ["modlAMP", "Biopython", "NumPy"],
    details: "关键性质包括：分子量、序列长度、净电荷（pH 7.0）、疏水性（Eisenberg量表）、疏水矩、等电点、氨基酸组成比例。这些性质与AMP的活性和稳定性密切相关。",
  },
  {
    step: 5,
    title: "MIC值预测模型",
    description: "参考LLAMP的思路，使用ESM-2预训练模型提取序列特征，训练回归模型预测MIC值。",
    tools: ["ESM-2", "PyTorch", "scikit-learn", "DBAASP数据"],
    details: "使用ESM-2提取序列的高维特征表示，然后训练梯度提升树或简单神经网络进行MIC值回归预测。需要DBAASP中包含MIC值的训练数据。可以针对不同菌种分别训练模型。",
  },
  {
    step: 6,
    title: "抗菌谱与安全性评估",
    description: "构建多标签分类模型，预测抗革兰氏阳性/阴性菌、抗真菌、溶血性等多种属性。",
    tools: ["Diff-AMP预测模块", "CNN/Transformer", "多标签分类"],
    details: "参考Diff-AMP的多属性预测模块设计。输入为肽序列特征，输出为多个二分类概率。可以预测的属性包括：抗革兰氏阳性菌、抗革兰氏阴性菌、抗真菌、抗病毒、溶血性、细胞毒性等。",
  },
];

// 论文引用趋势数据（用于图表）
export const citationTrends = [
  { year: 2020, vae: 15, gan: 25, diffusion: 2, llm: 0 },
  { year: 2021, vae: 28, gan: 42, diffusion: 5, llm: 3 },
  { year: 2022, vae: 35, gan: 48, diffusion: 18, llm: 8 },
  { year: 2023, vae: 52, gan: 55, diffusion: 45, llm: 22 },
  { year: 2024, vae: 60, gan: 58, diffusion: 78, llm: 55 },
  { year: 2025, vae: 65, gan: 60, diffusion: 95, llm: 88 },
];

// 评估维度雷达图数据
export const evaluationDimensions = [
  { dimension: "抗菌活性", description: "MIC值预测", importance: 95 },
  { dimension: "溶血性", description: "HC50预测", importance: 90 },
  { dimension: "细胞毒性", description: "细胞毒性评估", importance: 85 },
  { dimension: "抗菌谱", description: "革兰氏阳性/阴性/真菌", importance: 80 },
  { dimension: "稳定性", description: "血清半衰期", importance: 75 },
  { dimension: "物化性质", description: "电荷/疏水性/等电点", importance: 70 },
];
