/**
 * AMP Forge Landing Page
 * Project-first narrative with retained research context.
 */

import { useEffect, useRef, useState, type ReactNode } from "react";
import { motion, useInView } from "framer-motion";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts";
import {
  ArrowRight,
  BookOpen,
  CheckCircle2,
  ChevronDown,
  Cpu,
  Database,
  ExternalLink,
  FlaskConical,
  Github,
  Languages,
  Layers,
  Rocket,
  ShieldCheck,
  Sparkles,
  Target,
  Workflow,
} from "lucide-react";

type Locale = "en" | "zh";

function t(locale: Locale, en: string, zh: string) {
  return locale === "zh" ? zh : en;
}

const MODELS_IMG =
  "https://d2xsxph8kpxj0f.cloudfront.net/310519663390962009/nwNByvtJSze5gMGiNTNhXC/generation-models-imLzwiPRbyJQnYUeagQ5pB.webp";
const EVAL_IMG =
  "https://d2xsxph8kpxj0f.cloudfront.net/310519663390962009/nwNByvtJSze5gMGiNTNhXC/evaluation-pipeline-CkDvrw87vYQELZsHzvSFE7.webp";
const BG_IMG =
  "https://d2xsxph8kpxj0f.cloudfront.net/310519663390962009/nwNByvtJSze5gMGiNTNhXC/abstract-peptide-bg-7vmSS9d64cv7cVh5spxu7y.webp";

const dataSplit = [
  { split: "Train", value: 22828 },
  { split: "Val", value: 2854 },
  { split: "Test", value: 2854 },
];

const propertyCoverage = [
  { dimension: "MIC", value: 20.7 },
  { dimension: "Hemolysis", value: 18.0 },
  { dimension: "Toxicity", value: 8.9 },
  { dimension: "AMP Label", value: 100 },
  { dimension: "Length", value: 100 },
];

const projectMilestones = [
  {
    title: { en: "Data Foundation", zh: "数据底座" },
    desc: {
      en: "Integrated multi-source AMP records and unified cleaning into a trainable data foundation.",
      zh: "多源数据整合与统一清洗，形成可训练数据底座。",
    },
    status: "done",
  },
  {
    title: { en: "Model Core", zh: "模型核心" },
    desc: {
      en: "Connected the full ESM-DiffVAE v8 path from training to generation.",
      zh: "ESM-DiffVAE v8 训练与生成链路打通。",
    },
    status: "done",
  },
  {
    title: { en: "Evaluation Stack", zh: "评估体系" },
    desc: {
      en: "Implemented a multi-metric stack with physicochemical and variant analysis workflows.",
      zh: "多维指标、理化性质与变体分析流程落地。",
    },
    status: "done",
  },
  {
    title: { en: "Project Packaging", zh: "项目包装" },
    desc: {
      en: "Shipped bilingual documentation, GitHub presentation, and a project-first frontend narrative.",
      zh: "双语文档、GitHub 展示与前端项目化叙事。",
    },
    status: "in_progress",
  },
  {
    title: { en: "Next Expansion", zh: "下一阶段扩展" },
    desc: {
      en: "Next: experimental validation, model compression, and deployment-ready interfaces.",
      zh: "补充实验验证、模型压缩与部署接口。",
    },
    status: "next",
  },
];

const variantModes = [
  {
    mode: "c_sub",
    desc: {
      en: "Keep the N-terminus and substitute C-terminal residues.",
      zh: "保留 N 端，替换 C 端末尾位点",
    },
    focus: { en: "local edits", zh: "局部替换" },
  },
  {
    mode: "c_ext",
    desc: {
      en: "Keep the parent sequence and extend the C-terminus.",
      zh: "保留母序列并在 C 端延伸",
    },
    focus: { en: "extension", zh: "序列扩展" },
  },
  {
    mode: "c_trunc",
    desc: {
      en: "Truncate then regenerate C-terminal segment.",
      zh: "截断后重生 C 端",
    },
    focus: { en: "reconstruction", zh: "重构优化" },
  },
  {
    mode: "tag",
    desc: {
      en: "Append commonly used peptide tags.",
      zh: "追加常见 peptide tag",
    },
    focus: { en: "engineering tags", zh: "工程标签" },
  },
  {
    mode: "latent",
    desc: {
      en: "Perturb latent vectors before decoding.",
      zh: "潜空间扰动后解码",
    },
    focus: { en: "global diversity", zh: "全局多样性" },
  },
];

const methodLandscape = [
  {
    name: "HydrAMP (cVAE)",
    architecture: { en: "Conditional VAE", zh: "条件 VAE" },
    strengths: {
      en: "Strong for goal-guided analog design with a clear training path.",
      zh: "适合围绕先验目标做类似物设计，训练路径清晰。",
    },
    limitations: {
      en: "More optimization-oriented; limited for broad de novo diversity and transfer.",
      zh: "更偏向条件优化，跨目标迁移与 de novo 多样性受限。",
    },
    repo: "https://github.com/szczurek-lab/hydramp",
  },
  {
    name: "Diff-AMP",
    architecture: { en: "Generate + Identify + Predict + Optimize", zh: "生成 + 识别 + 预测 + 优化" },
    strengths: {
      en: "Comprehensive workflow with strong modular capabilities.",
      zh: "任务链路完整，模块化能力较强。",
    },
    limitations: {
      en: "High system complexity; replacing or iterating one module can be expensive.",
      zh: "模块多导致工程复杂度高，单模块替换与迭代成本较高。",
    },
    repo: "https://github.com/wrab12/diff-amp",
  },
  {
    name: "AMPGen",
    architecture: { en: "Evolutionary priors + Diffusion", zh: "进化信息 + 扩散" },
    strengths: {
      en: "Good at target-aware design and sequence conservation modeling.",
      zh: "对靶向设计和序列保守性建模较好。",
    },
    limitations: {
      en: "Limited public reproducibility artifacts and higher reuse barrier.",
      zh: "公开复现材料有限，工程复用门槛较高。",
    },
    repo: null,
  },
  {
    name: "LLM-based AMP Foundation",
    architecture: { en: "Large-model sequence generation", zh: "大模型序列生成" },
    strengths: {
      en: "Strong generation capacity with broader semantic coverage.",
      zh: "生成能力强，覆盖更广泛语义模式。",
    },
    limitations: {
      en: "High training cost and lower controllability for small teams.",
      zh: "训练资源需求高，面向小团队的成本和可控性仍是挑战。",
    },
    repo: null,
  },
];

const projectAdvantages = [
  {
    en: "A reproducible train-generate-evaluate loop that reduces script fragmentation.",
    zh: "统一成一个可复现的训练-生成-评估闭环，减少多脚本割裂。",
  },
  {
    en: "Latent diffusion as the core, balancing de novo generation and controllable variants.",
    zh: "以潜空间扩散为核心，兼顾 de novo 生成和可控变体生成。",
  },
  {
    en: "Multiple generation modes (c_sub / c_ext / latent / tag) for rapid experiment comparison.",
    zh: "支持多生成模式（c_sub / c_ext / latent / tag），便于快速实验对比。",
  },
  {
    en: "Standardized outputs (FASTA / JSON / plots) for version tracking and review.",
    zh: "结果目录标准化输出（FASTA / JSON / 可视化），方便版本追踪和复盘。",
  },
];

const unconditionalSamples = [
  { id: "generated_1", len: 7, sequence: "RNDFNPM" },
  { id: "generated_4", len: 46, sequence: "KKCWRQCYRWPWWCNCRKCCRYVCVTYRRNTRYTRSQQKHKPQNFP" },
  { id: "generated_11", len: 50, sequence: "WRRFKRYCKKHWRRYDMHRPRRKTHLPRNYKWRRRHRHRKRRRYKQKDRQ" },
  { id: "generated_31", len: 29, sequence: "WITTWTKWLMLAIHMFHKFHKFKTKKSGQ" },
  { id: "generated_56", len: 24, sequence: "WWDLWWWIKNWWPCHKHWWWKPYC" },
  { id: "generated_66", len: 29, sequence: "GWGKSIVKCGKGPIASAFKKNWQAGYKCP" },
  { id: "generated_78", len: 27, sequence: "KLKFILKAAWALLWGAFSFYTKWNWKY" },
  { id: "generated_100", len: 21, sequence: "WAPWAWWWLAKWAPSWPKWPR" },
];

const variantSamples = [
  { mode: "c_ext", sequence: "GIGKFLHSAKKFGKAFVGEIMNSG", identity: 0.9583, editDistance: 1 },
  { mode: "c_ext", sequence: "GIGKFLHSAKKFGKAFVGEIMNSYQ", identity: 0.92, editDistance: 2 },
  { mode: "c_sub", sequence: "GIGKFLHSAKKFGKAFVGEIMNC", identity: 0.9565, editDistance: 1 },
  { mode: "c_trunc", sequence: "GIGKFLHSAKKFGKAFVGEIMQS", identity: 0.9565, editDistance: 1 },
  { mode: "latent", sequence: "GIHKFLHKAKKFAKQFLGMIMNK", identity: 0.6957, editDistance: 7 },
  { mode: "tag_his8", sequence: "GIGKFLHSAKKFGKAFVGEIMNSHHHHHHHH", identity: 0.7419, editDistance: 8 },
];

const references = [
  {
    id: 1,
    text: "Wang et al. (2025). Discovery of antimicrobial peptides with notable antibacterial potency by an LLM-based foundation model.",
    doi: "10.1126/sciadv.ads8932",
  },
  {
    id: 2,
    text: "Szymczak et al. (2023). Discovering highly potent antimicrobial peptides with deep generative model HydrAMP.",
    doi: "10.1038/s41467-023-36994-z",
  },
  {
    id: 3,
    text: "Wang et al. (2024). Diff-AMP: tailored designed antimicrobial peptide framework with all-in-one generation, identification, prediction and optimization.",
    doi: "10.1093/bib/bbae078",
  },
  {
    id: 4,
    text: "Jin et al. (2025). AMPGen: an evolutionary information-reserved and diffusion-driven generative model for de novo design.",
    doi: "10.1038/s42003-025-08282-7",
  },
];

function Section({ children, id, className = "" }: { children: ReactNode; id: string; className?: string }) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-80px" });
  return (
    <motion.section
      ref={ref}
      id={id}
      initial={{ opacity: 0, y: 30 }}
      animate={isInView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.55, ease: "easeOut" }}
      className={`py-16 md:py-24 ${className}`}
    >
      {children}
    </motion.section>
  );
}

function SectionNumber({ num }: { num: string }) {
  return <span className="section-number select-none mr-4 inline-block">{num}</span>;
}

function NavBar({ locale, setLocale }: { locale: Locale; setLocale: (locale: Locale) => void }) {
  const [scrolled, setScrolled] = useState(false);
  const [activeSection, setActiveSection] = useState("hero");

  useEffect(() => {
    const onScroll = () => {
      setScrolled(window.scrollY > 60);
      const sections = [
        "hero",
        "overview",
        "landscape",
        "architecture",
        "data",
        "generation",
        "evaluation",
        "roadmap",
        "references",
      ];
      for (const s of sections.reverse()) {
        const el = document.getElementById(s);
        if (el && el.getBoundingClientRect().top < 200) {
          setActiveSection(s);
          break;
        }
      }
    };

    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const navItems = [
    { id: "overview", label: t(locale, "Overview", "项目概览") },
    { id: "landscape", label: t(locale, "Method Gap", "方法对比") },
    { id: "architecture", label: t(locale, "Architecture", "架构设计") },
    { id: "data", label: t(locale, "Data & Training", "数据与训练") },
    { id: "generation", label: t(locale, "Generation", "生成能力") },
    { id: "evaluation", label: t(locale, "Evaluation", "评估结果") },
    { id: "roadmap", label: t(locale, "Roadmap", "路线图") },
  ];

  return (
    <nav
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled ? "bg-background/95 backdrop-blur-sm shadow-sm border-b border-border" : ""
      }`}
    >
      <div className="container flex items-center justify-between h-14 gap-3">
        <a
          href="#hero"
          className="font-[family-name:var(--font-display)] text-lg font-semibold tracking-tight text-foreground hover:text-primary transition-colors"
        >
          AMP Forge
        </a>
        <div className="hidden md:flex items-center gap-1 flex-1 justify-center">
          {navItems.map((item) => (
            <a
              key={item.id}
              href={`#${item.id}`}
              className={`px-3 py-1.5 text-sm rounded-md transition-colors ${
                activeSection === item.id
                  ? "text-primary font-medium bg-primary/5"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {item.label}
            </a>
          ))}
        </div>
        <button
          type="button"
          onClick={() => setLocale(locale === "en" ? "zh" : "en")}
          className="inline-flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-md border border-border bg-background hover:bg-secondary transition-colors"
          aria-label={t(locale, "Switch language", "切换语言")}
        >
          <Languages className="w-3.5 h-3.5" />
          {locale === "en" ? "中文" : "EN"}
        </button>
      </div>
    </nav>
  );
}

function HeroSection({ locale }: { locale: Locale }) {
  return (
    <section id="hero" className="relative min-h-[85vh] flex items-end pb-16 pt-20">
      <div className="absolute inset-0 bg-cover bg-center opacity-[0.08]" style={{ backgroundImage: `url(${BG_IMG})` }} />
      <div className="container relative z-10">
        <div className="max-w-4xl">
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="text-sm uppercase tracking-[0.2em] text-primary font-medium mb-6"
          >
            {t(locale, "AMP Forge · Project Homepage · 2026", "AMP Forge · 项目主页 · 2026")}
          </motion.p>
          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.35, duration: 0.8 }}
            className="text-4xl md:text-6xl lg:text-7xl font-bold leading-[1.1] mb-8 text-foreground"
          >
            {t(locale, "Antimicrobial Peptide Generation", "抗菌肽生成项目")}
            <br />
            <span className="text-primary">{t(locale, "From Research to Deployable System", "从研究到可落地系统")}</span>
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.55 }}
            className="text-lg md:text-xl text-muted-foreground leading-relaxed max-w-3xl mb-10"
          >
            {t(
              locale,
              "Antimicrobial resistance keeps rising, while wet-lab AMP discovery remains costly, slow, and low-coverage. AMP Forge turns this challenge into an end-to-end reproducible system and tells one coherent story: problem context, method gaps, our design, and real outputs.",
              "抗生素耐药问题持续加剧，而湿实验筛选抗菌肽成本高、周期长、覆盖有限。AMP Forge 以“可复现工程”为中心，把项目背景、方法对比、模型实现与真实生成结果串成一条完整叙事链。"
            )}
          </motion.p>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.75 }}
            className="flex flex-wrap gap-3 text-sm text-muted-foreground"
          >
            <span className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-secondary">
              <Database className="w-3.5 h-3.5" /> {t(locale, "28,536 training-ready sequences", "28,536 条训练级序列")}
            </span>
            <span className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-secondary">
              <Layers className="w-3.5 h-3.5" /> ESM + VAE + Latent Diffusion
            </span>
            <span className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-secondary">
              <Workflow className="w-3.5 h-3.5" /> {t(locale, "Train-Generate-Evaluate loop", "训练-生成-评估闭环")}
            </span>
            <span className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-secondary">
              <Sparkles className="w-3.5 h-3.5" /> {t(locale, "Controllable variant generation", "可控变体生成")}
            </span>
          </motion.div>
        </div>
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.1 }}
          className="absolute bottom-8 left-1/2 -translate-x-1/2"
        >
          <ChevronDown className="w-5 h-5 text-muted-foreground animate-bounce" />
        </motion.div>
      </div>
    </section>
  );
}

function OverviewSection({ locale }: { locale: Locale }) {
  return (
    <Section id="overview">
      <div className="container">
        <div className="flex items-start mb-10">
          <SectionNumber num="01." />
          <div>
            <h2 className="text-3xl md:text-4xl font-bold mb-2">{t(locale, "Project Overview", "项目概览")}</h2>
            <p className="text-muted-foreground text-lg mt-3">{t(locale, "Project Scope", "AMP Forge 的目标、边界与核心能力")}</p>
            <div className="w-16 h-0.5 bg-primary mt-3" />
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-16">
          <div className="lg:col-span-8 prose-academic">
            <p>
              {t(
                locale,
                "AMP Forge is built for computational AMP design. The goal is not just a runnable model, but a reproducible workflow that unifies biological context, model strategy, and engineering execution.",
                "AMP Forge 面向抗菌肽（AMP）计算设计，目标不是只做一个“能跑”的模型，而是把背景问题、技术路线和工程落地统一到同一套可复现流程中。"
              )}
            </p>
            <p>
              {t(
                locale,
                "The page now follows one complete story arc: project background, limitations of existing approaches, our ESM-DiffVAE design choices, and real generated outputs with evaluation evidence.",
                "页面结构从项目背景出发，先说明现有代表方法及其不足，再引出我们的 ESM-DiffVAE 路线，最后给出真实输出样例和评估结果，让“为什么做、怎么做、做到了什么”形成闭环。"
              )}
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-8">
              <div className="bg-card rounded-lg border border-border p-5">
                <p className="text-sm uppercase tracking-wider text-muted-foreground mb-2">{t(locale, "Problem", "问题")}</p>
                <p className="font-semibold mb-2 flex items-center gap-2">
                  <Target className="w-4 h-4 text-primary" /> {t(locale, "Find effective AMPs in vast sequence space", "在巨大序列空间中寻找有效 AMP")}
                </p>
                <p className="text-sm text-muted-foreground">{t(locale, "Balance activity, safety, diversity, and synthesizability.", "兼顾活性、安全性、多样性与可合成性。")}</p>
              </div>

              <div className="bg-card rounded-lg border border-border p-5">
                <p className="text-sm uppercase tracking-wider text-muted-foreground mb-2">{t(locale, "Solution", "方案")}</p>
                <p className="font-semibold mb-2 flex items-center gap-2">
                  <Cpu className="w-4 h-4 text-primary" /> ESM-DiffVAE v8
                </p>
                <p className="text-sm text-muted-foreground">{t(locale, "PLM features + VAE encode/decode + latent diffusion sampling.", "PLM 特征 + VAE 编解码 + 潜空间扩散采样。")}</p>
              </div>

              <div className="bg-card rounded-lg border border-border p-5">
                <p className="text-sm uppercase tracking-wider text-muted-foreground mb-2">{t(locale, "Output", "输出")}</p>
                <p className="font-semibold mb-2 flex items-center gap-2">
                  <FlaskConical className="w-4 h-4 text-primary" /> {t(locale, "De novo generation + controllable variants", "从头生成 + 可控变体")}
                </p>
                <p className="text-sm text-muted-foreground">{t(locale, "Supports multiple C-terminal edits and latent perturbation strategies.", "支持多种 C 端变换与潜空间扰动策略。")}</p>
              </div>

              <div className="bg-card rounded-lg border border-border p-5">
                <p className="text-sm uppercase tracking-wider text-muted-foreground mb-2">{t(locale, "Validation", "验证")}</p>
                <p className="font-semibold mb-2 flex items-center gap-2">
                  <ShieldCheck className="w-4 h-4 text-primary" /> {t(locale, "Metrics + physicochemical + variant evaluation", "指标 + 理化 + 变体评估")}
                </p>
                <p className="text-sm text-muted-foreground">{t(locale, "Standardized outputs for version-to-version comparison.", "统一输出评估结果，便于版本对比。")}</p>
              </div>
            </div>

            <div className="mt-8 flex flex-wrap gap-3">
              <a
                href="https://github.com/unumbrela/amp-research2"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 text-sm px-4 py-2 rounded-md bg-primary text-primary-foreground hover:opacity-90 transition-opacity"
              >
                <Github className="w-4 h-4" /> {t(locale, "GitHub Repository", "GitHub 仓库")} <ArrowRight className="w-4 h-4" />
              </a>
              <a
                href="#generation"
                className="inline-flex items-center gap-2 text-sm px-4 py-2 rounded-md border border-border hover:border-primary/40 hover:bg-card transition-colors"
              >
                <Rocket className="w-4 h-4" /> {t(locale, "View Generation", "查看生成能力")}
              </a>
            </div>
          </div>

          <div className="lg:col-span-4 space-y-4 lg:pt-6">
            <div className="annotation-card">
              <p className="font-medium text-foreground mb-1">{t(locale, "Data Volume", "数据规模")}</p>
              <p>
                <strong className="text-primary">28,536</strong> {t(locale, "sequences, AMP ratio ~64.2%.", "序列，AMP 占比约 64.2%。")}
              </p>
            </div>
            <div className="annotation-card">
              <p className="font-medium text-foreground mb-1">{t(locale, "Training Structure", "训练结构")}</p>
              <p>{t(locale, "Three stages: VAE pretraining -> RL tuning -> diffusion training.", "三阶段训练：VAE 预训练 → RL 微调 → 扩散训练。")}</p>
            </div>
            <div className="annotation-card">
              <p className="font-medium text-foreground mb-1">{t(locale, "Engineering Status", "工程状态")}</p>
              <p>{t(locale, "Scripted pipeline and standardized outputs are ready; current focus is expansion and deployment.", "已具备脚本化流程和结果输出，当前重点是实验扩展与部署化。")}</p>
            </div>
            <div className="annotation-card">
              <p className="font-medium text-foreground mb-1">{t(locale, "Narrative Flow", "叙事主线")}</p>
              <p>{t(locale, "Background -> Method gaps -> Our approach -> Real outputs and evaluation.", "背景问题 → 现有方法不足 → 我们的方法与优势 → 真实输出与评估。")}</p>
            </div>
          </div>
        </div>
      </div>
    </Section>
  );
}

function ArchitectureSection({ locale }: { locale: Locale }) {
  return (
    <Section id="architecture" className="bg-secondary/30">
      <div className="container">
        <div className="flex items-start mb-10">
          <SectionNumber num="03." />
          <div>
            <h2 className="text-3xl md:text-4xl font-bold mb-2">{t(locale, "Architecture Design", "架构设计")}</h2>
            <p className="text-muted-foreground text-lg mt-3">
              {t(locale, "A unified architecture designed to close current method gaps with reproducibility and extensibility.", "针对现有方案短板，构建可复现且可扩展的统一架构")}
            </p>
            <div className="w-16 h-0.5 bg-primary mt-3" />
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-16">
          <div className="lg:col-span-8">
            <div className="bg-card rounded-lg border border-border p-5 mb-6">
              <p className="text-sm text-muted-foreground">
                {t(
                  locale,
                  "Our route combines ESM representation, VAE encoding/decoding, and latent diffusion. PLM features improve sequence representation, while latent diffusion improves sample diversity and controllability with practical engineering reuse.",
                  "我们的方法是 ESM 表征 + VAE 编解码 + 潜空间扩散：先用 PLM 提升序列表征质量，再在可控潜空间中学习多样化采样，目标是在“生成质量、可控性、工程复用”三者之间取得更稳健平衡。"
                )}
              </p>
            </div>
            <div className="rounded-lg overflow-hidden border border-border bg-card">
              <img src={MODELS_IMG} alt="AMP model architecture" className="w-full" />
              <p className="text-xs text-muted-foreground px-4 py-2 italic">
                {t(locale, "Figure: PLM representation + VAE + latent diffusion balances controllability and generation quality.", "图：项目采用 PLM 表征 + VAE + 潜空间扩散，兼顾可控性与生成质量。")}
              </p>
            </div>

            <div className="mt-6 bg-card rounded-lg border border-border p-5">
              <h3 className="text-lg font-semibold mb-3">{t(locale, "Core Modules", "核心模块与职责")}</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                <div className="rounded-md border border-border p-4">
                  <p className="font-semibold mb-1">PLM Extractor</p>
                  <p className="text-muted-foreground">{t(locale, "Supports ESM-2 / Ankh / ProtT5 backends.", "支持 ESM-2 / Ankh / ProtT5，多后端可切换。")}</p>
                </div>
                <div className="rounded-md border border-border p-4">
                  <p className="font-semibold mb-1">Hybrid AA Encoding</p>
                  <p className="text-muted-foreground">{t(locale, "BLOSUM62 + learnable embeddings for richer residue representation.", "BLOSUM62 + learnable embedding，增强残基表示。")}</p>
                </div>
                <div className="rounded-md border border-border p-4">
                  <p className="font-semibold mb-1">BiGRU + Non-AR Decoder</p>
                  <p className="text-muted-foreground">{t(locale, "Encodes latent variables and decodes in parallel to reduce autoregressive drift.", "编码潜变量并并行解码，减少自回归误差累积。")}</p>
                </div>
                <div className="rounded-md border border-border p-4">
                  <p className="font-semibold mb-1">Latent Diffusion</p>
                  <p className="text-muted-foreground">{t(locale, "Models in low-dimensional latent space for better sample quality and diversity.", "在低维潜空间建模，提升采样质量与多样性。")}</p>
                </div>
              </div>
            </div>
          </div>

          <div className="lg:col-span-4 space-y-4 lg:pt-2">
            <div className="annotation-card">
              <p className="font-medium text-foreground mb-1">{t(locale, "Latent Dimension", "潜变量维度")}</p>
              <p>
                <strong className="text-primary">64</strong> {t(locale, "dimensions for expressiveness and stability.", "维，兼顾表达能力与训练稳定性。")}
              </p>
            </div>
            <div className="annotation-card">
              <p className="font-medium text-foreground mb-1">{t(locale, "Max Sequence Length", "最大序列长度")}</p>
              <p>
                <strong className="text-primary">50 AA</strong>{t(locale, ", aligned with data cleaning rules.", "，与数据清洗规则保持一致。")}
              </p>
            </div>
            <div className="annotation-card">
              <p className="font-medium text-foreground mb-1">{t(locale, "Diffusion Steps", "扩散步数")}</p>
              <p>
                <strong className="text-primary">T = 50</strong>{t(locale, ", cosine schedule.", "，cosine schedule。")}
              </p>
            </div>
            <div className="annotation-card">
              <p className="font-medium text-foreground mb-1">{t(locale, "Sampling Controls", "生成策略")}</p>
              <p>{t(locale, "Supports top-p / top-k / temperature and CFG guidance.", "支持 top-p / top-k / 温度调节与 CFG 条件引导。")}</p>
            </div>
          </div>
        </div>
      </div>
    </Section>
  );
}

function DataTrainingSection({ locale }: { locale: Locale }) {
  return (
    <Section id="data">
      <div className="container">
        <div className="flex items-start mb-10">
          <SectionNumber num="04." />
          <div>
            <h2 className="text-3xl md:text-4xl font-bold mb-2">{t(locale, "Data & Training", "数据与训练")}</h2>
            <p className="text-muted-foreground text-lg mt-3">{t(locale, "Integrated data distribution, property coverage, and staged training.", "数据分布、属性覆盖与训练策略一体化设计")}</p>
            <div className="w-16 h-0.5 bg-primary mt-3" />
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-16">
          <div className="lg:col-span-7">
            <h3 className="text-xl font-semibold mb-4">{t(locale, "Dataset Split", "数据集划分")}</h3>
            <div className="bg-card rounded-lg border border-border p-5 mb-8">
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={dataSplit}>
                  <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.88 0.01 90)" />
                  <XAxis dataKey="split" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip
                    contentStyle={{
                      background: "oklch(0.99 0.003 90)",
                      border: "1px solid oklch(0.88 0.01 90)",
                      borderRadius: "8px",
                      fontSize: "13px",
                    }}
                  />
                  <Bar dataKey="value" name="Sequences" fill="oklch(0.65 0.2 25)" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <h3 className="text-xl font-semibold mb-4">{t(locale, "Property Coverage", "属性覆盖率")}</h3>
            <div className="bg-card rounded-lg border border-border p-5">
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart data={propertyCoverage}>
                  <PolarGrid stroke="oklch(0.88 0.01 90)" />
                  <PolarAngleAxis dataKey="dimension" tick={{ fontSize: 11 }} />
                  <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 10 }} />
                  <Radar dataKey="value" stroke="oklch(0.65 0.2 25)" fill="oklch(0.65 0.2 25 / 0.22)" strokeWidth={2} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="lg:col-span-5">
            <h3 className="text-xl font-semibold mb-4">{t(locale, "Training Flow (3 stages)", "训练流程（3 阶段）")}</h3>
            <div className="space-y-4">
              <div className="bg-card rounded-lg border border-border p-5">
                <p className="font-semibold mb-2">{t(locale, "Phase 1A · VAE Pretraining", "Phase 1A · VAE 预训练")}</p>
                <p className="text-sm text-muted-foreground mb-3">{t(locale, "Learn latent representations and baseline reconstruction.", "学习潜变量表示与基础重建能力。")}</p>
                <code className="text-xs bg-secondary px-2 py-1 rounded block">python training/train_vae.py --config configs/default.yaml</code>
              </div>
              <div className="bg-card rounded-lg border border-border p-5">
                <p className="font-semibold mb-2">{t(locale, "Phase 1B · RL Tuning", "Phase 1B · RL 微调")}</p>
                <p className="text-sm text-muted-foreground mb-3">{t(locale, "Improve generation quality via discriminator and policy-gradient signals.", "通过判别器与策略梯度提升生成质量。")}</p>
                <code className="text-xs bg-secondary px-2 py-1 rounded block">python training/train_vae_rl.py --config configs/default.yaml --vae-checkpoint checkpoints/vae_best.pt</code>
              </div>
              <div className="bg-card rounded-lg border border-border p-5">
                <p className="font-semibold mb-2">{t(locale, "Phase 2 · Diffusion Training", "Phase 2 · 扩散训练")}</p>
                <p className="text-sm text-muted-foreground mb-3">{t(locale, "Learn denoising in latent space to improve diversity.", "在潜空间学习去噪采样，增强多样性。")}</p>
                <code className="text-xs bg-secondary px-2 py-1 rounded block">python training/train_diffusion.py --config configs/default.yaml --vae-checkpoint checkpoints/vae_best_recon.pt</code>
              </div>
            </div>

            <div className="annotation-card mt-5">
              <p className="font-medium text-foreground mb-1">{t(locale, "Data Sources", "数据来源")}</p>
              <p>{t(locale, "DRAMP, UniProt, and local references merged into one AMP-oriented format.", "DRAMP、UniProt 与本地参考库合并，统一到 AMP 任务格式。")}</p>
            </div>
          </div>
        </div>
      </div>
    </Section>
  );
}

function GenerationSection({ locale }: { locale: Locale }) {
  return (
    <Section id="generation" className="bg-secondary/30">
      <div className="container">
        <div className="flex items-start mb-10">
          <SectionNumber num="05." />
          <div>
            <h2 className="text-3xl md:text-4xl font-bold mb-2">{t(locale, "Generation", "生成能力")}</h2>
            <p className="text-muted-foreground text-lg mt-3">{t(locale, "De novo generation + controllable variants + real output samples", "de novo 生成 + 可控变体 + 真实输出样例展示")}</p>
            <div className="w-16 h-0.5 bg-primary mt-3" />
          </div>
        </div>

        <Tabs defaultValue="uncond" className="mb-10">
          <TabsList className="mb-6 bg-secondary">
            <TabsTrigger value="uncond">{t(locale, "Unconditional", "无条件生成")}</TabsTrigger>
            <TabsTrigger value="variant">{t(locale, "Variants", "变体生成")}</TabsTrigger>
            <TabsTrigger value="interp">{t(locale, "Latent Interpolation", "潜空间插值")}</TabsTrigger>
          </TabsList>

          <TabsContent value="uncond">
            <div className="bg-card rounded-lg border border-border p-6">
              <p className="text-sm text-muted-foreground mb-4">{t(locale, "Sample directly from diffusion prior to produce de novo AMP candidates.", "从扩散先验直接采样，批量生成新 AMP 候选序列。")}</p>
              <pre className="text-xs bg-secondary rounded-md p-4 overflow-x-auto"><code>{`python generation/unconditional.py \\
  --config configs/default.yaml \\
  --checkpoint checkpoints/esm_diffvae_full.pt \\
  --n-samples 100 \\
  --top-p 0.9`}</code></pre>

              <div className="mt-5">
                <p className="font-semibold mb-2">{t(locale, "Real output samples (unconditional_generated.fasta)", "真实生成样例（unconditional_generated.fasta）")}</p>
                <p className="text-xs text-muted-foreground mb-3">{t(locale, "Sequences below are copied from `esm_diffvae/results/unconditional_generated.fasta`.", "以下序列直接来自 `esm_diffvae/results/unconditional_generated.fasta` 的实际输出。")}</p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {unconditionalSamples.map((sample) => (
                    <div key={sample.id} className="rounded-md border border-border bg-secondary/60 p-3">
                      <p className="text-xs text-muted-foreground mb-1 font-mono">
                        {sample.id} · len={sample.len}
                      </p>
                      <code className="text-xs break-all">{sample.sequence}</code>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="variant">
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              <div className="lg:col-span-7 bg-card rounded-lg border border-border p-6">
                <p className="text-sm text-muted-foreground mb-4">{t(locale, "Generate structure-aware variants from a parent sequence, including mixed-mode scheduling.", "基于母序列生成结构可控变体，支持混合模式调度。")}</p>
                <pre className="text-xs bg-secondary rounded-md p-4 overflow-x-auto"><code>{`python generation/variant.py \\
  --config configs/default.yaml \\
  --checkpoint checkpoints/esm_diffvae_full.pt \\
  --input-sequence "GIGKFLHSAKKFGKAFVGEIMNS" \\
  --mode mixed \\
  --n-variants 50`}</code></pre>

                <div className="mt-5 rounded-md border border-border bg-secondary/40 p-4">
                  <p className="text-sm font-medium mb-2">{t(locale, "Real variant samples (variants_generated.json)", "真实变体样例（variants_generated.json）")}</p>
                  <p className="text-xs text-muted-foreground mb-3 font-mono">parent: GIGKFLHSAKKFGKAFVGEIMNS</p>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="text-left border-b border-border">
                          <th className="py-1 pr-2">{t(locale, "mode", "模式")}</th>
                          <th className="py-1 pr-2">{t(locale, "identity", "一致性")}</th>
                          <th className="py-1 pr-2">{t(locale, "edit", "编辑距离")}</th>
                          <th className="py-1">{t(locale, "sequence", "序列")}</th>
                        </tr>
                      </thead>
                      <tbody>
                        {variantSamples.map((item) => (
                          <tr key={`${item.mode}-${item.sequence}`} className="border-b border-border/50 last:border-b-0">
                            <td className="py-1 pr-2 font-mono">{item.mode}</td>
                            <td className="py-1 pr-2">{item.identity.toFixed(4)}</td>
                            <td className="py-1 pr-2">{item.editDistance}</td>
                            <td className="py-1 font-mono break-all">{item.sequence}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
              <div className="lg:col-span-5 bg-card rounded-lg border border-border p-6">
                <p className="font-semibold mb-3">{t(locale, "Variant Mode Matrix", "变体模式矩阵")}</p>
                <div className="space-y-2">
                  {variantModes.map((v) => (
                    <div key={v.mode} className="flex items-start justify-between gap-3 border-b border-border pb-2 last:border-b-0">
                      <div>
                        <p className="text-sm font-medium font-mono">{v.mode}</p>
                        <p className="text-xs text-muted-foreground">{locale === "zh" ? v.desc.zh : v.desc.en}</p>
                      </div>
                      <span className="text-xs px-2 py-0.5 rounded bg-secondary text-secondary-foreground">{locale === "zh" ? v.focus.zh : v.focus.en}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="interp">
            <div className="bg-card rounded-lg border border-border p-6">
              <p className="text-sm text-muted-foreground mb-4">{t(locale, "Interpolate in latent space between two AMP sequences to observe smooth sequence-family transitions.", "在两条 AMP 序列之间做潜空间插值，观察平滑过渡的序列族。")}</p>
              <pre className="text-xs bg-secondary rounded-md p-4 overflow-x-auto"><code>{`python generation/interpolation.py \\
  --config configs/default.yaml \\
  --checkpoint checkpoints/esm_diffvae_full.pt \\
  --seq-a "GIGKFLHSAKKFGKAFVGEIMNS" \\
  --seq-b "ILPWKWPWWPWRR" \\
  --n-steps 10`}</code></pre>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </Section>
  );
}

function EvaluationSection({ locale }: { locale: Locale }) {
  return (
    <Section id="evaluation">
      <div className="container">
        <div className="flex items-start mb-10">
          <SectionNumber num="06." />
          <div>
            <h2 className="text-3xl md:text-4xl font-bold mb-2">{t(locale, "Evaluation & Validation", "评估结果与验证")}</h2>
            <p className="text-muted-foreground text-lg mt-3">{t(locale, "From aggregate metrics to physicochemical properties, building a comparable evaluation baseline.", "从统计指标到理化特性，建立可比较的输出基线")}</p>
            <div className="w-16 h-0.5 bg-primary mt-3" />
          </div>
        </div>

        <div className="rounded-lg overflow-hidden border border-border bg-card mb-8">
          <img src={EVAL_IMG} alt="evaluation pipeline" className="w-full" />
          <p className="text-xs text-muted-foreground px-4 py-2 italic">
            {t(locale, "Figure: evaluation pipeline covers sequence quality, functional tendency, and safety-related indicators.", "图：评估流程覆盖序列质量、功能倾向与安全相关指标。")}
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <div className="bg-card rounded-lg border border-border p-5">
            <p className="text-xs uppercase tracking-wider text-muted-foreground">{t(locale, "Uniqueness", "唯一性")}</p>
            <p className="text-2xl font-bold text-primary mt-1">1.00</p>
            <p className="text-xs text-muted-foreground mt-1">{t(locale, "500/500 unique", "500/500 无重复")}</p>
          </div>
          <div className="bg-card rounded-lg border border-border p-5">
            <p className="text-xs uppercase tracking-wider text-muted-foreground">{t(locale, "Novelty", "新颖性")}</p>
            <p className="text-2xl font-bold text-primary mt-1">1.00</p>
            <p className="text-xs text-muted-foreground mt-1">{t(locale, "Novel versus training set", "相对训练集新颖")}</p>
          </div>
          <div className="bg-card rounded-lg border border-border p-5">
            <p className="text-xs uppercase tracking-wider text-muted-foreground">{t(locale, "Mean Length", "平均长度")}</p>
            <p className="text-2xl font-bold text-primary mt-1">25.54</p>
            <p className="text-xs text-muted-foreground mt-1">{t(locale, "AA (unconditional generation)", "AA（无条件生成）")}</p>
          </div>
          <div className="bg-card rounded-lg border border-border p-5">
            <p className="text-xs uppercase tracking-wider text-muted-foreground">{t(locale, "Mean Diversity", "平均多样性")}</p>
            <p className="text-2xl font-bold text-primary mt-1">0.853</p>
            <p className="text-xs text-muted-foreground mt-1">{t(locale, "Normalized edit distance", "归一化编辑距离")}</p>
          </div>
        </div>

        <div className="bg-card rounded-lg border border-border p-6">
          <p className="font-semibold mb-3">{t(locale, "Evaluation Entry", "评估入口")}</p>
          <pre className="text-xs bg-secondary rounded-md p-4 overflow-x-auto mb-4"><code>{`python evaluation/run_evaluation.py \\
  --config configs/default.yaml \\
  --checkpoint checkpoints/esm_diffvae_full.pt`}</code></pre>
          <p className="text-sm text-muted-foreground">
            {t(locale, "Results are written to `esm_diffvae/results/evaluation/` as JSON, FASTA, and plots for cross-version comparisons.", "结果默认写入 `esm_diffvae/results/evaluation/`，包含 JSON、FASTA 与可视化图表，便于版本间对比。")}
          </p>
        </div>
      </div>
    </Section>
  );
}

function LandscapeSection({ locale }: { locale: Locale }) {
  return (
    <Section id="landscape" className="bg-secondary/30">
      <div className="container">
        <div className="flex items-start mb-10">
          <SectionNumber num="02." />
          <div>
            <h2 className="text-3xl md:text-4xl font-bold mb-2">{t(locale, "Existing Methods and Gaps", "现有方法与不足")}</h2>
            <p className="text-muted-foreground text-lg mt-3">{t(locale, "Use representative architectures and repositories to define our entry point.", "从代表性架构与仓库出发，明确我们的方法切入点")}</p>
            <div className="w-16 h-0.5 bg-primary mt-3" />
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-16">
          <div className="lg:col-span-7 prose-academic">
            <p>
              {t(
                locale,
                "AMP design already has multiple viable routes: cVAE, diffusion models, and LLMs. The central bottleneck is no longer model availability, but continuous reproducibility and iteration velocity in real projects.",
                "AMP 设计已经有 cVAE、扩散模型和 LLM 等多条技术路线。问题不在“有没有模型”，而在“能否持续复现并快速迭代”。"
              )}{" "}
              {t(
                locale,
                "The methods below are practical references and also reveal concrete engineering pain points.",
                "下列方法代表了当前可借鉴的主流方向，也暴露了工程复用中的真实痛点。"
              )}
            </p>

            <div className="mt-6 space-y-3">
              {methodLandscape.map((item) => (
                <div key={item.name} className="bg-card rounded-lg border border-border p-4">
                  <div className="flex flex-wrap items-start justify-between gap-3 mb-2">
                    <p className="font-semibold flex items-center gap-2">
                      <BookOpen className="w-4 h-4 text-primary" /> {item.name}
                    </p>
                    <span className="text-xs px-2 py-0.5 rounded bg-secondary text-secondary-foreground">{locale === "zh" ? item.architecture.zh : item.architecture.en}</span>
                  </div>
                  <p className="text-sm text-muted-foreground">{t(locale, "Strength:", "优势：")} {locale === "zh" ? item.strengths.zh : item.strengths.en}</p>
                  <p className="text-sm text-muted-foreground mt-1">{t(locale, "Limitation:", "不足：")} {locale === "zh" ? item.limitations.zh : item.limitations.en}</p>
                  {item.repo && (
                    <a
                      href={item.repo}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 text-xs text-primary hover:underline mt-2"
                    >
                      {t(locale, "GitHub Repository", "GitHub 仓库")} <ExternalLink className="w-3 h-3" />
                    </a>
                  )}
                </div>
              ))}
            </div>
          </div>

          <div className="lg:col-span-5 space-y-4 lg:pt-4">
            <div className="bg-card rounded-lg border border-border p-5">
              <p className="font-semibold mb-2 flex items-center gap-2">
                <Target className="w-4 h-4 text-primary" /> {t(locale, "Why Our Method Works", "我们的方法为什么合理")}
              </p>
              <p className="text-sm text-muted-foreground">
                {t(locale, "AMP Forge turns research feasibility into engineering executability, prioritizing reproducibility, comparable outputs, and fast iteration loops.", "AMP Forge 的定位是把研究可行性转成工程可执行性，优先解决训练链路割裂、输出不可比、实验难复盘等问题。")}
              </p>
            </div>

            <div className="bg-card rounded-lg border border-border p-5">
              <p className="font-semibold mb-3 flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-primary" /> {t(locale, "Our Advantages", "项目优势")}
              </p>
              <div className="space-y-2">
                {projectAdvantages.map((adv) => (
                  <p key={adv.en} className="text-sm text-muted-foreground">
                    - {locale === "zh" ? adv.zh : adv.en}
                  </p>
                ))}
              </div>
            </div>

            <div className="annotation-card">
              <p className="font-medium text-foreground mb-1">{t(locale, "Main Repository", "核心仓库")}</p>
              <p>
                <a
                  href="https://github.com/unumbrela/amp-research2"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 text-primary hover:underline"
                >
                  unumbrela/amp-research2 <ExternalLink className="w-3 h-3" />
                </a>
              </p>
            </div>
          </div>
        </div>
      </div>
    </Section>
  );
}

function RoadmapSection({ locale }: { locale: Locale }) {
  return (
    <Section id="roadmap">
      <div className="container">
        <div className="flex items-start mb-10">
          <SectionNumber num="07." />
          <div>
            <h2 className="text-3xl md:text-4xl font-bold mb-2">{t(locale, "Project Roadmap", "项目路线图")}</h2>
            <p className="text-muted-foreground text-lg mt-3">{t(locale, "Current progress and next expansion directions.", "当前进展与下一阶段扩展方向")}</p>
            <div className="w-16 h-0.5 bg-primary mt-3" />
          </div>
        </div>

        <div className="relative">
          <div className="absolute left-6 md:left-1/2 top-0 bottom-0 w-px bg-border md:-translate-x-px" />

          {projectMilestones.map((m, i) => {
            const statusClass =
              m.status === "done"
                ? "bg-primary/10 text-primary"
                : m.status === "in_progress"
                ? "bg-chart-2/15 text-chart-2"
                : "bg-secondary text-muted-foreground";
            const statusLabel =
              m.status === "done"
                ? t(locale, "Done", "已完成")
                : m.status === "in_progress"
                ? t(locale, "In Progress", "进行中")
                : t(locale, "Next", "下一步");

            return (
              <motion.div
                key={m.title.en}
                initial={{ opacity: 0, x: i % 2 === 0 ? -20 : 20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true, margin: "-50px" }}
                transition={{ duration: 0.45, delay: i * 0.05 }}
                className={`relative flex items-start mb-8 ${i % 2 === 0 ? "md:flex-row" : "md:flex-row-reverse"}`}
              >
                <div className={`ml-16 md:ml-0 md:w-[calc(50%-2rem)] ${i % 2 === 0 ? "md:pr-8 md:text-right" : "md:pl-8"}`}>
                  <div className={`inline-block ${i % 2 === 0 ? "md:ml-auto" : ""}`}>
                    <span className={`inline-block text-xs px-2 py-0.5 rounded-full mb-2 ${statusClass}`}>{statusLabel}</span>
                    <h4 className="font-semibold text-base">{locale === "zh" ? m.title.zh : m.title.en}</h4>
                    <p className="text-sm text-muted-foreground mt-1">{locale === "zh" ? m.desc.zh : m.desc.en}</p>
                  </div>
                </div>

                <div className="absolute left-4 md:left-1/2 md:-translate-x-1/2 w-4 h-4 rounded-full bg-card border-2 border-primary z-10 mt-1" />

                <div className={`hidden md:block md:w-[calc(50%-2rem)] ${i % 2 === 0 ? "md:pl-8" : "md:pr-8 md:text-right"}`}>
                  <span className="font-[family-name:var(--font-display)] text-2xl font-bold text-primary/20">{String(i + 1).padStart(2, "0")}</span>
                </div>

                <span className="absolute left-0 top-0 text-xs font-mono text-primary md:hidden">{String(i + 1).padStart(2, "0")}</span>
              </motion.div>
            );
          })}
        </div>
      </div>
    </Section>
  );
}

function ReferencesSection({ locale }: { locale: Locale }) {
  return (
    <Section id="references" className="bg-secondary/30">
      <div className="container">
        <div className="flex items-start mb-10">
          <SectionNumber num="08." />
          <div>
            <h2 className="text-3xl md:text-4xl font-bold mb-2">{t(locale, "References", "参考文献")}</h2>
            <p className="text-muted-foreground text-lg mt-3">{t(locale, "Representative works directly referenced by this project.", "项目路线直接参考的代表性工作")}</p>
            <div className="w-16 h-0.5 bg-primary mt-3" />
          </div>
        </div>

        <div className="max-w-4xl space-y-4">
          {references.map((ref) => (
            <div key={ref.id} className="flex gap-4 text-sm">
              <span className="text-primary font-mono font-medium shrink-0">[{ref.id}]</span>
              <div>
                <span className="text-foreground">{ref.text}</span>
                {ref.doi && (
                  <a
                    href={`https://doi.org/${ref.doi}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="ml-2 text-primary hover:underline inline-flex items-center gap-0.5"
                  >
                    DOI <ExternalLink className="w-3 h-3" />
                  </a>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </Section>
  );
}

function Footer({ locale }: { locale: Locale }) {
  return (
    <footer className="border-t border-border py-12 mt-8">
      <div className="container">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="text-center md:text-left">
            <p className="font-[family-name:var(--font-display)] text-lg font-semibold">AMP Forge</p>
            <p className="text-sm text-muted-foreground mt-1">{t(locale, "AMP project homepage · technical narrative and project introduction", "抗菌肽生成项目主页 · 项目介绍与技术综述")}</p>
          </div>
          <div className="text-xs text-muted-foreground flex items-center gap-2">
            <CheckCircle2 className="w-3.5 h-3.5" /> {t(locale, "Background-driven story with method comparison and real outputs", "从背景问题到真实结果的完整项目叙事")}
          </div>
        </div>
      </div>
    </footer>
  );
}

export default function Home() {
  const [locale, setLocale] = useState<Locale>("en");

  return (
    <div className="min-h-screen" lang={locale === "zh" ? "zh-CN" : "en"}>
      <NavBar locale={locale} setLocale={setLocale} />
      <HeroSection locale={locale} />
      <OverviewSection locale={locale} />
      <LandscapeSection locale={locale} />
      <ArchitectureSection locale={locale} />
      <DataTrainingSection locale={locale} />
      <GenerationSection locale={locale} />
      <EvaluationSection locale={locale} />
      <RoadmapSection locale={locale} />
      <ReferencesSection locale={locale} />
      <Footer locale={locale} />
    </div>
  );
}
