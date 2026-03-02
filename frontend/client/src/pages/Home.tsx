/**
 * AMP Research Report - Academic Journal Style
 * Design: Swiss International Style + Academic Publishing
 * Color: Warm white (#FAFAF5) + Charcoal (#1A1A2E) + Coral (#E8634A)
 * Typography: Playfair Display (headings) + Source Sans 3 (body) + Fira Code (code)
 */

import { useState, useEffect, useRef } from "react";
import { motion, useInView } from "framer-motion";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  generationModels,
  architectureComparison,
  evaluationTools,
  githubRepos,
  timeline,
  implementationSteps,
  citationTrends,
  evaluationDimensions,
} from "@/lib/data";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
  LineChart, Line, AreaChart, Area,
} from "recharts";
import {
  ExternalLink, Github, BookOpen, FlaskConical, Cpu, ArrowRight, ChevronDown,
  Star, Database, Beaker, Code2, FileText, Layers, Zap, Target, Shield,
} from "lucide-react";

const HERO_IMG = "https://d2xsxph8kpxj0f.cloudfront.net/310519663390962009/nwNByvtJSze5gMGiNTNhXC/hero-amp-structure-gJSYNxhRQsz5bcnvnQNtBP.webp";
const MODELS_IMG = "https://d2xsxph8kpxj0f.cloudfront.net/310519663390962009/nwNByvtJSze5gMGiNTNhXC/generation-models-imLzwiPRbyJQnYUeagQ5pB.webp";
const EVAL_IMG = "https://d2xsxph8kpxj0f.cloudfront.net/310519663390962009/nwNByvtJSze5gMGiNTNhXC/evaluation-pipeline-CkDvrw87vYQELZsHzvSFE7.webp";
const BG_IMG = "https://d2xsxph8kpxj0f.cloudfront.net/310519663390962009/nwNByvtJSze5gMGiNTNhXC/abstract-peptide-bg-7vmSS9d64cv7cVh5spxu7y.webp";

// Section wrapper with animation
function Section({ children, id, className = "" }: { children: React.ReactNode; id: string; className?: string }) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-80px" });
  return (
    <motion.section
      ref={ref}
      id={id}
      initial={{ opacity: 0, y: 30 }}
      animate={isInView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.6, ease: "easeOut" }}
      className={`py-16 md:py-24 ${className}`}
    >
      {children}
    </motion.section>
  );
}

// Section number badge
function SectionNumber({ num }: { num: string }) {
  return (
    <span className="section-number select-none mr-4 inline-block">{num}</span>
  );
}

// Navigation
function NavBar() {
  const [scrolled, setScrolled] = useState(false);
  const [activeSection, setActiveSection] = useState("hero");

  useEffect(() => {
    const onScroll = () => {
      setScrolled(window.scrollY > 60);
      const sections = ["hero", "abstract", "models", "evaluation", "repos", "timeline", "implementation", "references"];
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
    { id: "abstract", label: "摘要" },
    { id: "models", label: "生成模型" },
    { id: "evaluation", label: "评估方法" },
    { id: "repos", label: "开源项目" },
    { id: "timeline", label: "发展脉络" },
    { id: "implementation", label: "实现指南" },
  ];

  return (
    <nav className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${scrolled ? "bg-background/95 backdrop-blur-sm shadow-sm border-b border-border" : ""}`}>
      <div className="container flex items-center justify-between h-14">
        <a href="#hero" className="font-[family-name:var(--font-display)] text-lg font-semibold tracking-tight text-foreground hover:text-primary transition-colors">
          AMP Research
        </a>
        <div className="hidden md:flex items-center gap-1">
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
      </div>
    </nav>
  );
}

// Hero Section
function HeroSection() {
  return (
    <section id="hero" className="relative min-h-[85vh] flex items-end pb-16 pt-20">
      <div
        className="absolute inset-0 bg-cover bg-center opacity-[0.08]"
        style={{ backgroundImage: `url(${BG_IMG})` }}
      />
      <div className="container relative z-10">
        <div className="max-w-4xl">
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="text-sm uppercase tracking-[0.2em] text-primary font-medium mb-6"
          >
            深度调研报告 · 2026
          </motion.p>
          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4, duration: 0.8 }}
            className="text-4xl md:text-6xl lg:text-7xl font-bold leading-[1.1] mb-8 text-foreground"
          >
            抗菌肽生成技术
            <br />
            <span className="text-primary">深度研究与实现指南</span>
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
            className="text-lg md:text-xl text-muted-foreground leading-relaxed max-w-2xl mb-10"
          >
            系统梳理近年来基于深度学习的抗菌肽（AMP）生成、评估与优化技术前沿，
            涵盖VAE、GAN、扩散模型、LLM等主流架构，以及MIC预测、溶血性评估等关键工具。
          </motion.p>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.8 }}
            className="flex flex-wrap gap-3 text-sm text-muted-foreground"
          >
            <span className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-secondary">
              <BookOpen className="w-3.5 h-3.5" /> 20+ 论文调研
            </span>
            <span className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-secondary">
              <Github className="w-3.5 h-3.5" /> 7+ 开源项目
            </span>
            <span className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-secondary">
              <FlaskConical className="w-3.5 h-3.5" /> 10+ 评估工具
            </span>
            <span className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-secondary">
              <Cpu className="w-3.5 h-3.5" /> 6 步实现方案
            </span>
          </motion.div>
        </div>
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.2 }}
          className="absolute bottom-8 left-1/2 -translate-x-1/2"
        >
          <ChevronDown className="w-5 h-5 text-muted-foreground animate-bounce" />
        </motion.div>
      </div>
    </section>
  );
}

// Abstract Section
function AbstractSection() {
  return (
    <Section id="abstract">
      <div className="container">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-16">
          <div className="lg:col-span-8">
            <div className="flex items-start mb-8">
              <SectionNumber num="01." />
              <div>
                <h2 className="text-3xl md:text-4xl font-bold mb-2">摘要与背景</h2>
                <div className="w-16 h-0.5 bg-primary mt-3" />
              </div>
            </div>
            <div className="prose-academic">
              <p>
                随着抗生素耐药性问题日益严峻，抗菌肽（Antimicrobial Peptides, AMPs）作为传统抗生素的潜在替代品，受到了广泛关注。据估计，2019年全球有近495万例死亡与细菌抗微生物药物耐药性（AMR）有关，其中包括127万例可直接归因于细菌AMR的死亡。这一严峻形势推动了新型抗菌策略的研究。
              </p>
              <p>
                AMP通常由10至50个氨基酸组成，通过破坏细胞膜等多种机制发挥作用，具有不易产生耐药性的优点。然而，巨大的肽序列空间（长度不超过32个残基的肽约有4.5×10<sup>41</sup>种）和复杂的构效关系，为新型AMP的发现带来了巨大挑战。
              </p>
              <p>
                近年来，以深度学习为代表的人工智能技术在AMP的从头设计、筛选和优化方面取得了显著进展。从早期的循环神经网络（RNN）到变分自编码器（VAE）、生成对抗网络（GAN），再到最新的扩散模型（Diffusion Models）和大型语言模型（LLM），这些技术为在广阔的化学空间中高效探索和设计具有特定功能的全新AMP序列提供了强大的计算引擎。
              </p>
            </div>
            <div className="mt-8 rounded-lg overflow-hidden border border-border">
              <img src={HERO_IMG} alt="抗菌肽与细菌膜相互作用示意图" className="w-full" />
              <p className="text-xs text-muted-foreground px-4 py-2 bg-secondary/50 italic">
                图 1. 抗菌肽（AMP）与细菌细胞膜相互作用的示意图。AMP通过α-螺旋结构穿透脂质双分子层，形成孔道，破坏膜完整性。
              </p>
            </div>
          </div>
          <div className="lg:col-span-4 space-y-6 lg:pt-24">
            <div className="annotation-card">
              <p className="font-medium text-foreground mb-1">关键数字</p>
              <p>全球每年约<strong className="text-primary">495万</strong>例死亡与细菌AMR有关</p>
            </div>
            <div className="annotation-card">
              <p className="font-medium text-foreground mb-1">序列空间</p>
              <p>长度≤32的肽序列空间约<strong className="text-primary">4.5×10<sup>41</sup></strong>种</p>
            </div>
            <div className="annotation-card">
              <p className="font-medium text-foreground mb-1">已批准AMP</p>
              <p>目前仅约<strong className="text-primary">10种</strong>AMP获得监管机构批准</p>
            </div>
            <div className="annotation-card">
              <p className="font-medium text-foreground mb-1">核心挑战</p>
              <p>新颖性（Novelty）与有效性（Validity）之间的权衡是AMP生成的核心难题</p>
            </div>
          </div>
        </div>
      </div>
    </Section>
  );
}

// Models Section
function ModelsSection() {
  return (
    <Section id="models" className="bg-secondary/30">
      <div className="container">
        <div className="flex items-start mb-12">
          <SectionNumber num="02." />
          <div>
            <h2 className="text-3xl md:text-4xl font-bold mb-2">生成模型</h2>
            <p className="text-muted-foreground text-lg mt-3">从VAE到LLM：AMP生成技术的演进</p>
            <div className="w-16 h-0.5 bg-primary mt-3" />
          </div>
        </div>

        {/* Evolution image */}
        <div className="mb-12 rounded-lg overflow-hidden border border-border bg-card">
          <img src={MODELS_IMG} alt="深度学习模型演进" className="w-full" />
          <p className="text-xs text-muted-foreground px-4 py-2 italic">
            图 2. AMP生成深度学习模型的演进：从VAE的潜在表示学习，到GAN的对抗训练，再到扩散模型的迭代去噪，最终到LLM的序列建模。
          </p>
        </div>

        {/* Architecture comparison chart */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-16 mb-16">
          <div className="lg:col-span-8">
            <h3 className="text-xl font-semibold mb-6">模型架构性能对比</h3>
            <div className="bg-card rounded-lg border border-border p-6">
              <ResponsiveContainer width="100%" height={360}>
                <BarChart data={architectureComparison} layout="vertical" margin={{ left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.88 0.01 90)" />
                  <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 12 }} />
                  <YAxis dataKey="type" type="category" width={140} tick={{ fontSize: 12 }} />
                  <Tooltip
                    contentStyle={{
                      background: "oklch(0.99 0.003 90)",
                      border: "1px solid oklch(0.88 0.01 90)",
                      borderRadius: "8px",
                      fontSize: "13px",
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: "13px" }} />
                  <Bar dataKey="diversityScore" name="多样性" fill="oklch(0.65 0.2 25)" radius={[0, 4, 4, 0]} />
                  <Bar dataKey="qualityScore" name="质量" fill="oklch(0.55 0.15 250)" radius={[0, 4, 4, 0]} />
                  <Bar dataKey="speedScore" name="速度" fill="oklch(0.6 0.15 150)" radius={[0, 4, 4, 0]} />
                  <Bar dataKey="stabilityScore" name="稳定性" fill="oklch(0.7 0.12 60)" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="lg:col-span-4 space-y-4">
            {architectureComparison.map((arch) => (
              <div key={arch.type} className="annotation-card">
                <p className="font-medium text-foreground mb-1">{arch.type}</p>
                <p className="text-xs leading-relaxed">{arch.principle}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Citation trends */}
        <div className="mb-16">
          <h3 className="text-xl font-semibold mb-6">各类方法论文发表趋势（2020-2025）</h3>
          <div className="bg-card rounded-lg border border-border p-6">
            <ResponsiveContainer width="100%" height={320}>
              <AreaChart data={citationTrends}>
                <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.88 0.01 90)" />
                <XAxis dataKey="year" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip
                  contentStyle={{
                    background: "oklch(0.99 0.003 90)",
                    border: "1px solid oklch(0.88 0.01 90)",
                    borderRadius: "8px",
                    fontSize: "13px",
                  }}
                />
                <Legend wrapperStyle={{ fontSize: "13px" }} />
                <Area type="monotone" dataKey="llm" name="LLM" stroke="oklch(0.65 0.2 25)" fill="oklch(0.65 0.2 25 / 0.2)" strokeWidth={2} />
                <Area type="monotone" dataKey="diffusion" name="扩散模型" stroke="oklch(0.55 0.15 250)" fill="oklch(0.55 0.15 250 / 0.15)" strokeWidth={2} />
                <Area type="monotone" dataKey="gan" name="GAN" stroke="oklch(0.6 0.15 150)" fill="oklch(0.6 0.15 150 / 0.1)" strokeWidth={2} />
                <Area type="monotone" dataKey="vae" name="VAE" stroke="oklch(0.7 0.12 60)" fill="oklch(0.7 0.12 60 / 0.1)" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
            <p className="text-xs text-muted-foreground mt-3 italic">
              图 3. 2020-2025年各类AMP生成方法的论文发表数量趋势。扩散模型和LLM方法近年增长最为迅猛。
            </p>
          </div>
        </div>

        {/* Model cards */}
        <h3 className="text-xl font-semibold mb-6">代表性模型详解</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {generationModels.slice(0, 6).map((model) => (
            <div key={model.name} className="bg-card rounded-lg border border-border p-6 hover:shadow-md transition-shadow">
              <div className="flex items-start justify-between mb-3">
                <div>
                  <h4 className="text-lg font-semibold">{model.name}</h4>
                  <p className="text-xs text-muted-foreground">{model.journal} · {model.year}</p>
                </div>
                <span className="text-xs px-2 py-1 rounded-full bg-primary/10 text-primary font-medium">
                  {model.citations} 引用
                </span>
              </div>
              <p className="text-sm text-muted-foreground mb-3 leading-relaxed">{model.description}</p>
              <div className="flex flex-wrap gap-1.5 mb-3">
                {model.features.map((f) => (
                  <span key={f} className="text-xs px-2 py-0.5 rounded bg-secondary text-secondary-foreground">{f}</span>
                ))}
              </div>
              <div className="flex items-center gap-3 text-xs text-muted-foreground">
                <span className="font-mono bg-secondary px-2 py-0.5 rounded">{model.architecture}</span>
                {model.github && (
                  <a href={model.github} target="_blank" rel="noopener noreferrer" className="flex items-center gap-1 text-primary hover:underline">
                    <Github className="w-3 h-3" /> {model.stars} stars
                  </a>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* Architecture comparison table */}
        <div className="mt-12">
          <h3 className="text-xl font-semibold mb-6">架构对比总览</h3>
          <div className="overflow-x-auto rounded-lg border border-border bg-card">
            <table className="table-academic">
              <thead>
                <tr>
                  <th>架构类型</th>
                  <th>代表模型</th>
                  <th>优势</th>
                  <th>劣势</th>
                </tr>
              </thead>
              <tbody>
                {architectureComparison.map((arch) => (
                  <tr key={arch.type}>
                    <td className="font-medium whitespace-nowrap">{arch.type}</td>
                    <td className="text-sm text-muted-foreground">{arch.representatives}</td>
                    <td className="text-sm">{arch.pros}</td>
                    <td className="text-sm text-muted-foreground">{arch.cons}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </Section>
  );
}

// Evaluation Section
function EvaluationSection() {
  const radarData = evaluationDimensions.map((d) => ({
    subject: d.dimension,
    value: d.importance,
    fullMark: 100,
  }));

  return (
    <Section id="evaluation">
      <div className="container">
        <div className="flex items-start mb-12">
          <SectionNumber num="03." />
          <div>
            <h2 className="text-3xl md:text-4xl font-bold mb-2">评估方法与工具</h2>
            <p className="text-muted-foreground text-lg mt-3">从物理化学性质到MIC预测的多维度评估体系</p>
            <div className="w-16 h-0.5 bg-primary mt-3" />
          </div>
        </div>

        {/* Evaluation pipeline image */}
        <div className="mb-12 rounded-lg overflow-hidden border border-border bg-card">
          <img src={EVAL_IMG} alt="评估流程" className="w-full" />
          <p className="text-xs text-muted-foreground px-4 py-2 italic">
            图 4. AMP序列评估流程：从序列输入到物理化学性质计算、活性预测、安全性评估，最终得到综合评分。
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-16 mb-16">
          <div className="lg:col-span-8">
            <h3 className="text-xl font-semibold mb-6">评估维度重要性</h3>
            <div className="bg-card rounded-lg border border-border p-6">
              <ResponsiveContainer width="100%" height={360}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="oklch(0.88 0.01 90)" />
                  <PolarAngleAxis dataKey="subject" tick={{ fontSize: 12 }} />
                  <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 10 }} />
                  <Radar name="重要性" dataKey="value" stroke="oklch(0.65 0.2 25)" fill="oklch(0.65 0.2 25 / 0.25)" strokeWidth={2} />
                </RadarChart>
              </ResponsiveContainer>
              <p className="text-xs text-muted-foreground mt-2 italic">
                图 5. AMP评估各维度的相对重要性。抗菌活性（MIC）和溶血性是最关键的评估指标。
              </p>
            </div>
          </div>
          <div className="lg:col-span-4 space-y-4 lg:pt-8">
            {evaluationDimensions.map((d) => (
              <div key={d.dimension} className="annotation-card">
                <p className="font-medium text-foreground mb-1">{d.dimension}</p>
                <p className="text-xs">{d.description}</p>
                <div className="mt-2 h-1.5 bg-secondary rounded-full overflow-hidden">
                  <div className="h-full bg-primary rounded-full transition-all" style={{ width: `${d.importance}%` }} />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Evaluation tools by category */}
        <Tabs defaultValue="mic" className="mb-8">
          <TabsList className="mb-6 bg-secondary">
            <TabsTrigger value="mic">MIC预测</TabsTrigger>
            <TabsTrigger value="safety">安全性评估</TabsTrigger>
            <TabsTrigger value="spectrum">抗菌谱预测</TabsTrigger>
            <TabsTrigger value="properties">物化性质</TabsTrigger>
          </TabsList>

          <TabsContent value="mic">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {evaluationTools.filter((t) => t.category === "MIC预测").map((tool) => (
                <ToolCard key={tool.name} tool={tool} />
              ))}
            </div>
            <div className="mt-6 prose-academic">
              <p>
                最小抑菌浓度（MIC）是衡量AMP抗菌活性最重要的定量指标。LLAMP基于ESM-2蛋白质语言模型进行微调，能够针对特定菌种预测MIC值，是目前最先进的工具之一。ANIA则采用Inception-Attention网络架构，在多个基准测试中展现了优越的预测性能。MBC-Attention是较早将注意力机制引入MIC回归预测的工作。
              </p>
            </div>
          </TabsContent>

          <TabsContent value="safety">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {evaluationTools.filter((t) => t.category === "溶血性预测").map((tool) => (
                <ToolCard key={tool.name} tool={tool} />
              ))}
            </div>
            <div className="mt-6 prose-academic">
              <p>
                溶血性是AMP安全性评估中最关键的指标之一。HAPPENN作为经典的神经网络分类器，被广泛引用和使用。AmpLyze则更进一步，能够直接预测HC50的具体数值，为定量评估提供了更精确的工具。此外，PyAMPA/AMPSolve平台还整合了毒性和血清半衰期等多维度安全性评估。
              </p>
            </div>
          </TabsContent>

          <TabsContent value="spectrum">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {evaluationTools.filter((t) => t.category === "多属性预测" || t.category === "综合评估" || t.category === "活性预测").map((tool) => (
                <ToolCard key={tool.name} tool={tool} />
              ))}
            </div>
            <div className="mt-6 prose-academic">
              <p>
                抗菌谱预测旨在判断AMP对不同类型微生物（革兰氏阳性菌、革兰氏阴性菌、真菌、病毒等）的活性。Diff-AMP的多属性预测模块可同时预测20种不同的活性属性。CalcAMP专注于预测对革兰氏阳性和阴性菌的活性。PyAMPA/AMPSolve则提供了最全面的高通量评估平台。
              </p>
            </div>
          </TabsContent>

          <TabsContent value="properties">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {evaluationTools.filter((t) => t.category === "物理化学性质" || t.category === "数据库+预测").map((tool) => (
                <ToolCard key={tool.name} tool={tool} />
              ))}
            </div>
            <div className="mt-6 prose-academic">
              <p>
                物理化学性质是AMP评估的基础。modlAMP是目前最成熟的Python包，支持计算疏水性（Eisenberg量表）、疏水矩、净电荷、等电点等多种关键描述符。DBAASP作为最全面的AMP数据库，不仅提供了丰富的数据资源，还内置了在线预测工具和API接口，是AMP研究不可或缺的基础设施。
              </p>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </Section>
  );
}

function ToolCard({ tool }: { tool: typeof evaluationTools[0] }) {
  return (
    <div className="bg-card rounded-lg border border-border p-5 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between mb-2">
        <h4 className="font-semibold">{tool.name}</h4>
        <span className="text-xs text-muted-foreground">{tool.year}</span>
      </div>
      <p className="text-sm text-muted-foreground mb-3 leading-relaxed">{tool.description}</p>
      <div className="flex flex-wrap gap-1.5 mb-2">
        {tool.features.map((f) => (
          <span key={f} className="text-xs px-2 py-0.5 rounded bg-secondary text-secondary-foreground">{f}</span>
        ))}
      </div>
      {tool.github && (
        <a href={tool.github} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1 text-xs text-primary hover:underline mt-1">
          <Github className="w-3 h-3" /> 查看代码
        </a>
      )}
    </div>
  );
}

// Repos Section
function ReposSection() {
  return (
    <Section id="repos" className="bg-secondary/30">
      <div className="container">
        <div className="flex items-start mb-12">
          <SectionNumber num="04." />
          <div>
            <h2 className="text-3xl md:text-4xl font-bold mb-2">开源项目</h2>
            <p className="text-muted-foreground text-lg mt-3">可直接复用的GitHub代码仓库</p>
            <div className="w-16 h-0.5 bg-primary mt-3" />
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {githubRepos.map((repo) => (
            <a
              key={repo.name}
              href={repo.url}
              target="_blank"
              rel="noopener noreferrer"
              className="group bg-card rounded-lg border border-border p-5 hover:shadow-lg hover:border-primary/30 transition-all"
            >
              <div className="flex items-center gap-2 mb-3">
                <Github className="w-4 h-4 text-muted-foreground group-hover:text-primary transition-colors" />
                <span className="text-sm font-mono text-muted-foreground">{repo.fullName}</span>
              </div>
              <h4 className="font-semibold mb-2 group-hover:text-primary transition-colors">{repo.name}</h4>
              <p className="text-sm text-muted-foreground mb-3 leading-relaxed">{repo.description}</p>
              <div className="flex flex-wrap gap-1.5 mb-3">
                {repo.features.map((f) => (
                  <span key={f} className="text-xs px-2 py-0.5 rounded bg-secondary text-secondary-foreground">{f}</span>
                ))}
              </div>
              <div className="flex items-center gap-4 text-xs text-muted-foreground">
                <span className="flex items-center gap-1">
                  <Star className="w-3 h-3" /> {repo.stars}
                </span>
                <span className="flex items-center gap-1">
                  <Code2 className="w-3 h-3" /> {repo.language}
                </span>
              </div>
            </a>
          ))}
        </div>
      </div>
    </Section>
  );
}

// Timeline Section
function TimelineSection() {
  const typeColors: Record<string, string> = {
    model: "bg-primary text-primary-foreground",
    tool: "bg-chart-2 text-white",
    database: "bg-chart-3 text-white",
  };
  const typeLabels: Record<string, string> = {
    model: "模型",
    tool: "工具",
    database: "数据库",
  };

  return (
    <Section id="timeline">
      <div className="container">
        <div className="flex items-start mb-12">
          <SectionNumber num="05." />
          <div>
            <h2 className="text-3xl md:text-4xl font-bold mb-2">发展脉络</h2>
            <p className="text-muted-foreground text-lg mt-3">AMP计算设计领域的关键里程碑</p>
            <div className="w-16 h-0.5 bg-primary mt-3" />
          </div>
        </div>

        <div className="relative">
          {/* Timeline line */}
          <div className="absolute left-6 md:left-1/2 top-0 bottom-0 w-px bg-border md:-translate-x-px" />

          {timeline.map((event, i) => (
            <motion.div
              key={`${event.year}-${event.title}`}
              initial={{ opacity: 0, x: i % 2 === 0 ? -30 : 30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.5, delay: i * 0.05 }}
              className={`relative flex items-start mb-8 ${i % 2 === 0 ? "md:flex-row" : "md:flex-row-reverse"}`}
            >
              {/* Content */}
              <div className={`ml-16 md:ml-0 md:w-[calc(50%-2rem)] ${i % 2 === 0 ? "md:pr-8 md:text-right" : "md:pl-8"}`}>
                <div className={`inline-block ${i % 2 === 0 ? "md:ml-auto" : ""}`}>
                  <span className={`inline-block text-xs px-2 py-0.5 rounded-full mb-2 ${typeColors[event.type]}`}>
                    {typeLabels[event.type]}
                  </span>
                  <h4 className="font-semibold text-base">{event.title}</h4>
                  <p className="text-sm text-muted-foreground mt-1">{event.description}</p>
                </div>
              </div>

              {/* Dot */}
              <div className="absolute left-4 md:left-1/2 md:-translate-x-1/2 w-4 h-4 rounded-full bg-card border-2 border-primary z-10 mt-1" />

              {/* Year label */}
              <div className={`hidden md:block md:w-[calc(50%-2rem)] ${i % 2 === 0 ? "md:pl-8" : "md:pr-8 md:text-right"}`}>
                <span className="font-[family-name:var(--font-display)] text-2xl font-bold text-primary/20">{event.year}</span>
              </div>

              {/* Mobile year */}
              <span className="absolute left-0 top-0 text-xs font-mono text-primary md:hidden">{event.year}</span>
            </motion.div>
          ))}
        </div>
      </div>
    </Section>
  );
}

// Implementation Section
function ImplementationSection() {
  const [activeStep, setActiveStep] = useState(0);

  return (
    <Section id="implementation" className="bg-secondary/30">
      <div className="container">
        <div className="flex items-start mb-12">
          <SectionNumber num="06." />
          <div>
            <h2 className="text-3xl md:text-4xl font-bold mb-2">实现指南</h2>
            <p className="text-muted-foreground text-lg mt-3">从零开始构建AMP生成与评估系统的分步指南</p>
            <div className="w-16 h-0.5 bg-primary mt-3" />
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-12">
          {/* Step navigation */}
          <div className="lg:col-span-4">
            <div className="lg:sticky lg:top-20 space-y-2">
              {implementationSteps.map((step, i) => (
                <button
                  key={step.step}
                  onClick={() => setActiveStep(i)}
                  className={`w-full text-left p-4 rounded-lg border transition-all ${
                    activeStep === i
                      ? "border-primary bg-card shadow-sm"
                      : "border-transparent hover:bg-card/50"
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <span className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-semibold ${
                      activeStep === i ? "bg-primary text-primary-foreground" : "bg-secondary text-muted-foreground"
                    }`}>
                      {step.step}
                    </span>
                    <span className={`text-sm font-medium ${activeStep === i ? "text-foreground" : "text-muted-foreground"}`}>
                      {step.title}
                    </span>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Step content */}
          <div className="lg:col-span-8">
            <motion.div
              key={activeStep}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              className="bg-card rounded-lg border border-border p-8"
            >
              <div className="flex items-center gap-3 mb-4">
                <span className="w-10 h-10 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold text-lg">
                  {implementationSteps[activeStep].step}
                </span>
                <h3 className="text-xl font-semibold">{implementationSteps[activeStep].title}</h3>
              </div>

              <p className="text-muted-foreground mb-6 leading-relaxed">{implementationSteps[activeStep].description}</p>

              <div className="mb-6">
                <h4 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground mb-3">推荐工具</h4>
                <div className="flex flex-wrap gap-2">
                  {implementationSteps[activeStep].tools.map((tool) => (
                    <span key={tool} className="text-sm px-3 py-1 rounded-full bg-secondary text-secondary-foreground font-mono">
                      {tool}
                    </span>
                  ))}
                </div>
              </div>

              <div className="bg-secondary/50 rounded-lg p-5">
                <h4 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground mb-3">详细说明</h4>
                <p className="text-sm text-foreground leading-relaxed">{implementationSteps[activeStep].details}</p>
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    </Section>
  );
}

// References Section
function ReferencesSection() {
  const references = [
    { id: 1, text: "Wang, J., et al. (2025). Discovery of antimicrobial peptides with notable antibacterial potency by an LLM-based foundation model.", journal: "Science Advances, 11(10)", doi: "10.1126/sciadv.ads8932" },
    { id: 2, text: "Szymczak, P., et al. (2023). Discovering highly potent antimicrobial peptides with deep generative model HydrAMP.", journal: "Nature Communications, 14(1), 1394", doi: "10.1038/s41467-023-36994-z" },
    { id: 3, text: "Wang, R., et al. (2024). Diff-AMP: tailored designed antimicrobial peptide framework with all-in-one generation, identification, prediction and optimization.", journal: "Briefings in Bioinformatics, 25(2)", doi: "10.1093/bib/bbae078" },
    { id: 4, text: "Jin, S., et al. (2025). AMPGen: an evolutionary information-reserved and diffusion-driven generative model for de novo design of antimicrobial peptides.", journal: "Communications Biology, 8(1), 582", doi: "10.1038/s42003-025-08282-7" },
    { id: 5, text: "Li, Y., et al. (2025). BroadAMP-GPT: AI-Driven generation of broad-spectrum antimicrobial peptides for combating multidrug-resistant ESKAPE pathogens.", journal: "Gut Microbes, 17(1)", doi: "10.1080/19490976.2025.2523811" },
    { id: 6, text: "Bae, D., et al. (2025). LLAMP: AI-Guided Discovery and Optimization of Antimicrobial Peptides Through Species-Aware Language Model.", journal: "bioRxiv", doi: "" },
    { id: 7, text: "Timmons, P. B., & Hewage, C. M. (2020). HAPPENN is a novel tool for hemolytic activity prediction for therapeutic peptides.", journal: "Scientific Reports, 10(1), 10869", doi: "10.1038/s41598-020-67701-3" },
    { id: 8, text: "Ramos-Llorens, M., et al. (2024). PyAMPA: a high-throughput prediction and optimization tool for antimicrobial peptides.", journal: "mSystems, 9(4)", doi: "10.1128/msystems.01358-23" },
    { id: 9, text: "Mueller, A., et al. (2017). modlAMP: Python for antimicrobial peptides.", journal: "Bioinformatics, 33(17), 2753-2755", doi: "" },
    { id: 10, text: "Pirtskhalava, M., et al. (2021). DBAASP v3: database of antimicrobial/cytotoxic activity and structure of peptides.", journal: "Nucleic Acids Research, 49(D1), D288-D297", doi: "" },
  ];

  return (
    <Section id="references">
      <div className="container">
        <div className="flex items-start mb-12">
          <SectionNumber num="07." />
          <div>
            <h2 className="text-3xl md:text-4xl font-bold mb-2">参考文献</h2>
            <div className="w-16 h-0.5 bg-primary mt-3" />
          </div>
        </div>
        <div className="max-w-3xl space-y-4">
          {references.map((ref) => (
            <div key={ref.id} className="flex gap-4 text-sm">
              <span className="text-primary font-mono font-medium shrink-0">[{ref.id}]</span>
              <div>
                <span className="text-foreground">{ref.text}</span>
                <span className="text-muted-foreground italic ml-1">{ref.journal}</span>
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

// Footer
function Footer() {
  return (
    <footer className="border-t border-border py-12 mt-8">
      <div className="container">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="text-center md:text-left">
            <p className="font-[family-name:var(--font-display)] text-lg font-semibold">AMP Research Report</p>
            <p className="text-sm text-muted-foreground mt-1">抗菌肽生成技术深度研究与实现指南 · 2026</p>
          </div>
          <p className="text-xs text-muted-foreground">
            由 Manus AI 调研并生成 · 数据截至 2026年2月
          </p>
        </div>
      </div>
    </footer>
  );
}

// Main page
export default function Home() {
  return (
    <div className="min-h-screen">
      <NavBar />
      <HeroSection />
      <AbstractSection />
      <ModelsSection />
      <EvaluationSection />
      <ReposSection />
      <TimelineSection />
      <ImplementationSection />
      <ReferencesSection />
      <Footer />
    </div>
  );
}
