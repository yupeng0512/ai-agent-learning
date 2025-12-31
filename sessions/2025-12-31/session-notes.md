# 学习记录 - 2025-12-31

## 学习概览
- **日期**: 2025-12-31
- **主题**: RAG 深度原理（Embedding、FAISS、Rerank）
- **对应知识领域**: E. 记忆系统 / F. 主流框架实战

---

## 学习内容

### 1. 向量检索 vs 协同过滤

**讨论问题**：向量检索和推荐系统中的协同过滤有什么相同之处？

**关键结论**：
- **共同点**：都是在高维空间中找"最近邻"，使用余弦相似度等距离度量
- **本质区别**：
  - 向量检索：向量由神经网络学习，捕捉**语义**
  - 协同过滤：向量由用户行为构成，捕捉**偏好模式**
- **趋势**：现代推荐系统也用 Embedding（如 Item2Vec），两者正在融合

### 2. Embedding 原理深入

#### 2.1 核心思想
- **分布式假设**："一个词的含义由它的上下文决定"
- One-Hot → 稠密向量：解决维度爆炸、无法表达语义相似性问题

#### 2.2 Word2Vec 架构
- **CBOW**：上下文 → 预测中心词
- **Skip-gram**：中心词 → 预测上下文

#### 2.3 现代文本 Embedding 进化
```
Word2Vec（静态词向量）
    ↓
BERT（上下文相关词向量）
    ↓
Sentence-BERT / BGE（句子级向量，适合检索）
```

**BGE-M3 特点**：
- Multi-lingual：多语言
- Multi-granularity：多粒度
- Multi-functionality：多功能

### 3. FAISS 索引机制

#### 3.1 核心索引类型
| 索引类型 | 原理 | 适用场景 |
|---------|------|---------|
| Flat | 暴力搜索 | <10万，需精确结果 |
| IVF | 分桶加速 | 10万-100万 |
| PQ | 乘积量化压缩 | 100万-1000万 |
| HNSW | 分层图索引 | >1000万，高性能 |

#### 3.2 IVF 原理
- 所有向量通过聚类分到不同桶
- 查询时只在最近的 nprobe 个桶内搜索
- 权衡：nprobe↑ → 精度↑ 速度↓

#### 3.3 HNSW 原理
- 分层可导航小世界图
- 从顶层（稀疏）到底层（稠密）逐层搜索
- 优势：查询快、精度高

### 4. Rerank 算法细节

#### 4.1 为什么需要 Rerank
- 向量检索是"双塔模型"，query 和 doc 独立编码
- 无法捕捉 query-doc 之间的细粒度交互
- Rerank 使用交叉编码器，能看到 token 级别的交互

#### 4.2 三种 Reranker 对比
| 类型 | 原理 | 速度 | 精度 |
|------|------|------|------|
| SimpleReranker | 关键词匹配 + 位置权重 | 很快 | 一般 |
| LLMReranker | LLM 评分相关性 | 慢 | 高 |
| HybridReranker | 两阶段：粗筛+精排 | 中 | 较高 |

#### 4.3 工业级方案
- 使用专门的 Cross-Encoder（如 BGE-Reranker）
- 比 LLM Rerank 快很多，精度也高

---

## 项目实战：智能文档问答系统

### 完成的功能
1. ✅ 文档加载与切分（PDF/MD/TXT/DOCX）
2. ✅ 向量数据库（FAISS + BGE-M3 Embedding）
3. ✅ RAG 问答引擎
4. ✅ Reranker 重排序（三种实现）
5. ✅ Gradio Web UI
6. ✅ 分批处理（解决 SiliconFlow API 64 限制）

### 解决的问题
- **文件上传无反应**：Gradio 6.x 返回文件路径字符串，修复类型处理
- **Embedding API 413 错误**：实现分批处理（每批 32 个）

---

## 深入学习：Embedding 模型训练与对比学习（已讲解，待消化）

### 1. 训练自己的 Embedding 模型

#### 为什么需要训练
- 通用模型对领域术语理解不足（如医疗、法律）
- 公司内部术语/产品名无法识别

#### 训练数据格式
```python
# (query, positive, negative) 三元组
{"query": "什么是心梗？", "positive": "心肌梗死是...", "negative": "感冒是..."}
```

#### 训练框架：Sentence-Transformers
```python
from sentence_transformers import SentenceTransformer, losses
model = SentenceTransformer('BAAI/bge-base-zh-v1.5')
train_loss = losses.MultipleNegativesRankingLoss(model)
model.fit(train_objectives=[(dataloader, train_loss)], epochs=3)
```

#### 训练策略
- 全量微调：效果好，需 10k+ 数据
- LoRA 微调：数据需求少（1k-5k），显存友好

### 2. 对比学习损失函数

#### 核心思想
- 让相似样本向量接近，不相似样本向量远离

#### 主流损失函数
| 损失函数 | 特点 |
|---------|------|
| Contrastive Loss | 处理成对样本，需显式构造正负样本 |
| Triplet Loss | (anchor, positive, negative) 三元组 |
| InfoNCE Loss | 对比学习标准损失，用于 SimCLR/CLIP/BGE |
| MultipleNegativesRankingLoss | batch 内负采样，Sentence-Transformers 默认 |

#### 温度参数 τ
- τ 小（0.01）：分布尖锐，关注最相似样本
- τ 大（0.5）：分布平滑，训练更稳定
- 常用值：0.05 ~ 0.1

---

## 待回答的理解检验题（明日继续）
1. 为什么 in-batch negatives 方法需要较大的 batch_size？
2. 温度参数 τ 设置为 0.01 和 0.5 时，模型学习行为有什么不同？
3. 如果你的领域数据只有 500 条 (query, doc) 对，你会选择什么训练策略？

---

## 关键收获
1. 理解了 Embedding 从 Word2Vec 到 BGE 的进化路线
2. 掌握了 FAISS 不同索引类型的原理和选型
3. 理解了 Rerank 的必要性和实现方案
4. 完成了文档问答系统的核心功能
5. 了解了 Embedding 模型训练流程和对比学习损失函数（待消化）
