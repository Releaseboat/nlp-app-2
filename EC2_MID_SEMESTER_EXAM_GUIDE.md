# EC2 Mid-Semester Examination: Conversational AI
## Study Guide & Practice Question Paper

**Course**: Conversational AI (EC2)  
**Examination Type**: Mid-Semester  
**Total Marks**: 30  
**Duration**: Suggested 90 minutes  
**Date**: June 2026  

---

## Table of Contents

1. [Topic Breakdown & Learning Objectives](#topic-breakdown--learning-objectives)
2. [Key Concepts Summary](#key-concepts-summary)
3. [Practice Question Paper](#practice-question-paper)
4. [Answer Key & Marking Rubric](#answer-key--marking-rubric)
5. [Formulas & Reference Materials](#formulas--reference-materials)

---

## Topic Breakdown & Learning Objectives

### **Sessions 1 & 2: Foundations and Embeddings**

#### Learning Objectives:
- Understand foundation models and their role in conversational AI
- Master embedding techniques and vector representations
- Comprehend semantic search and retrieval mechanisms

#### Key Concepts:
1. **Foundation Models (LLMs)**
   - Architecture: Transformer-based (GPT, BERT, T5)
   - Pretraining objectives: Language modeling, masked language modeling
   - Scale: Parameters, compute, data requirements
   
2. **Word & Sentence Embeddings**
   - Word2Vec (CBOW, Skip-gram)
   - GloVe (Global Vectors)
   - FastText (subword embeddings)
   - Contextual embeddings (ELMo, BERT)
   - Sentence-level embeddings (Sentence-BERT, Universal Sentence Encoder)

3. **Vector Representations**
   - Embedding dimensions and trade-offs
   - Similarity metrics: Cosine, Euclidean, Manhattan
   - Dimensionality reduction: PCA, t-SNE, UMAP

4. **Semantic Understanding**
   - Token-level vs sentence-level semantics
   - Semantic drift and context windows
   - Polysemy and contextual meaning

---

### **Sessions 2 & 3: ANN Methods and Hybrid Search**

#### Learning Objectives:
- Implement Approximate Nearest Neighbor (ANN) search
- Combine dense and sparse retrieval methods
- Optimize search for conversational AI systems

#### Key Concepts:
1. **Approximate Nearest Neighbor (ANN) Search**
   - Brute force vs approximate methods
   - Algorithms: HNSW, IVF, LSH, Product Quantization
   - Trade-offs: Speed vs accuracy
   
2. **Vector Indexing**
   - Faiss (Facebook AI Similarity Search)
   - Milvus, Pinecone, Weaviate
   - Index structures and complexity analysis
   
3. **Sparse & Dense Retrieval**
   - BM25 (sparse, keyword-based)
   - Dense retrieval (embedding-based)
   - Hybrid search combining both
   
4. **Ranking & Re-ranking**
   - Multi-stage ranking pipelines
   - Cross-encoders vs bi-encoders
   - Relevance scoring

---

### **Session 4: Model Landscape and Cost Engineering**

#### Learning Objectives:
- Compare open-source and proprietary LLMs
- Calculate inference costs and optimize
- Design cost-effective conversational systems

#### Key Concepts:
1. **LLM Landscape**
   - Proprietary: GPT-4, Claude, PaLM
   - Open-source: Llama, Mistral, Falcon
   - Model sizes: 7B, 13B, 70B, 405B parameters
   
2. **Cost Engineering**
   - Token-based pricing models
   - Compute cost: FLOPs, GPU utilization
   - Latency optimization
   - Batch vs real-time inference
   
3. **Model Selection Criteria**
   - Task performance (benchmark scores)
   - Latency requirements
   - Cost constraints
   - Hardware availability
   
4. **Infrastructure Considerations**
   - On-premises vs cloud deployment
   - Edge deployment for latency
   - Caching and KV-cache optimization

---

### **Session 5: Function Calling, ReAct, and Prompting**

#### Learning Objectives:
- Design function calling schemas for agents
- Implement ReAct (Reasoning + Acting) patterns
- Master prompt engineering techniques

#### Key Concepts:
1. **Function Calling**
   - API schema definition (JSON Schema)
   - Function routing and dispatch
   - Parameter extraction from natural language
   - Error handling and retry mechanisms
   
2. **ReAct Framework**
   - Think → Act → Observe loop
   - Tool usage integration
   - Multi-step reasoning chains
   - Action history and context management
   
3. **Prompt Engineering**
   - Few-shot prompting (in-context learning)
   - Chain-of-Thought (CoT)
   - Self-consistency
   - Prompt templates and composition
   
4. **Agent Patterns**
   - Sequential reasoning
   - Parallel function calling
   - Tool selection (routing)
   - State management

---

### **Sessions 6 & 7: Instruction Tuning, RLHF, and Fine-Tuning**

#### Learning Objectives:
- Fine-tune models for specific tasks
- Implement RLHF (Reinforcement Learning from Human Feedback)
- Apply instruction tuning for conversational tasks

#### Key Concepts:
1. **Instruction Tuning**
   - Dataset format (instruction → response)
   - Multi-task fine-tuning
   - Task diversity and generalization
   - Evaluation metrics for instruction-following
   
2. **Fine-Tuning Approaches**
   - Full fine-tuning (expensive)
   - Parameter-Efficient Fine-Tuning (PEFT):
     - LoRA (Low-Rank Adaptation)
     - QLoRA (Quantized LoRA)
     - Prefix tuning, Adapter modules
   
3. **RLHF Pipeline**
   - Human preference annotations
   - Reward model training
   - Policy optimization (PPO)
   - Alignment metrics
   
4. **Training Stability & Convergence**
   - Learning rate scheduling
   - Gradient accumulation
   - Mixed precision training
   - Loss curves and early stopping

---

## Key Concepts Summary

### **Embedding Space Mathematics**

**Cosine Similarity**:
```
sim(u, v) = (u · v) / (||u|| × ||v||)
Range: [-1, 1]
```

**Euclidean Distance**:
```
d(u, v) = √(Σ(u_i - v_i)²)
```

**Manhattan Distance**:
```
d(u, v) = Σ|u_i - v_i|
```

---

### **ANN Complexity Analysis**

| Algorithm | Build Time | Query Time | Memory |
|-----------|-----------|-----------|--------|
| Brute Force | O(1) | O(n×d) | O(n×d) |
| HNSW | O(n log n) | O(log n) | O(n×d) |
| IVF | O(n log n) | O(k + n×d/c) | O(n×d) |
| LSH | O(n) | O(L) | O(n×L) |

*n=number of vectors, d=dimensions, k=number of IVF clusters, L=number of hash tables*

---

### **LLM Cost Calculation**

**Token Counting**:
```
Input Tokens = len(prompt.split()) × avg_tokens_per_word (~1.3)
Output Tokens = estimated_response_length × avg_tokens_per_word
```

**Cost Formula**:
```
Cost = (Input_Tokens × Input_Price/1M) + (Output_Tokens × Output_Price/1M)
```

**Example**: GPT-4 Turbo
- Input: $0.01/1K tokens
- Output: $0.03/1K tokens

---

### **Fine-Tuning Memory Requirements**

**Full Fine-Tuning**:
```
Memory = Model_Size + Optimizer_States + Gradients + Activations
       ≈ Model_Size × (1 + 2 + 1 + 1) for Adam optimizer
       ≈ 5 × Model_Size
```

**LoRA Fine-Tuning**:
```
Memory ≈ Model_Size + LoRA_Matrices
       = Model_Size + (2 × d_model × r × num_layers)
       where r (rank) << d_model
```

**Memory Reduction**: LoRA reduces by ~95% for typical configurations.

---

### **RLHF Reward Model Training**

**Preference Pair Loss**:
```
Loss = -log(sigmoid(r(x, y_w) - r(x, y_l)))
```

where:
- r(x, y) = reward model output
- y_w = preferred response
- y_l = non-preferred response

---

## Practice Question Paper

### **Section A: Conceptual Questions (10 marks)**
*Answer any 5 questions (2 marks each)*

---

#### **Q1: Embeddings and Semantic Search**

**What is the difference between sparse embeddings (BM25) and dense embeddings (BERT)?** 
- Explain with an example
- Compare advantages and disadvantages
- When would you use each approach?

**Marking Rubric (2 marks)**:
- Definition clarity: 0.5 marks
- Concrete example: 0.5 marks
- Comparison table/explanation: 0.75 marks
- Use case recommendations: 0.25 marks

---

#### **Q2: Foundation Models**

**Explain the concept of "in-context learning" in GPT-style models.**
- How does the context window enable this capability?
- What are the limitations?
- How does this differ from fine-tuning?

**Marking Rubric (2 marks)**:
- Mechanism explanation: 0.75 marks
- Context window role: 0.5 marks
- Limitations: 0.5 marks
- Fine-tuning comparison: 0.25 marks

---

#### **Q3: ANN Search Algorithms**

**Compare HNSW (Hierarchical Navigable Small World) and IVF (Inverted File).**
- Algorithm principles
- Speed vs accuracy trade-offs
- When to use each?

**Marking Rubric (2 marks)**:
- Algorithm explanation: 0.75 marks
- Trade-offs analysis: 0.75 marks
- Use case selection: 0.5 marks

---

#### **Q4: Function Calling**

**Design a function calling schema for a travel booking conversational AI.**
- Define 3-4 functions with parameters
- Explain parameter extraction process
- How would the model handle ambiguous user inputs?

**Marking Rubric (2 marks)**:
- Schema design: 0.75 marks
- Parameter extraction logic: 0.75 marks
- Ambiguity handling: 0.5 marks

---

#### **Q5: ReAct Framework**

**Explain the ReAct (Reasoning + Acting) loop with a specific example.**
- Walk through Think → Act → Observe cycle
- How does it improve over vanilla reasoning?
- What are potential failure modes?

**Marking Rubric (2 marks)**:
- Framework explanation: 0.5 marks
- Example walkthrough: 0.75 marks
- Improvement over baselines: 0.5 marks
- Failure mode analysis: 0.25 marks

---

#### **Q6: Fine-Tuning Techniques**

**Compare full fine-tuning vs LoRA (Low-Rank Adaptation).**
- Explain the mathematical concept of LoRA
- Memory and compute savings
- Trade-offs in model quality

**Marking Rubric (2 marks)**:
- Concept explanation: 0.75 marks
- Mathematical foundation: 0.5 marks
- Resource comparison: 0.5 marks
- Quality trade-offs: 0.25 marks

---

#### **Q7: RLHF Process**

**Describe the RLHF (Reinforcement Learning from Human Feedback) pipeline.**
- Steps involved
- How human preferences are incorporated
- Challenges in scaling RLHF

**Marking Rubric (2 marks)**:
- Pipeline steps: 0.75 marks
- Preference incorporation: 0.75 marks
- Scaling challenges: 0.5 marks

---

#### **Q8: Prompt Engineering**

**What is Chain-of-Thought (CoT) prompting and why is it effective?**
- Explain the technique
- Provide an example
- When does it fail?

**Marking Rubric (2 marks)**:
- Technique explanation: 0.5 marks
- Example quality: 0.75 marks
- Failure cases: 0.75 marks

---

#### **Q9: Model Selection**

**How would you choose between GPT-4, Llama 70B, and Mistral 7B for a customer support chatbot?**
- Evaluation criteria
- Trade-offs analysis
- Cost-benefit considerations

**Marking Rubric (2 marks)**:
- Selection criteria: 0.75 marks
- Trade-offs analysis: 0.75 marks
- Economic considerations: 0.5 marks

---

#### **Q10: Hybrid Retrieval**

**Explain a hybrid retrieval system combining BM25 and dense embeddings.**
- Architecture overview
- How results are combined (fusion methods)
- Performance improvements

**Marking Rubric (2 marks)**:
- Architecture clarity: 0.75 marks
- Fusion techniques: 0.75 marks
- Performance metrics: 0.5 marks

---

### **Section B: Scenario-Based Case Studies (12 marks)**
*Answer any 2 questions (6 marks each)*

---

#### **Case Study 1: E-commerce Chatbot Cost Optimization**

**Scenario**:
You're building a conversational AI chatbot for an e-commerce platform handling 1 million queries per day. The requirements are:
- Response latency: <2 seconds
- Cost budget: $5,000/month
- Accuracy: >85% on task completion
- 24/7 availability

**Current Setup**:
- Using GPT-4 API at $0.03/1K output tokens
- Average query: 50 input tokens
- Average response: 150 output tokens
- System uptime: 99.9%

**Questions**:

**Q1a)** Calculate the monthly cost of the current GPT-4 setup.

```
Solution:
Daily queries: 1,000,000
Monthly queries: 1,000,000 × 30 = 30,000,000

Input tokens/day: 1,000,000 × 50 = 50M tokens
Output tokens/day: 1,000,000 × 150 = 150M tokens

Monthly input: 50M × 30 = 1.5B tokens
Monthly output: 150M × 30 = 4.5B tokens

Cost calculation:
- Input cost: (1.5B / 1M) × $0.01 = $15,000
- Output cost: (4.5B / 1M) × $0.03 = $135,000
- Total: $150,000/month

This EXCEEDS the $5,000 budget by 30x!
```

**Marks: 2**

**Q1b)** Propose a cost optimization strategy using multiple models:
- Llama 70B (self-hosted): $0.001/1K tokens (all)
- GPT-4 for complex queries (10% of traffic): $0.03/1K output
- Local embeddings for retrieval

Calculate new monthly cost and service architecture.

```
Solution:

Optimized Architecture:
1. Routing layer classifies queries
   - Simple queries (90%): Route to Llama 70B
   - Complex queries (10%): Route to GPT-4

Simple Query Cost (90% = 900K queries/day):
- Input: 900K × 50 = 45M tokens/day
- Output: 900K × 150 = 135M tokens/day
- Cost/day: (45M + 135M) / 1M × $0.001 = $180
- Monthly: $180 × 30 = $5,400

Complex Query Cost (10% = 100K queries/day):
- Input: 100K × 50 = 5M tokens/day
- Output: 100K × 150 = 15M tokens/day
- Cost/day: 5M × $0.01/1M + 15M × $0.03/1M = $50 + $450 = $500
- Monthly: $500 × 30 = $15,000

Total: $5,400 + $15,000 = $20,400/month (still over budget)

Better optimization:
- Route 95% to Llama 70B: $9,234/month
- Route 5% to GPT-4: $7,500/month
- Total: $16,734 (still high - need more optimization)

Alternative: Use caching layer
- Cache 70% of responses (common queries)
- Only 30% need fresh generation
- Effective cost: $16,734 × 0.3 = $5,020/month ✓
```

**Marks: 2**

**Q1c)** Design a caching and retrieval strategy to stay within budget.
- Propose vector indexing approach (ANN algorithm choice)
- Calculate cache hit rate needed
- Memory requirements for 1M cached responses

```
Solution:

Caching Strategy:
1. Implement Redis cache layer
   - Cache successful responses (chat history)
   - TTL: 7 days for e-commerce queries
   
2. Vector indexing for semantic search:
   - Use HNSW (Hierarchical Navigable Small World)
   - Embedding model: MiniLM-L6-v2 (384 dim)
   - Index 1M responses
   - Query time: ~5ms
   
3. Memory calculation:
   - 1M responses × 384 dimensions × 4 bytes (float32)
   - = 1.5 GB for embeddings
   - + ~100MB for HNSW metadata
   - Total: ~2GB (reasonable for single node Redis)

4. Cache hit rate analysis:
   Let x = cache hit rate needed
   Cost equation:
   $5,020 × (1 - x) = $5,000
   (1 - x) = 0.9975
   x = 0.25% (practically impossible!)
   
   This suggests: Further optimization needed:
   - Batch queries for off-peak processing
   - Use cheaper model for all queries
   - Implement request deduplication
```

**Marks: 2**

---

#### **Case Study 2: Multi-language Conversational AI with RLHF**

**Scenario**:
You're fine-tuning a Llama 70B model for customer support in 5 languages (English, Spanish, French, German, Japanese). You have 10,000 preference-annotated examples per language.

**Current Constraints**:
- GPU: 8× A100 (80GB each)
- Training time budget: 72 hours
- Model: Llama 70B (70B parameters)
- Batch size: 16 sequences

**Questions**:

**Q2a)** Calculate memory requirements for full fine-tuning vs LoRA.

```
Solution:

Full Fine-Tuning Memory:
- Model weights: 70B × 2 bytes (fp16) = 140GB
- Optimizer states (Adam): 70B × 8 bytes = 560GB
- Gradients: 70B × 2 bytes = 140GB
- Activations (batch=16): ~200GB
- Total: ~1,040GB per GPU!

Available: 8 × 80GB = 640GB (NOT ENOUGH)

LoRA Fine-Tuning Memory:
- Model weights: 70B × 2 bytes = 140GB (frozen, shared)
- LoRA adapters: 2 × d_model × r × num_layers
  = 2 × 4,096 × 16 × 80 = ~41.7MB per layer
  ≈ 3.3GB for all layers
- Optimizer states for LoRA: 3.3GB × 8 = ~26.4GB
- Gradients: ~3.3GB
- Activations: ~100GB
- Total per GPU: ~140GB + 26.4GB + 3.3GB + 100GB = ~270GB

Using 8 GPUs with LoRA: ~270GB per GPU (requires careful sharding)
Effective memory per GPU: 80GB < 270GB (need distributed training)

Solution: Use QLoRA (Quantized LoRA)
- Model weights: 70B × 1 byte (int8) = 70GB
- LoRA + optimizer: ~30GB
- Total: ~100GB per GPU → Fits on 8× A100
```

**Marks: 2**

**Q2b)** Design a multilingual RLHF pipeline for this scenario.

```
Solution:

RLHF Pipeline Architecture:
1. Reward Model Training (Phase 1: 24 hours)
   - Use 5,000 preference pairs per language (75% of data)
   - Train single multilingual reward model
   - Model: Llama 13B (smaller, faster)
   - Cross-lingual knowledge sharing
   - Validation: 2,500 pairs per language (25%)
   - Loss: -log(sigmoid(r(x,y_w) - r(x,y_l)))

2. Policy Optimization (Phase 2: 36 hours)
   - Algorithm: PPO (Proximal Policy Optimization)
   - Reference model: Original Llama 70B (frozen)
   - Learnable model: LoRA-adapted Llama 70B
   - Learning rate: 5e-6 (PPO-specific tuning)
   - Epochs: 3
   - KL divergence penalty: β = 0.01

3. Evaluation (Phase 3: 12 hours)
   - Alignment metrics (automatic):
     * BLEU (semantic similarity)
     * BERTScore (contextual matching)
   - Human evaluation (100 samples per language)
   - Performance per language

Timeline:
- Hours 0-24: Reward model training
- Hours 24-60: PPO policy optimization
- Hours 60-72: Evaluation and checkpointing

Challenges:
- Language imbalance (e.g., Japanese has fewer data)
- Cultural differences in preferences
- Translation quality for cross-lingual training
```

**Marks: 2**

**Q2c)** Propose a solution for handling language imbalance in training data.

```
Solution:

Language Imbalance Handling:

1. Data Augmentation:
   - Back-translation: Translate en→fr→en
   - Pseudo-labeling: Use high-confidence model predictions
   - Synthetic generation: Generate diverse support scenarios
   
   For Japanese (assuming smallest dataset):
   - Original: 10,000 examples
   - Back-translation: +5,000 (50% augmentation)
   - Synthetic generation: +5,000
   - Total: 20,000 examples → balanced with other languages

2. Weighted Sampling Strategy:
   - Sample probability ∝ 1/sqrt(language_data_size)
   - English (10K): weight = 0.32
   - Japanese (10K after augmentation): weight = 0.32
   - Ensures equal representation per epoch

3. Multi-task Learning:
   - Task 1: Language-agnostic support
   - Task 2: Language-specific nuances
   - Weight loss: L_total = L_generic + α × L_language_specific
   - α = 0.5 (hyperparameter)

4. Cross-lingual Transfer:
   - Leverage multilingual embeddings
   - Shared token vocabulary (tokenizer: mT5)
   - Knowledge transfer from English to low-resource languages

Implementation in LoRA:
- Shared base model weights
- Language-specific LoRA adapters
  - LoRA_EN for English nuances
  - LoRA_JA for Japanese nuances
- Combined during inference: output = base + LoRA_language
```

**Marks: 2**

---

### **Section C: Numerical Problems (8 marks)**
*Answer any 2 questions (4 marks each)*

---

#### **Problem 1: Vector Retrieval and ANN Search**

**Given**:
- Database: 10 million product descriptions (e-commerce)
- Embedding dimension: 768 (BERT-base)
- Similarity metric: Cosine similarity
- Query response time SLA: <100ms
- Available memory: 256GB on search server

**Question**:

**P1a)** Calculate memory requirements and suggest an ANN algorithm.

```
Solution:

Memory calculation for brute force (baseline):
- 10M vectors × 768 dimensions × 4 bytes (float32)
- = 10 × 10^6 × 768 × 4 bytes
- = 30,720 × 10^6 bytes
- = ~30.7 GB (fits in memory, but slow!)

Query time for brute force:
- 10M vectors × 768 dims = 7.68 × 10^9 operations
- Modern CPU: ~10^9 ops/sec → ~7.68 seconds per query
- EXCEEDS 100ms SLA ✗

Recommended: HNSW (Hierarchical Navigable Small World)

HNSW Memory:
- Embeddings: 30.7 GB
- Graph structure (M=16, average 16 connections/node):
  = 10M × 16 × 8 bytes (index pointers) = 1.28 GB
- Additional metadata: ~5 GB
- Total: ~37 GB (well under 256GB available)

HNSW Query Performance:
- Search time: O(log N) ≈ O(log 10^7) ≈ 24 hops
- Per-hop distance: 768 × 4 = 3,072 operations
- Total: 24 × 3,072 ≈ 74K operations
- At 10^9 ops/sec: ~0.074ms
- Plus overhead: ~10-20ms for I/O and graph traversal
- Total: ~20-30ms ✓ (within 100ms SLA)

Recommendation: HNSW is optimal for this use case
```

**Marks: 2**

**P1b)** Design a two-stage retrieval system combining sparse (BM25) and dense (HNSW) search with score fusion.

```
Solution:

Two-Stage Retrieval Pipeline:

Stage 1: Candidate Generation
┌─────────────────┐
│ User Query      │
└────────┬────────┘
         │
    ┌────┴─────────────────────┐
    │                           │
┌───▼──────────┐    ┌──────────▼──┐
│ BM25 Search  │    │ HNSW Search  │
│ (Sparse)     │    │ (Dense)      │
└───┬──────────┘    └──────────┬──┘
    │ Top 1K        │ Top 100
    │               │
    └───┬───────────┘
        │
    ┌───▼────────┐
    │ Union      │
    │ ~1.1K      │
    └───┬────────┘
        │

Stage 2: Re-ranking & Fusion

BM25 Score Calculation:
score_BM25 = Σ(IDF_i × (f_i × (k1 + 1)) / (f_i + k1 × (1 - b + b × (doc_len/avg_len))))

Parameters:
- k1 = 1.5 (term saturation)
- b = 0.75 (length normalization)
- IDF = log((N - n_i + 0.5) / (n_i + 0.5))

Example:
- Query: "black running shoes"
- BM25 for doc1: 3.2 (good match)
- Dense similarity: 0.85 (cosine similarity)

Score Fusion (Reciprocal Rank Fusion):
score_final = α × (1 / (60 + rank_BM25)) + (1 - α) × similarity_dense

α = 0.4 (weight for BM25)

For doc1:
- BM25 rank: 50
- Dense rank: 3
- score_final = 0.4 × (1/110) + 0.6 × (1/63)
- = 0.00364 + 0.00952
- = 0.0132 (final score)

Alternative fusion (Normalized score combination):
score_normalized_BM25 = (score_BM25 - min_score) / (max_score - min_score)
score_normalized_dense = similarity_dense (already [0,1])
score_final = 0.4 × score_normalized_BM25 + 0.6 × score_normalized_dense
```

**Marks: 2**

---

#### **Problem 2: Fine-tuning Cost and Performance Trade-off**

**Given**:
- Base model: Mistral 7B
- Training dataset: 100,000 customer support conversations
- Hardware: 1× A100 GPU (80GB)
- Training time limit: 24 hours
- Evaluation metric: Exact Match (EM) on test set

**Baseline (Pre-trained Model)**:
- EM: 62%
- Inference latency: 50ms
- Cost: $0 (using free model)

**Question**:

**P2a)** Compare full fine-tuning vs LoRA for this task and recommend the best approach.

```
Solution:

Scenario 1: Full Fine-Tuning

Parameters:
- Model size: 7B parameters
- Batch size: 16 (max for 80GB GPU)
- Learning rate: 2e-5
- Epochs: 3
- Dataset: 100K samples

Memory calculation:
- Model (fp16): 7B × 2 = 14GB
- Optimizer (Adam, fp32): 7B × 8 = 56GB
- Gradients (fp16): 7B × 2 = 14GB
- Batch activations: ~10GB
- Overhead: ~5GB
- Total: ~99GB ✗ EXCEEDS 80GB GPU!

Solution: Use gradient checkpointing + mixed precision
- Model (fp16): 14GB
- Optimizer (fp32): 56GB → reduce to 28GB with 8-bit optimizer
- Gradients: ~7GB (checkpointing reduces)
- Batch: ~5GB
- Total: ~54GB ✓ Fits!

Training time:
- 100K samples, batch size 16 → 6,250 iterations per epoch
- 3 epochs → 18,750 iterations
- Time per iteration: ~2.5 seconds (including backprop)
- Total: 18,750 × 2.5 = 46,875 seconds ≈ 13 hours ✓

Expected improvement:
- EM: 62% → 78% (+16 points)
- Cost: $0 (self-hosted)
- Latency: 50ms → 55ms (+10%)

---

Scenario 2: LoRA Fine-Tuning

Parameters:
- LoRA rank: 8
- LoRA alpha: 16
- Target modules: query, value projections in all layers
- Batch size: 32 (can increase with reduced memory)

Memory calculation:
- Model (fp16, frozen): 14GB
- LoRA adapters:
  = 2 × hidden_dim × rank × num_layers
  = 2 × 4,096 × 8 × 32 = ~2GB
- Optimizer (LoRA only): 2GB × 8 = 16GB
- Gradients: ~0.5GB
- Batch activations: ~20GB (larger batch)
- Total: ~52.5GB ✓ Fits!

Training time:
- 100K samples, batch size 32 → 3,125 iterations per epoch
- 3 epochs → 9,375 iterations
- Time per iteration: ~1.5 seconds (less computation)
- Total: 9,375 × 1.5 = 14,062 seconds ≈ 3.9 hours ✓

Expected improvement:
- EM: 62% → 75% (+13 points, slightly less than full FT)
- Cost: $0 (self-hosted)
- Latency: 50ms → 51ms (+2%, negligible)

---

Scenario 3: QLoRA (Quantized LoRA)

Memory calculation:
- Model (int4 quantized): 7B × 1 byte = 7GB
- LoRA adapters: 2GB
- Optimizer: ~8GB
- Batch: ~15GB
- Total: ~32GB ✓ Can use batch size 64!

Training time:
- 100K samples, batch size 64 → 1,563 iterations per epoch
- 3 epochs → 4,688 iterations
- Time per iteration: ~1.2 seconds
- Total: ~1.5 hours ✓

Expected improvement:
- EM: 62% → 72% (+10 points, some quality loss from quantization)

---

RECOMMENDATION:

Best choice: LoRA Fine-Tuning

Justification:
1. Excellent balance: EM improvement 62% → 75% (+13 pts)
2. Fast training: ~4 hours within 24-hour budget
3. Low cost: Self-hosted, no API calls
4. Minimal latency impact: 50ms → 51ms
5. Leaves time for: Multiple experiments, hyperparameter tuning

Comparison Table:
┌──────────────┬────────────┬───────────┬──────────┬───────────┐
│ Method       │ EM Score   │ Train hrs │ Latency  │ Recom.    │
├──────────────┼────────────┼───────────┼──────────┼───────────┤
│ Full FT      │ 62% → 78%  │ 13 hrs    │ +10%     │ Fallback  │
│ LoRA         │ 62% → 75%  │ 3.9 hrs   │ +2%      │ ✓ Best    │
│ QLoRA        │ 62% → 72%  │ 1.5 hrs   │ +3%      │ For speed │
│ No tuning    │ 62%        │ 0 hrs     │ baseline │ Baseline  │
└──────────────┴────────────┴──────��────┴──────────┴───────────┘
```

**Marks: 2**

**P2b)** Calculate ROI (Return on Investment) if deploying fine-tuned model with increased accuracy.

```
Solution:

Economic Model:

Current (Pre-trained only):
- Accuracy: 62% EM
- Failed queries: 38% → require escalation to human agents
- Cost per query: API call ($0.001) + human review ($5)
- Daily volume: 10,000 queries
- Daily cost: 10,000 × 0.001 + 3,800 × 5 = $10 + $19,000 = $19,010/day

After LoRA Fine-Tuning (75% EM):
- Accuracy improvement: +13 points
- Failed queries: 25% → escalation
- Cost per query: API call ($0.001) + human review (if needed)
- Daily cost: 10,000 × 0.001 + 2,500 × 5 = $10 + $12,500 = $12,510/day

Cost Savings:
- Daily savings: $19,010 - $12,510 = $6,500/day
- Monthly: $6,500 × 30 = $195,000/month
- Annual: $195,000 × 12 = $2,340,000/year

Fine-tuning Investment:
- GPU time: 4 hours × $1.50/hour (A100 on-demand) = $6
- Data annotation: $0 (using existing logs)
- Engineering: ~20 hours × $150/hour = $3,000
- Total one-time: ~$3,006

ROI Calculation:
ROI = (Benefit - Cost) / Cost × 100%
    = ($195,000 - $3,006) / $3,006 × 100%
    = 6,386% ✓ Extremely positive!

Payback period:
    = Investment / Monthly savings
    = $3,006 / $195,000
    = 0.015 months ≈ 0.5 days (immediate payback!)

Additional benefits (not in ROI):
- Faster query resolution (improved UX)
- Reduced human workload
- Customer satisfaction improvement
- Data for continuous learning

Sensitivity Analysis:
- If improvement only 5 points (67% EM):
  Daily savings: 500 × 5 = $2,500
  Monthly: $75,000
  ROI: ($75,000 - $3,006) / $3,006 = 2,394% ✓ Still highly positive

Conclusion: LoRA fine-tuning is economically justified
```

**Marks: 2**

---

## Answer Key & Marking Rubric

### **Section A: Conceptual Questions**

All questions: 2 marks each

**Evaluation Criteria**:
1. **Conceptual accuracy** (0.75 marks): Correct understanding of concepts
2. **Clarity of explanation** (0.5 marks): Clear, concise writing
3. **Examples/Evidence** (0.5 marks): Relevant examples or supporting evidence
4. **Completeness** (0.25 marks): Addresses all sub-questions

---

### **Section B: Case Studies**

All questions: 6 marks each (2 marks per sub-question)

**Evaluation Criteria**:
1. **Problem understanding** (1 mark): Correctly identifies constraints and requirements
2. **Solution approach** (2 marks): Sound methodology and reasoning
3. **Calculations** (1.5 marks): Accurate computations
4. **Justification** (1 mark): Explains why this approach is optimal
5. **Completeness** (0.5 marks): Addresses all aspects of the scenario

---

### **Section C: Numerical Problems**

All questions: 4 marks each (2 marks per sub-question)

**Evaluation Criteria**:
1. **Formula application** (1 mark): Correct formulas used
2. **Computation accuracy** (0.75 marks): Correct calculations
3. **Units & precision** (0.25 marks): Appropriate units and significant figures
4. **Interpretation** (1 mark): Explains what results mean
5. **Optimization insight** (1 mark): Suggests practical improvements

---

## Formulas & Reference Materials

### **Embedding & Similarity Metrics**

**Cosine Similarity**:
```
cos(θ) = (A · B) / (||A|| × ||B||)
       = Σ(A_i × B_i) / (√Σ(A_i²) × √Σ(B_i²))
```

**Euclidean Distance**:
```
d(p, q) = √(Σ(p_i - q_i)²)
```

**Manhattan Distance** (L1):
```
d(p, q) = Σ|p_i - q_i|
```

---

### **Embedding Dimension Trade-offs**

| Dimension | Use Case | Speed | Accuracy |
|-----------|----------|-------|----------|
| 64-128 | Fast retrieval | ✓✓✓ | ✗ |
| 256-384 | Balanced | ✓✓ | ✓✓ |
| 768-1024 | High accuracy | ✓ | ✓✓✓ |
| 2048+ | Very complex tasks | ✗ | ✓✓✓ |

---

### **ANN Algorithm Complexity**

| Algorithm | Build | Query | Space | Best For |
|-----------|-------|-------|-------|----------|
| Brute Force | O(n) | O(n·d) | O(n·d) | Baseline, <1M vectors |
| HNSW | O(n·log n) | O(log n) | O(n·d) | General purpose, fast |
| IVF | O(n·k) | O(k+n·d/k) | O(n·d) | Very large scale |
| LSH | O(n) | O(L) | O(n·L) | High-dimensional data |
| Product Quantization | O(n·k) | O(L·k) | O(n·log k) | Memory-constrained |

---

### **LLM Token Pricing Models**

**Prompt tokens**: Usually cheaper (read-only)  
**Completion tokens**: Usually 2-3× more expensive (generated)

**Popular Models (Apr 2024)**:
- GPT-4 Turbo: $0.01/$0.03 per 1K tokens
- GPT-3.5: $0.0005/$0.0015 per 1K tokens
- Claude 3: $0.003/$0.015 per 1K tokens
- Open-source (self-hosted): ~$0.0001-$0.001 per 1K tokens

---

### **Memory Requirements Formula**

**Full Fine-Tuning**:
```
Memory = Model_Size + 
         Optimizer_States + 
         Gradients + 
         Activations

For Adam optimizer:
Memory ≈ Model_Size × (1 + 2 + 1 + ~0.1) 
       ≈ 4.1 × Model_Size
```

**LoRA Fine-Tuning**:
```
Memory = Model_Size (frozen) + 
         LoRA_Matrices + 
         Optimizer_States (LoRA only)

LoRA_Matrices = 2 × hidden_dim × rank × num_layers
Memory_Reduction ≈ 99% for typical configs (rank=8, rank << hidden_dim)
```

**QLoRA Fine-Tuning**:
```
Memory = Model_Size_Quantized (int4) + 
         LoRA_Matrices + 
         Optimizer_States

Model_Size_Quantized = Original_Size × 0.25 (int4)
Memory_Reduction ≈ 99.5%
```

---

### **Training Time Calculations**

**Throughput (tokens/second)**:
```
Throughput = (Batch_Size × Seq_Length × GPU_Count) / Time_Per_Step
```

**Training Duration**:
```
Hours = Total_Tokens / (Throughput × 3600)
      = (Dataset_Size × Seq_Length × Epochs) / (Throughput × 3600)
```

**FLOPs (Floating Point Operations)**:
```
FLOPs_forward = 2 × Model_Params × Seq_Length × Batch_Size
FLOPs_backward = 2 × FLOPs_forward
FLOPs_total = FLOPs_forward + FLOPs_backward = 6 × Model_Params × Seq_Length × Batch_Size
```

---

### **RLHF Reward Model Loss**

**Bradley-Terry Loss**:
```
L = -log(σ(r(x, y_w) - r(x, y_l)))

where:
- r(x, y) = reward model score for context x and response y
- y_w = preferred response
- y_l = non-preferred response
- σ = sigmoid function
```

**PPO Loss**:
```
L_PPO = E_t[min(r_t(θ) × A_t, clip(r_t(θ), 1-ε, 1+ε) × A_t)]

where:
- r_t(θ) = policy/value ratio
- A_t = advantage
- ε = clip parameter (typically 0.2)
```

---

### **Evaluation Metrics for QA**

**Exact Match (EM)**:
```
EM = (# correct predictions) / (# total examples)
     (requires perfect match after normalization)
```

**F1 Score**:
```
precision = (common_tokens) / (predicted_tokens)
recall = (common_tokens) / (reference_tokens)
F1 = 2 × (precision × recall) / (precision + recall)
```

**BLEU Score**:
```
BLEU = BP × exp(Σ(w_n × log(p_n)))

where:
- BP = brevity penalty
- w_n = weight for n-gram (typically 0.25 for each n=1,2,3,4)
- p_n = precision for n-grams
```

**ROUGE-L (Longest Common Subsequence)**:
```
ROUGE-L = F_lcs = 2 × (R_lcs × P_lcs) / (R_lcs + P_lcs)

where:
- R_lcs = LCS / reference_length (recall)
- P_lcs = LCS / predicted_length (precision)
```

---

## Exam Tips & Study Strategy

### **Time Management**

**90-minute exam breakdown**:
- Section A (Conceptual): 25-30 minutes (answer 5 questions)
- Section B (Case Studies): 40-45 minutes (answer 2 questions)
- Section C (Numerical): 20-25 minutes (answer 2 problems)
- Buffer/review: 5-10 minutes

---

### **Key Areas to Revise**

1. **Embeddings**: Cosine similarity, dimensions, normalization
2. **ANN Algorithms**: HNSW vs IVF trade-offs
3. **Cost Calculation**: Token pricing, model selection
4. **Function Calling**: JSON schema design
5. **ReAct**: Think → Act → Observe loop
6. **Fine-tuning**: Full vs LoRA vs QLoRA memory/speed trade-offs
7. **RLHF**: Bradley-Terry loss, reward modeling
8. **Hybrid Retrieval**: BM25 + dense fusion, score normalization

---

### **Common Mistakes to Avoid**

1. ✗ Forgetting to account for optimizer states in memory
2. ✗ Using wrong similarity metric (e.g., Euclidean when cosine expected)
3. ✗ Confusing token pricing (input vs output rates)
4. ✗ Not considering GPU/batch size constraints
5. ✗ Mixing up MRR and Recall@K concepts
6. ✗ Forgetting to explain trade-offs (not just listing pros/cons)
7. ✗ Incorrect complexity analysis (log vs linear)

---

### **Study Resources**

**Essential Papers**:
- "Attention is All You Need" (Transformers)
- "BERT: Pre-training of Deep Bidirectional Transformers"
- "Language Models are Unsupervised Multitask Learners" (GPT-2)
- "RLHF from Human Preferences" (InstructGPT)
- "ReAct: Reasoning + Acting"
- "LoRA: Low-Rank Adaptation"

**Benchmark Datasets**:
- SQuAD v2.0 (QA)
- GLUE (NLU tasks)
- SuperGLUE (harder NLU)
- MMLU (knowledge)

---

## Additional Practice Scenarios

### **Scenario: Real-time Customer Support Chatbot**

**Context**:
- Platform: Retail company with 100K daily support queries
- Requirements: <1 second latency, 90%+ accuracy
- Infrastructure: Kubernetes cluster with GPU nodes

**Design Challenge**:
- Choose between GPT-4 (proprietary, fast), Llama 70B (open, slower), or Mistral 7B (balanced)
- Design retrieval system for FAQ/past tickets
- Implement fallback to human agents
- Calculate costs and ROI

**Expected Answer Components**:
1. Model selection with justification
2. Hybrid retrieval architecture (sparse + dense)
3. Fallback mechanism design
4. Cost analysis with ROI
5. Scalability considerations

---

## Conclusion

This examination tests both:
1. **Conceptual depth**: Understanding of foundational concepts
2. **Practical skills**: Ability to apply concepts to real scenarios
3. **Problem-solving**: Numerical analysis and optimization
4. **Communication**: Explaining complex trade-offs clearly

**Success requires**:
- Deep understanding of architecture trade-offs
- Practical calculation skills (memory, cost, latency)
- Ability to design systems for real constraints
- Clear reasoning and justification

---

**Good luck with your exam! 🎓**

*Last Updated: June 2026*  
*Version: 1.0*
