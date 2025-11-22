# MLSys-Reading-List-For-RAG

## RAG

Interesting Repo

1.https://github.com/NirDiamant/RAG_Techniques

2.https://github.com/athina-ai/rag-cookbooks

## RL+RAG

Grounding by Trying: LLMs with Reinforcement Learning-Enhanced Retrieval https://arxiv.org/html/2410.23214v2
COT+RAG

1.RAT:(COT+RAG) PKU--> https://arxiv.org/html/2403.05313v1

2.PlanxRAG: MSR---> https://arxiv.org/pdf/2410.20753

3.https://arxiv.org/pdf/2410.01782 --> OPEN-RAG: Enhanced Retrieval-Augmented Reasoning with Open-Source Large Language Models(runlin recommend iterative RAG system)

4.RAG-Star:https://arxiv.org/abs/2412.12881

## Parametric RAG

Parametric Retrieval Augmented Generation https://arxiv.org/pdf/2501.15915
RAG on KVcache

1.https://arxiv.org/html/2407.01178v1 Memory3: Language Modeling with Explicit Memory

2.https://arxiv.org/html/2412.15605v1 CAT

seems no killing app, can't beat full attention, and also not efficient as RAG

## Agentic RAG:

1.https://github.com/sunnynexus/Search-o1

## Scaling

1.Inference Scaling for Long-Context Retrieval Augmented Generation : Deepmind-->https://arxiv.org/abs/2410.04343 high score in iclr

2.Scaling Retrieval-Based Language Models with a Trillion-Token Datastore: UW rulin shao-->https://arxiv.org/abs/2407.12854

https://arxiv.org/pdf/2412.18069 online large scale RAG: Improving Factuality with Explicit Working Memory (Rulin) token prediction，retrieval，veriscore computation combine together. slow generation.

## RAG system paper

1.Cornell:Towards Understanding Systems Trade-offs in Retrieval-Augmented Generation Model Inference-->https://arxiv.org/abs/2412.11854

2.juncheng&Ravi: RAGserve-->https://arxiv.org/html/2412.10543v1

3.EdgeRAG: https://arxiv.org/pdf/2412.21023

4.RAGcache xin jin: https://arxiv.org/pdf/2404.12457

5.RAG+ spec infer zhihao jia:https://arxiv.org/abs/2401.14021 base: in context retrieval augment generation https://arxiv.org/abs/2302.00083 6.PipeRAG:https://arxiv.org/abs/2403.05676 base: improving language models by retrieving from trillions of tokens. In International conference on machine learning --> https://arxiv.org/abs/2112.04426 Deepmind

Teola: Towards End-to-End Optimization of LLM-based Applications https://arxiv.org/pdf/2407.00326
8.https://www.arxiv.org/abs/2502.20969 Tele RAG Look ahead RAG

ASPLOS 2025 https://arxiv.org/pdf/2412.15246 RAG acceleration
10.https://arxiv.org/pdf/2503.14649 RAG serving from google

## KV cache management for RAG

1.Cache Blend

2.Cache Craft https://skejriwal44.github.io/docs/CacheCraft_SIGMOD_2025.pdf

## Best Practice of RAG

Enhancing Retrieval-Augmented Generation: A Study of Best Practices: https://arxiv.org/pdf/2501.07391 2.Searching for Best Practices in Retrieval-Augmented Generation https://arxiv.org/pdf/2407.01219

## New Algorithm

1.Self RAG: https://arxiv.org/abs/2310.11511 400+ citation

2.RAG 2.0: https://contextual.ai/introducing-rag2/

## Grounding & Trustworthy

1.Trust align rag: 888 in iclr-->https://arxiv.org/pdf/2409.11242

2.ReDeEP: Detecting Hallucination in Retrieval-Augmented Generation via Mechanistic Interpretability https://arxiv.org/abs/2410.11414 7.33 in ICLR24

3.ENHANCING LARGE LANGUAGE MODELS’ SITUATED FAITHFULNESS TO EXTERNAL CONTEXTS:https://arxiv.org/abs/2410.14675 (when LLM meet wrong context)

## GraphRAG and Knowledge Graph Related

1.STructRAG: https://arxiv.org/abs/2410.08815

2.GraphCOT:https://arxiv.org/abs/2404.07103

3.Benchmark:MMQA: Evaluating LLMs with Multi-Table Multi-Hop Complex Questions https://openreview.net/forum?id=GGlpykXDCa 888 iclr24

4.SIRERAG: INDEXING SIMILAR AND RELATED INFORMATION FOR MULTIHOP REASONING:https://arxiv.org/abs/2412.06206

5.GraphRAG:https://github.com/microsoft/graphrag NanoGraphRAG:https://github.com/gusye1234/nano-graphrag

## RAG+Coder

1.CodeRAG-Bench: Can Retrieval Augment Code Generation? https://arxiv.org/abs/2406.14497 2.https://arxiv.org/abs/2406.07003 Graphcoder

## When Long context meets RAG

1.RankRAG by Nvidia: high score in nips24: https://arxiv.org/abs/2407.02485

2.Accelerating Inference of Retrieval-Augmented Generation via Sparse Context Selection:-->https://arxiv.org/abs/2405.16178

related:https://openreview.net/forum?id=TDy5Ih78b4-->Provence: efficient and robust context pruning for retrieval-augmented generation (Prune for RAG)

3.Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG--> DeepMind JiaWei: https://arxiv.org/abs/2410.05983

4.https://www.databricks.com/blog/author/quinn-leng Databricks Tech Report

5.https://arxiv.org/abs/2502.05252 GSM-Infinite: How Do Your LLMs Behave over Infinitely Increasing Context Length and Reasoning Complexity? from CMU beidi

## Retriever?

1.Sufficient Context: A New Lens on Retrieval Augmented Generation Systems--> https://arxiv.org/abs/2411.06037

2.LLM-Augmented Retrieval: Enhancing Retrieval Models Through Language Models and Doc-Level Embedding https://openreview.net/forum?id=7yncrX80CN--> use LLM to argument embedding model like colbertv2

3.NV embedd: https://arxiv.org/abs/2405.17428

## Chat and RAG

1.ChatQA:https://arxiv.org/pdf/2401.10225 NIPS high score NVIDIA Followed bt ChatQA2

## Multimodal

1.VisRAG:https://arxiv.org/abs/2410.10594

2.https://arxiv.org/pdf/2307.06940 Animate-A-Story: Storytelling with Retrieval-Augmented Video Generation

3.Image RAG: https://arxiv.org/abs/2502.09411

## Vector database related:

1.DiskANN : DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node -->https://papers.nips.cc/paper_files/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf Graph Based Method HNSW:https://arxiv.org/abs/1603.09320

2.IVF paper Revisiting the Inverted Indices for Billion-Scale Approximate Nearest Neighbors-->https://arxiv.org/abs/1802.02422 old ICF: original one-->Inverted files for text search engines

3.Dynamic Update IVF:Incremental IVF Index Maintenance for Streaming Vector Search https://arxiv.org/abs/2411.00970

4.SPANN:https://arxiv.org/abs/2111.08566

## IR retriever Alg:

1.Colbert: https://arxiv.org/abs/2004.12832 and Colbertv2:https://arxiv.org/abs/2112.01488 and PLAID:https://arxiv.org/abs/2205.09707

## GOOD Survey

1.Danqi Luke: https://arxiv.org/pdf/2403.03187

## Basic RAG Alg

1.in-context retrieval-augmented language models (retrieval with stride)

## Fault tolerance

Mitigating Stragglers in the Decentralized Training on Heterogeneous Clusters
CPR: Understanding and Improving Failure Tolerant Training for Deep Learning Recommendation with Partial Recovery
Efficient Replica Maintenance for Distributed Storage Systems
GEMINI: Fast Failure Recovery in Distributed Training with In-Memory Checkpoints SOSP 23 RICE
https://www.abstractsonline.com/pp8/#!/10856/presentation/9287
Elastic Averaging for Efficient Pipelined DNN Training
Understanding the Effects of Permanent Faults in GPU's Parallelism Management and Control Units | Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis
Straggler-Resistant Distributed Matrix Computation via Coding Theory: Removing a Bottleneck in Large-Scale Data Processing
[1901.05162] Coded Matrix Multiplication on a Group-Based Model
Parity models: erasure-coded resilience for prediction serving systems
[2002.02440] A locality-based approach for coded computation
Straggler Mitigation in Distributed Optimization Through Data Encoding
Learning Effective Straggler Mitigation from Experience and Modeling
Straggler-Resistant Distributed Matrix Computation via Coding Theory: Removing a Bottleneck in Large-Scale Data Processing
Bamboo: Making Preemptible Instances Resilient for Affordable Training of Large DNNs NSDI23 UCLA
Varuna: Scalable, Low-cost Training of Massive Deep Learning Models Eurosys21 MSR
Swift: Expedited Failure Recovery for Large-scale DNN Training PPoPP23 HKU
Oobleck: Resilient Distributed Training of Large Models Using Pipeline Templates SOSP23 Umich
LLM Program

## system

CUHK https://arxiv.org/pdf/2407.00326 Optimize LLM application
New Recommendation system

Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations (HSTU Meta new recsys linear and attention mixed)
MCTS powered LLM

Good Library to ACC: https://github.com/N8python/n8loom
Algorithmn

MCTSr: https://arxiv.org/html/2406.07394v1 Code: https://github.com/trotsky1997/MathBlackBox

Rest MCTS ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search THU NIPS24 code: https://github.com/THUDM/ReST-MCTS

SC-MCTS https://arxiv.org/abs/2410.01707 new lol THU

LLM for Video Understanding

repo : https://github.com/yunlong10/Awesome-LLMs-for-Video-Understanding
Diffusion Model System

## Model

1.DDPM https://arxiv.org/abs/2006.11239 cannot skip timesteps

2.DDIM https://arxiv.org/abs/2010.02502 can skip timesteps

3.Latent Diffusion Models https://github.com/CompVis/latent-diffusion in latent space rather than pixel space

4.[NSDI24] Approximate Caching for Efficiently Serving Diffusion Models https://arxiv.org/abs/2312.04429

5.LoRA for diffusion model parameter efficient finetune

6.ControlNet https://arxiv.org/abs/2302.05543

7.Video Diffusion Model https://arxiv.org/abs/2204.03458 3D unet More: https://github.com/showlab/Awesome-Video-Diffusion
