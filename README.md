# Awesome-Resources-for-DCAI-in-Tabular-Data

This repository focuses exclusively on data-centric AI methods for tabular data. For more general data-centric AI approaches, please refer to [SJTU-DMTai/awesome-ml-data-quality-papers](https://github.com/SJTU-DMTai/awesome-ml-data-quality-papers/blob/main/README.md). 

## Benchmark
### Latest Benchmarks for Tabular Data (Since 2024)

| Benchmark Name | Task Type     | Year | Source                                                                                  | TLDR |
|:--------------:|:-------------:|:----:|:---------------------------------------------------------------------------------------:|:-----|
| MMQA           | Reasoning     | 2025 | [Link](https://openreview.net/pdf?id=GGlpykXDCa)  [Repo](https://github.com/WuJian1995/MMQA/issues/2)| MMQA is a multi-table multi-hop question answering dataset with 3,312 tables across 138 domains, evaluating LLMs' capabilities in multi-table retrieval, Text-to-SQL, Table QA, and primary/foreign key selection.
| ToRR           | Reasoning     | 2025 | [Link](https://arxiv.org/pdf/2502.19412)  [Repo](https://github.com/IBM/unitxt/blob/main/prepare/benchmarks/torr.py)| ToRR is a benchmark assessing LLMs' table reasoning and robustness across 10 datasets with diverse table serializations and perturbations, revealing models' brittleness to format variations.
| MMTU           | Comprehensive | 2025 | [Link](https://arxiv.org/pdf/2506.05587)  [Repo](https://github.com/MMTU-Benchmark/MMTU)| MMTU is a massive multi-task table understanding and reasoning benchmark with over 30K questions across 25 real-world table tasks, designed to evaluate models' ability to understand, reason, and manipulate tables.
| RADAR          | Reasoning     | 2025 | [Link](https://kenqgu.com/assets/pdf/RADAR_ARXIV.pdf)  [Repo](https://huggingface.co/datasets/kenqgu/RADAR)| RADAR is a benchmark for evaluating language models' data-aware reasoning on imperfect tabular data with 5 common data artifact types like outlier value or inconsistent format, which ensures that direct calculation on the perturbed table will yield an incorrect answer, forcing the model to handle the artifacts to obtain the correct result.
| Spider2        | Text2SQL      | 2025 | [Link](https://arxiv.org/abs/2411.07763)  [Repo](https://github.com/xlang-ai/Spider2)|
| DataBench      | Reasoning     | 2024 | [Link](https://aclanthology.org/2024.lrec-main.1179.pdf)  [Repo](https://huggingface.co/datasets/cardiffnlp/databench)|
| TableBench     | Reasoning     | 2024 | [Link](https://arxiv.org/abs/2408.09174)  [Repo](https://github.com/TableBench/TableBench)|
| TQA-Bench      | Reasoning     | 2024 | [Link](https://arxiv.org/pdf/2411.19504)  [Repo](https://github.com/Relaxed-System-Lab/TQA-Bench)|

### Selected Classical Benchmarks for Tabular Data
| Benchmark Name | Task Type     | Year | Source                                                                                  | TLDR |
|:--------------:|:-------------:|:----:|:---------------------------------------------------------------------------------------:|:-----|
| WikiTableQuestions| Simple QA  | 2016 | [Link](https://arxiv.org/pdf/1508.00305) [Repo](https://github.com/ppasupat/WikiTableQuestions)|
| Seq2SQL        | Text2SQL      | 2017 | [Link](https://arxiv.org/pdf/1709.00103)  [Repo](https://github.com/salesforce/WikiSQL)|
| Spider         | Text2SQL      | 2018 | [Link](https://arxiv.org/pdf/1809.08887)  [Repo](https://yale-lily.github.io/spider)|
| TABFACT        | Fact-checking | 2020 | [Link](https://arxiv.org/pdf/1909.02164)  [Repo](https://github.com/wenhuchen/Table-Fact-Checking)|
| TAPAS          | Text2SQL      | 2020 | [Link](https://arxiv.org/pdf/2004.02349)  [Repo](https://github.com/google-research/tapas)|
| FinQA          | Reasoning     | 2021 | [Link](https://arxiv.org/pdf/2109.00122)  [Repo](https://github.com/czyssrs/FinQA)|
| FeTaQA         | Reasoning     | 2021 | [Link](https://arxiv.org/pdf/2104.00369)  [Repo](https://github.com/Yale-LILY/FeTaQA)|
| Archer         | Text2SQL      | 2022 | [Link](https://arxiv.org/pdf/2402.12554)  [Repo](https://sig4kg.github.io/archer-bench/dataset/database.zip)|
| BIRD           | Text2SQL      | 2023 | [Link](https://arxiv.org/pdf/2305.03111.pdf)  [Repo](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/bird)|

## Design Space for Table Reasoning Pipeline
\[Building\]

## Data-Centric AI in Tabular Data Management
We have collected recent influential papers on Data-Centric AI Technologies employed to enhance the performance of LLMs in tabular data-related tasks, with annotations on the relevant technology classes featured in these papers. Here, `FS` stands for Feature Selection, `FG` for Feature Generation and `DA` for data augmentation. The papers listed below are arranged in a roughly chronological order of publication.

| Venue     | Paper                                                        |                            Links                             |   Tags    | TLDR                                                         |
| :-------- | :----------------------------------------------------------- | :----------------------------------------------------------: | :-------: | :----------------------------------------------------------- |
| KDD'25    | Continuous Optimization for Feature Selection with Permutation-Invariant Embedding and Policy-Guided Search |   [paper](https://arxiv.org/pdf/2505.11601)    | `FS` | 
| SIGMOD'25    | Adda: Towards Efficient in-Database Feature Generation via LLM-based Agents | [paper](https://dl.acm.org/doi/10.1145/3725262)    | `FG` | 
| AAAI'25    | Dynamic and Adaptive Feature Generation with LLM | [paper](https://ojs.aaai.org/index.php/AAAI/article/view/33851)    | `FG` | 
| IJCAI'25    | Evolutionary Large Language Model for Automated Feature Transformation | [paper](https://arxiv.org/pdf/2406.03505)    | `FG` | 
| ICDM'25    | OpenFE++: Efficient Automated Feature Generation via Feature Interaction |   [paper](https://arxiv.org/pdf/2504.17356)    | `FS` | 
| NAACL'25    | ALTER: Augmentation for Large-Table-Based Reasoning |   [paper](https://aclanthology.org/2025.naacl-long.9/)    | `DA` | 
| Arxiv'2504    | Comprehend, Divide, and Conquer: Feature Subspace Exploration via Multi-Agent Hierarchical Reinforcement Learning |   [paper](https://arxiv.org/pdf/2504.17356)    | `FS` | 
| Arxiv'2507    | Reinforcement Learning-based Feature Generation Algorithm for Scientific Data | [paper](https://arxiv.org/abs/2507.03498)    | `FG` | 
| KDD'24    | Unsupervised Generative Feature Transformation via Graph Contrastive Pre-training and Multi-objective Fine-tuning |   [paper](https://arxiv.org/pdf/2405.16879)    | `FG` | 
| KDD'24    | Feature selection as deep sequential generative learning. |   [paper](https://arxiv.org/pdf/2403.03838)    | `FS` | 
| SIGMOD'24    | SAGA: A Scalable Framework for Optimizing Data Cleaning Pipelines for Machine Learning Applications |   [paper](https://dl.acm.org/doi/10.1145/3617338)    | `FS` | 
| CIKM'24    | Reinforcement feature transformation for polymer property performance prediction | [paper](https://dl.acm.org/doi/abs/10.1145/3627673.3680105)    | `FG` | 
| ICDM'24    | Feature interaction aware automated data representation transformation. |   [paper](https://arxiv.org/pdf/2309.17011)    | `FG` | 
| NIPS'23    | Reinforcement-enhanced autoregressive feature transformation: gradient-steered search in continuous space for postfix expressions |   [paper](https://arxiv.org/pdf/2010.08784)    | `FG` `FS` | 
| ICML'23    | OpenFE: Automated Feature Generation with Expert-level Performance |   [paper](https://arxiv.org/abs/2211.12507)    | `FG` | 
| ICDE'23    | Toward Efficient Automated Feature Engineering |   [paper](https://epubs.siam.org/doi/pdf/10.1137/1.9781611978520.3?download=true)    | `FG` `FS` | 
| ICDE'23    | PA-FEAT: Fast Feature Selection for Structured Data via Progress-Aware Multi-Task Deep Reinforcement Learning |   [paper](https://ieeexplore.ieee.org/abstract/document/10184534)    | `FS` | 
| AAAI'23    | Efficient Top-K Feature Selection Using Coordinate Descent Method | [paper](https://dl.acm.org/doi/10.1609/aaai.v37i9.26258)    | `FS` | 
| AAAI'23    | T2G-FORMER: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction | [paper](https://arxiv.org/pdf/2211.16887)    | `FG` | 
| AAAI'23    | Weight Predictor Network with Feature Selection for Small Sample Tabular Biomedical Data | [paper](https://dl.acm.org/doi/10.1609/aaai.v37i8.26090)    | `FS` | 
| KDD'23     | Treatment Effect Estimation with Adjustment Feature Selection |   [paper](https://dl.acm.org/doi/pdf/10.1145/3580305.3599531)    | `FS` | 
| KDD'23     | LATTE: A Framework for Learning Item-Features to Make a Domain-Expert for Effective Conversational Recommendation |   [paper](https://dl.acm.org/doi/10.1145/3580305.3599401)    | `FS` | 
| KDD'23     | Explicit Feature Interaction-aware Uplift Network for Online Marketing |   [paper](https://arxiv.org/pdf/2306.00315)    | `FG` | 
| KDD'23     | Cognitive Evolutionary Search to Select Feature Interactions for Click-Through Rate Prediction |   [paper](https://dl.acm.org/doi/10.1145/3580305.3599277)    | `FG` `FS` | 
| KDD'23     | Scenario-Adaptive Feature Interaction for Click-Through Rate Prediction |   [paper](https://dl.acm.org/doi/10.1145/3580305.3599936)    | `FG` `FS` | 
| SIGIR'23   | Single-shot feature selection for multi-task recommendations |   [paper](https://dl.acm.org/doi/abs/10.1145/3539618.3591767)    | `FS` |
| TKDE'23    | Automated feature selection: A reinforcement learning perspective |   [paper](https://ieeexplore.ieee.org/abstract/document/9547816)    | `FS` | 
| ICDM'23    | Self-optimizing Feature Generation via Categorical Hashing Representation and Hierarchical Reinforcement Crossing |   [paper](https://arxiv.org/pdf/2309.04612)    | `FG` | 
| ICDM'23    | Beyond discrete selection: Continuous embedding space optimization for generative feature selection |   [paper](https://arxiv.org/pdf/2302.13221)    | `FS` | 
| ICML'22    | Difer: differentiable automated feature engineering. |   [paper](https://arxiv.org/abs/2211.12507)    | `FG` `FS` | 
| KDD'22     | AutoFAS: Automatic Feature and Architecture Selection for Pre-Ranking System |   [paper](https://arxiv.org/pdf/2205.09394)    | `FS` | 
| KDD'22     | Group-wise reinforcement feature generation for optimal and explainable representation space reconstruction. |   [paper](https://arxiv.org/pdf/2205.14526)    | `FG` | 
| KDD'22     | AdaFS: Adaptive Feature Selection in Deep Recommender System |   [paper](https://dl.acm.org/doi/10.1145/3534678.3539204)    | `FS` | 
| WWW'22     | Autofield: Automating feature selection in deep recommender systems |   [paper](https://dl.acm.org/doi/abs/10.1145/3485447.3512071)    | `FS` | 
| EMNLP'22   | Realistic Data Augmentation Framework for Enhancing Tabular Reasoning |   [paper](https://aclanthology.org/2022.findings-emnlp.324.pdf)    | `DA` | 
| KDD'21     | Fives: Feature interaction via edge search for large-scale tabular data |   [paper](https://arxiv.org/pdf/2007.14573)    | `FG` | 
| CIKM'20    | Tolerant Markov Boundary Discovery for Feature Selection |   [paper](https://dl.acm.org/doi/10.1145/3340531.3415927)    | `FS` | 
| ICDM'20    | AutoFS: Automated Feature Selection via Diversity-aware Interactive Reinforcement Learning |   [paper](https://arxiv.org/pdf/2008.12001)    | `FS` | 
| ICDM'20    | Simplifying reinforced feature selection via restructured choice strategy of single agent. |   [paper](https://arxiv.org/pdf/2009.09230)    | `FS` | 
| KDD'19     | Automating Feature Subspace Exploration via Multi-Agent Reinforcement Learning |   [paper](https://dl.acm.org/doi/10.1145/3292500.3330868)    | `FS` | 

### Tutorial and Survey Paper
[A Survey on Table Mining with Large Language Models: Challenges, Advancements and Prospects](https://d197for5662m48.cloudfront.net/documents/publicationstatus/252177/preprint_pdf/3d9c9b7d57481675d0d6e486c8bb7985.pdf)
[Tabular Data-centric AI: Challenges, Techniques and Future Perspectives](https://dl.acm.org/doi/pdf/10.1145/3627673.3679102)
[A Survey on Data-Centric AI: Tabular Learning from Reinforcement Learning and Generative AI Perspective](https://arxiv.org/pdf/2502.08828)
[A Survey of Table Reasoning with Large Language Models](https://arxiv.org/abs/2402.08259)
[Table Meets LLM: Can Large Language Models Understand Structured Table Data? A Benchmark and Empirical Study](https://arxiv.org/pdf/2305.13062)
