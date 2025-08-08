# Awesome-Resources-for-DCAI-in-Tabular-Data

This repository focuses exclusively on data-centric AI methods for tabular data. For more general data-centric AI approaches, please refer to [SJTU-DMTai/awesome-ml-data-quality-papers](https://github.com/SJTU-DMTai/awesome-ml-data-quality-papers/blob/main/README.md). 

## Benchmark
### Latest Benchmarks for Tabular Data (Since 2024)

| Benchmark Name | Task Type     | Year | Source                                                                                  | TLDR |
|:--------------:|:-------------:|:----:|:---------------------------------------------------------------------------------------:|:-----|
| MDBench        | Reasoning     | 2025 | [Link](https://github.com/jpeper/MDBench) | MDBench introduces a new multi-document reasoning benchmark synthetically generated through knowledge-guided prompting. |
| MMQA           | Reasoning     | 2025 | [Link](https://openreview.net/pdf?id=GGlpykXDCa)  [Repo](https://github.com/WuJian1995/MMQA/issues/2)| MMQA is a multi-table multi-hop question answering dataset with 3,312 tables across 138 domains, evaluating LLMs' capabilities in multi-table retrieval, Text-to-SQL, Table QA, and primary/foreign key selection.
| ToRR           | Reasoning     | 2025 | [Link](https://arxiv.org/pdf/2502.19412)  [Repo](https://github.com/IBM/unitxt/blob/main/prepare/benchmarks/torr.py)| ToRR is a benchmark assessing LLMs' table reasoning and robustness across 10 datasets with diverse table serializations and perturbations, revealing models' brittleness to format variations.
| MMTU           | Comprehensive | 2025 | [Link](https://arxiv.org/pdf/2506.05587)  [Repo](https://github.com/MMTU-Benchmark/MMTU)| MMTU is a massive multi-task table understanding and reasoning benchmark with over 30K questions across 25 real-world table tasks, designed to evaluate models' ability to understand, reason, and manipulate tables.
| RADAR          | Reasoning     | 2025 | [Link](https://kenqgu.com/assets/pdf/RADAR_ARXIV.pdf)  [Repo](https://huggingface.co/datasets/kenqgu/RADAR)| RADAR is a benchmark for evaluating language models' data-aware reasoning on imperfect tabular data with 5 common data artifact types like outlier value or inconsistent format, which ensures that direct calculation on the perturbed table will yield an incorrect answer, forcing the model to handle the artifacts to obtain the correct result.
| Spider2        | Text2SQL      | 2025 | [Link](https://arxiv.org/abs/2411.07763)  [Repo](https://github.com/xlang-ai/Spider2)|
| DataBench      | Reasoning     | 2024 | [Link](https://aclanthology.org/2024.lrec-main.1179.pdf)  [Repo](https://huggingface.co/datasets/cardiffnlp/databench)|
| TableBench     | Reasoning     | 2024 | [Link](https://arxiv.org/abs/2408.09174)  [Repo](https://github.com/TableBench/TableBench)|
| TQA-Bench      | Reasoning     | 2024 | [Link](https://arxiv.org/pdf/2411.19504)  [Repo](https://github.com/Relaxed-System-Lab/TQA-Bench)|
| SpreadsheetBench | Reasoning     | 2024 | [Link](https://arxiv.org/pdf/2406.14991)  [Repo](https://github.com/RUCKBReasoning/SpreadsheetBench/tree/main/data)|

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
| HiTab          | Reasoning     | 2022 | [Link](https://aclanthology.org/2022.acl-long.78.pdf) [Repo](https://github.com/microsoft/HiTab)
| Archer         | Text2SQL      | 2022 | [Link](https://arxiv.org/pdf/2402.12554)  [Repo](https://sig4kg.github.io/archer-bench/dataset/database.zip)|
| BIRD           | Text2SQL      | 2023 | [Link](https://arxiv.org/pdf/2305.03111.pdf)  [Repo](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/bird)|

## Design Space for Table Reasoning Pipeline
\[Building\]

## Data-Centric AI in Tabular Data Management
We have collected recent influential papers on Data-Centric AI Technologies employed to enhance the performance of LLMs in tabular data-related tasks, with annotations on the relevant technology classes featured in these papers. Here, `DA` stands for Data Augmentation, `FS` stands for Feature Selection, `FG` for Feature Generation and `DA` for data augmentation. The papers listed below are arranged in a roughly chronological order of publication.

| Venue       | Paper                                                        | Corresp. Author |                           Links                             |   Tags    | TLDR                                                         |
| :---------- | :----------------------------------------------------------- | :-------------: |:----------------------------------------------------------: | :-------: | :----------------------------------------------------------- |
| KDD'25      | Continuous Optimization for Feature Selection with Permutation-Invariant Embedding and Policy-Guided Search |   |  [paper](https://arxiv.org/pdf/2505.11601)    | `FS` | 
| SIGMOD'25   | Adda: Towards Efficient in-Database Feature Generation via LLM-based Agents | |  [paper](https://dl.acm.org/doi/10.1145/3725262)    | `FG` | 
| SIGMOD'25   | GEIL: A Graph-Enhanced Interpretable Data Cleaning Framework with Large Language Models | |  [paper](https://dl.acm.org/doi/10.1145/3698811)    | `` | 
| SIGMOD'25   | Auto-Test: Learning Semantic-Domain Constraints for Unsupervised Error Detection in Tables | |  [paper](https://arxiv.org/pdf/2504.10762) | ? |
| AAAI'25     | Dynamic and Adaptive Feature Generation with LLM | |  [paper](https://ojs.aaai.org/index.php/AAAI/article/view/33851)    | `FG` | 
| ICML'25     | Are Large Language Models Ready for Multi-Turn Tabular Data Analysis? | |  [paper](https://openreview.net/attachment?id=flKhxGTBj2&name=pdf)    | `` | 
| ICML'25     | Compositional Condition Question Answering in Tabular Understanding | |  [paper](https://openreview.net/attachment?id=aXU48nrA2v&name=pdf)    | `` | 
| ICML'25     | Quantifying Prediction Consistency Under Fine-tuning Multiplicity in Tabular LLMs | |  [paper](https://arxiv.org/pdf/2407.04173v2)    | `` |
| ICML'25     | TabICL: A Tabular Foundation Model for In-Context Learning on Large Data | |  [paper](https://arxiv.org/pdf/2502.05564)    | `` |
| IJCAI'25    | Evolutionary Large Language Model for Automated Feature Transformation | |  [paper](https://arxiv.org/pdf/2406.03505)    | `FG` | 
| ICDM'25     | OpenFE++: Efficient Automated Feature Generation via Feature Interaction |   |  [paper](https://arxiv.org/pdf/2504.17356)    | `FS` | 
| NAACL'25    | ALTER: Augmentation for Large-Table-Based Reasoning |   |  [paper](https://aclanthology.org/2025.naacl-long.9/)    | Table Sampling | The ALTER framework enhances large-table reasoning through a workflow that augments queries into sub-queries and tables with schema, semantic, and literal information, filters relevant rows/columns via embedding-based sampling and LLM-driven selection, generates and executes SQL to obtain sub-tables, and uses a joint reasoner to aggregate results from primary and sub-query workflows.
| NAACL'25    | TART: An Open-Source Tool-Augmented Framework for Explainable Table-based Reasoning |   |  [paper](https://arxiv.org/pdf/2409.11724)    | `DA` | TART contains three key components: a table formatter (clean and unify the format of the table) to ensure accurate data representation, a tool maker (python code) to develop specific computational tools, and an explanation generator to maintain explainability. 
| NAACL'25    | H-STAR: LLM-driven Hybrid SQL-Text Adaptive Reasoning on Tables | |  [paper](https://aclanthology.org/2025.naacl-long.445.pdf)    | ?  | 
| Arxiv'2507    | Reinforcement Learning-based Feature Generation Algorithm for Scientific Data | |  [paper](https://arxiv.org/abs/2507.03498)    | `FG` | 
| Arxiv'2506    | What to Keep and What to Drop: Adaptive Table Filtering Framework | |  [paper](https://arxiv.org/pdf/2506.23463)    | `` | 
| Arxiv'2501    | TableMaster: A Recipe to Advance Table Understanding with Language Models          | |  [paper](https://arxiv.org/pdf/2501.19378)    | ?    |
| Arxiv'2502    | Towards Question Answering over Large Semi-structured Tables          | |  [paper](https://arxiv.org/pdf/2502.13422)    | ?    |
| Arxiv'2505    | Weaver: Interweaving SQL and LLM for Table Reasoning | |  [paper](https://arxiv.org/pdf/2505.18961)    | ?    | |
| NIPS'24    | TableRAG: Million-Token Table Understanding with Language Models |   |  [paper](https://arxiv.org/pdf/2410.04739)    | ? | 
| ICLR'24    | OpenTab: Advancing Large Language Models as Open-domain Table Reasoners |   |  [paper](https://arxiv.org/pdf/2402.14361)    | ? | 
| ICLR'24    | CABINET: Content Relevance based Noise Reduction for Table Question Answering |   |  [paper](https://arxiv.org/pdf/2402.01155)    | ? | 
| ICLR'24    | Chain-of-Table: Evolving Tables in the Reasoning Chain for Table Understanding |   |  [paper](https://arxiv.org/pdf/2401.04398)    | ? | 
| ICLR'24    | ReMasker: Imputing Tabular Data with Masked Autoencoding |   |  [paper](https://openreview.net/pdf?id=KI9NqjLVDT)    | ? | 
| ICLR'24    | Making Pre-trained Language Models Great on Tabular Prediction |   |  [paper](https://openreview.net/pdf?id=KI9NqjLVDT)    | ? | 
| ICML'24    | Large Language Models Can Automatically Engineer Features for Few-Shot Tabular Learning |   |  [paper](https://openreview.net/pdf?id=anzIzGZuLi)    | ? |
| ICML'24 workshop   | Learning to Reduce: Towards Improving Performance of Large Language Models on Structured Data|   |  [paper](https://arxiv.org/pdf/2407.02750)    | ? | 
| KDD'24      | Unsupervised Generative Feature Transformation via Graph Contrastive Pre-training and Multi-objective Fine-tuning |   |  [paper](https://arxiv.org/pdf/2405.16879)    | `FG` | 
| KDD'24      | Feature selection as deep sequential generative learning. |   |  [paper](https://arxiv.org/pdf/2403.03838)    | `FS` | 
| KDD'24      | Can a Deep Learning Model be a Sure Bet for Tabular Prediction? | |  [paper](https://dl.acm.org/doi/10.1145/3637528.3671893) ||
| KDD'24      | From Supervised to Generative: A Novel Paradigm for Tabular Deep Learning with Large Language Models | |  [paper](https://arxiv.org/pdf/2310.07338) ||
| SIGMOD'24   | SAGA: A Scalable Framework for Optimizing Data Cleaning Pipelines for Machine Learning Applications |   |  [paper](https://dl.acm.org/doi/10.1145/3617338)    | `FS` | 
| VLDB'24     | ReAcTable: Enhancing ReAct for Table Question Answering |   |  [paper](https://arxiv.org/pdf/2310.00815)    | ? | 
| EMNLP'24    | NormTab: Improving Symbolic Reasoning in LLMs Through Tabular Data Normalization |   |  [paper](https://arxiv.org/pdf/2406.17961)    | ? | 
| EMNLP'24    | TAP4LLM: Table Provider on Sampling, Augmenting, and Packing Semi-structured Data for Large Language Model Reasoning |   |  [paper](https://arxiv.org/pdf/2312.09039)    | `Pipeline` | This paper constructed a pipeline to boost LLM's performance of table reasoning by introducing three components, Table Sampling Module, Table Augmentationd Module and Table Packing Module. In each module, the authors designed and compared several common methods under various usage scenarios, aiming to searching for best practices for leveraging LLMs for table reasoning tasks.  |
| EMNLP'24 (Demo)   | OpenT2T: An Open-Source Toolkit for Table-to-Text Generation |   |  [paper](https://aclanthology.org/2024.emnlp-demo.27.pdf)    | ? | 
| NAACL'24    | Rethinking Tabular Data Understanding with Large Language Models | |  [paper](https://arxiv.org/pdf/2312.16702) | ? |
| NAACL'24    | TabSQLify: Enhancing Reasoning Capabilities of LLMs Through Table Decomposition |   |  [paper](https://arxiv.org/pdf/2404.10150)    | ? | 
| NAACL'24    | e5: zero-shot hierarchical table analysis using augmented llms via explain, extract, execute, exhibit and extrapolate |   |  [paper](https://aclanthology.org/2024.naacl-long.68.pdf)    | ? |  First, the model is guided by designed prompt to understand the hierarchical structure of the table, including the multi-level headers and their implicit semantic relationships. Then, it generates code to pull out the data rows and columns relevant to the query, along with performing necessary operations like filtering or calculations. Next, an external tool runs this code to get accurate results, preventing the model from making up information. These results are then presented clearly. Finally, the model uses its reasoning ability to analyze these results and derive the final answer to the query. For large tables that exceed token limits, the pipeline first compresses them by identifying and keeping only the most relevant data, while adding back potentially useful information to ensure key details aren’t lost, before proceeding with the above steps.
| CIKM'24    | Reinforcement feature transformation for polymer property performance prediction | |  [paper](https://dl.acm.org/doi/abs/10.1145/3627673.3680105)    | `FG` | 
| ICDM'24    | Feature interaction aware automated data representation transformation. |   |  [paper](https://arxiv.org/pdf/2309.17011)    | `FG` | 
| Arxiv'2412    | AutoPrep: Natural Language Question-Aware Data Preparation with a Multi-Agent Framework |   |  [paper](https://arxiv.org/pdf/2412.10422)    | `Pipeline` | 
| Arxiv'2411    | Tablegpt2: A large multimodal model with tabular data integration |   |  [paper](https://arxiv.org/pdf/2411.02059)    | | 
| NIPS'23    | Reinforcement-enhanced autoregressive feature transformation: gradient-steered search in continuous space for postfix expressions |   |  [paper](https://arxiv.org/pdf/2010.08784)    | `FG` `FS` | 
| NIPS'23    | DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with Self-Correction |   |  [paper](https://arxiv.org/pdf/2304.11015)    |  | 
| ICML'23    | OpenFE: Automated Feature Generation with Expert-level Performance |   |  [paper](https://arxiv.org/abs/2211.12507)    | `FG` | 
| ICDE'23    | Toward Efficient Automated Feature Engineering |   |  [paper](https://epubs.siam.org/doi/pdf/10.1137/1.9781611978520.3?download=true)    | `FG` `FS` | 
| ICDE'23    | PA-FEAT: Fast Feature Selection for Structured Data via Progress-Aware Multi-Task Deep Reinforcement Learning |   |  [paper](https://ieeexplore.ieee.org/abstract/document/10184534)    | `FS` | 
| ICDE'23    | Toward a Unified Framework for Unsupervised Complex Tabular Reasoninh |   |  [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10184763&tag=1) | `Pipeline` |
| AAAI'23    | Efficient Top-K Feature Selection Using Coordinate Descent Method | |  [paper](https://dl.acm.org/doi/10.1609/aaai.v37i9.26258)    | `FS` | 
| AAAI'23    | T2G-FORMER: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction | |  [paper](https://arxiv.org/pdf/2211.16887)    | `FG` | 
| AAAI'23    | Weight Predictor Network with Feature Selection for Small Sample Tabular Biomedical Data | |  [paper](https://dl.acm.org/doi/10.1609/aaai.v37i8.26090)    | `FS` | 
| ACL'23 (Demo)    | OpenRT: An Open-source Framework for Reasoning Over Tabular Data | |  [paper](https://aclanthology.org/2023.acl-demo.32.pdf)    | ? | 
| ACL'23     | MURMUR: Modular multi-step reasoning for semistructured data-to-text generation | |  [paper](https://arxiv.org/pdf/2212.08607)    | ? | 
| KDD'23     | Treatment Effect Estimation with Adjustment Feature Selection |   |  [paper](https://dl.acm.org/doi/pdf/10.1145/3580305.3599531)    | `FS` | 
| KDD'23     | LATTE: A Framework for Learning Item-Features to Make a Domain-Expert for Effective Conversational Recommendation |   |  [paper](https://dl.acm.org/doi/10.1145/3580305.3599401)    | `FS` | 
| KDD'23     | Explicit Feature Interaction-aware Uplift Network for Online Marketing |   |  [paper](https://arxiv.org/pdf/2306.00315)    | `FG` | 
| KDD'23     | Cognitive Evolutionary Search to Select Feature Interactions for Click-Through Rate Prediction |   |  [paper](https://dl.acm.org/doi/10.1145/3580305.3599277)    | `FG` `FS` | 
| KDD'23     | Scenario-Adaptive Feature Interaction for Click-Through Rate Prediction |   |  [paper](https://dl.acm.org/doi/10.1145/3580305.3599936)    | `FG` `FS` | 
| SIGIR'23   | Single-shot feature selection for multi-task recommendations |   |  [paper](https://dl.acm.org/doi/abs/10.1145/3539618.3591767)    | `FS` |
| SIGIR'23   | Large Language Models are Versatile Decomposers: Decomposing Evidence and Questions for Table-based Reasoning |   |  [paper](https://dl.acm.org/doi/abs/10.1145/3539618.3591708)    | ? |
| TKDE'23    | Automated feature selection: A reinforcement learning perspective |   |  [paper](https://ieeexplore.ieee.org/abstract/document/9547816)    | `FS` | 
| ICDM'23    | Beyond discrete selection: Continuous embedding space optimization for generative feature selection |   |  [paper](https://arxiv.org/pdf/2302.13221)    | `FS` | 
| Arxiv/2307    | TableGPT: Towards Unifying Tables, Nature Language and Commands into One GPT |   |  [paper](https://arxiv.org/abs/2307.08674v3)    | `` | 
| ICML'22    | Difer: differentiable automated feature engineering. |   |  [paper](https://arxiv.org/abs/2211.12507)    | `FG` `FS` | 
| KDD'22     | AutoFAS: Automatic Feature and Architecture Selection for Pre-Ranking System |   |  [paper](https://arxiv.org/pdf/2205.09394)    | `FS` | 
| KDD'22     | Group-wise reinforcement feature generation for optimal and explainable representation space reconstruction. |   |  [paper](https://arxiv.org/pdf/2205.14526)    | `FG` | 
| KDD'22     | AdaFS: Adaptive Feature Selection in Deep Recommender System |   |  [paper](https://dl.acm.org/doi/10.1145/3534678.3539204)    | `FS` | 
| WWW'22     | Autofield: Automating feature selection in deep recommender systems |   |  [paper](https://dl.acm.org/doi/abs/10.1145/3485447.3512071)    | `FS` | 
| EMNLP'22   | Realistic Data Augmentation Framework for Enhancing Tabular Reasoning |   |  [paper](https://aclanthology.org/2022.findings-emnlp.324.pdf)    | `` | 
| EMNLP'22   | Leveraging Data Recasting to Enhance Tabular Reasoning |   |  [paper](https://aclanthology.org/2022.findings-emnlp.324.pdf)    | `` | 
| KDD'21     | Fives: Feature interaction via edge search for large-scale tabular data |   |  [paper](https://arxiv.org/pdf/2007.14573)    | `FG` | 
| CIKM'20    | Tolerant Markov Boundary Discovery for Feature Selection |   |  [paper](https://dl.acm.org/doi/10.1145/3340531.3415927)    | `FS` | 
| ICDM'20    | AutoFS: Automated Feature Selection via Diversity-aware Interactive Reinforcement Learning |   |  [paper](https://arxiv.org/pdf/2008.12001)    | `FS` | 
| ICDM'20    | Simplifying reinforced feature selection via restructured choice strategy of single agent. |   |  [paper](https://arxiv.org/pdf/2009.09230)    | `FS` | 
| KDD'19     | Automating Feature Subspace Exploration via Multi-Agent Reinforcement Learning |   |  [paper](https://dl.acm.org/doi/10.1145/3292500.3330868)    | `FS` | 

### Books, Tutorials and Survey Papers
[Principles of Data Wrangling: Practical Techniques for Data Preparation](https://dl.acm.org/doi/book/10.5555/3165161)

[Large Language Models for Tabular Data: Progresses and Future Directions](https://dl.acm.org/doi/abs/10.1145/3626772.3661384)

[A Survey on Table Mining with Large Language Models: Challenges, Advancements and Prospects](https://d197for5662m48.cloudfront.net/documents/publicationstatus/252177/preprint_pdf/3d9c9b7d57481675d0d6e486c8bb7985.pdf)

[Tabular Data-centric AI: Challenges, Techniques and Future Perspectives](https://dl.acm.org/doi/pdf/10.1145/3627673.3679102)

[A Survey on Data-Centric AI: Tabular Learning from Reinforcement Learning and Generative AI Perspective](https://arxiv.org/pdf/2502.08828)

[A Survey of Table Reasoning with Large Language Models](https://arxiv.org/abs/2402.08259)
[Table Meets LLM: Can Large Language Models Understand Structured Table Data? A Benchmark and Empirical Study](https://arxiv.org/pdf/2305.13062)
