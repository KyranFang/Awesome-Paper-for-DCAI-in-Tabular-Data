# Awesome-Resources-for-Better-Table-Reasoning

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
| VLDB'25     | Data Imputation with Limited Data Redundancy Using Data Lakes | Nan Tang  |      | `TP` |
| VLDB'25     | Birdie: Natural Language-Driven Table Discovery Using Differentiable Search Index | Yunjun Gao & Mingwei Zhou | [Paper](https://arxiv.org/pdf/2504.21282)   | `TP` |
| VLDB'25     | AutoPrep: Natural Language Question-Aware Data Preparation with a Multi-Agent Framework | Xiaoyong Du  |  [paper](https://arxiv.org/pdf/2412.10422)    | `TP` | This paper introduces Autoprep, a system that designed for automatically pre-process the tables for table reasoning task, regarding to the query. Three main data problems are emphasized in this work, Missing Semantics(a needed column missed), Inconsistent Value(values appears in different forms in a certain column) and Irrelevant column. AutoPrep decomposes the data prep process into three stages: planning stage, programming stage and executing stage. Planner will generate an SQL-like Analysis Sketch similar to Binder(ICLR'23), outlining how the table should be transformed to produce the answer. In programming stage, Programmer agents translate a high-level logical plan into a physical plan by generating low-level code, which is then passed to an Executor agent for code execution and interactive debugging. One should note all those processings are online. |
| SIGMOD'25   | Pneuma: Leveraging LLMs for Tabular Data Representation and Retrieval in an End-to-End System | Raul Castro Fernandez  |   [paper](https://arxiv.org/pdf/2504.09207) | `TP` | Pneuma system is a retrieval-augmented generation (RAG) system designed to efficiently discover tabular data. It does not only consider the table content information but also the context information. The system will firstly utilize LLM's knowledge to normalize provide meaningful column descriptions, even for abbreviations or domain-specific terms that may be challenging for humans or smaller models to interpret. Then, it use embedding model to generate vector representation of the table with context. In retrieval stage, the system will retrieves tables based on the user query 𝑄 by integrating three signals: lexical (BM25), semantic (vector search), and a signal based on LLM judgment. |
| SIGMOD'25   | Data+AI: LLM4Data and Data4LLM | Guoliang Li |  [paper](https://dl.acm.org/doi/10.1145/3722212.3725641)   |  |
| SIGMOD'25   | Adda: Towards Efficient in-Database Feature Generation via LLM-based Agents | |  [paper](https://dl.acm.org/doi/10.1145/3725262)    | `FG` | 
| SIGMOD'25   | GEIL: A Graph-Enhanced Interpretable Data Cleaning Framework with Large Language Models | |  [paper](https://dl.acm.org/doi/10.1145/3698811)    | `` | 
| SIGMOD'25   | Auto-Test: Learning Semantic-Domain Constraints for Unsupervised Error Detection in Tables | |  [paper](https://arxiv.org/pdf/2504.10762) | ? |
| AAAI'25     | Dynamic and Adaptive Feature Generation with LLM | |  [paper](https://ojs.aaai.org/index.php/AAAI/article/view/33851)    | `FG` | 
| Nature'25   | Accurate predictions on small data with a tabular foundation model |  Frank Hutter |  [paper](https://www.nature.com/articles/s41586-024-08328-6)    | |
| ICML'25     | Are Large Language Models Ready for Multi-Turn Tabular Data Analysis? | |  [paper](https://openreview.net/attachment?id=flKhxGTBj2&name=pdf)    | `` | 
| ICML'25     | Compositional Condition Question Answering in Tabular Understanding | |  [paper](https://openreview.net/attachment?id=aXU48nrA2v&name=pdf)    | `` | 
| ICML'25     | Quantifying Prediction Consistency Under Fine-tuning Multiplicity in Tabular LLMs | |  [paper](https://arxiv.org/pdf/2407.04173v2)    | `` |
| ICML'25     | TabICL: A Tabular Foundation Model for In-Context Learning on Large Data | Marine Le Morvan |  [paper](https://arxiv.org/pdf/2502.05564)    | `` |
| ICML'25     | FairPFN: A Tabular Foundation Model for Causal Fairness | Frank Hutter |  [paper](https://arxiv.org/pdf/2407.05732)    | `` |
| ICML'25     | TabPFN Unleashed: A Scalable and Effective Solution to Tabular Classification Problems | Han-Jia Ye |  [paper](https://arxiv.org/abs/2502.02527)    | `` |
| ICML'25     | Compositional Condition Question Answering in Tabular Understanding | Han-Jia Ye |  [paper](https://openreview.net/attachment?id=aXU48nrA2v&name=pdf)    | `` |
| ICLR'25     | Exploring LLM Agents for Cleaning Tabular Machine Learning Datasets | Christian Holz |  [paper](https://arxiv.org/abs/2503.06664)    | `` |
| IJCAI'25    | Evolutionary Large Language Model for Automated Feature Transformation | |  [paper](https://arxiv.org/pdf/2406.03505)    | `FG` | 
| ICDM'25     | OpenFE++: Efficient Automated Feature Generation via Feature Interaction |   |  [paper](https://arxiv.org/pdf/2504.17356)    | `FS` | 
| NAACL'25    | ALTER: Augmentation for Large-Table-Based Reasoning | Hanfang Yang |  [paper](https://aclanthology.org/2025.naacl-long.9/)    | Table Sampling | The ALTER framework enhances large-table reasoning through a workflow that augments queries into sub-queries and tables with schema, semantic, and literal information, filters relevant rows/columns via embedding-based sampling and LLM-driven selection, generates and executes SQL to obtain sub-tables, and uses a joint reasoner to aggregate results from primary and sub-query workflows.
| NAACL'25    | TART: An Open-Source Tool-Augmented Framework for Explainable Table-based Reasoning |   |  [paper](https://arxiv.org/pdf/2409.11724)    | `DA` | TART contains three key components: a table formatter (clean and unify the format of the table) to ensure accurate data representation, a tool maker (python code) to develop specific computational tools, and an explanation generator to maintain explainability. 
| NAACL'25    | H-STAR: LLM-driven Hybrid SQL-Text Adaptive Reasoning on Tables | Dan Roth & Chandan K. Reddy |  [paper](https://aclanthology.org/2025.naacl-long.445.pdf)    | ?  | 
| Arxiv'2505  | Text-to-Pipeline: Bridging Natural Language and Data Preparation Pipelin | Yunjun Gao |  [paper](https://arxiv.org/abs/2505.15874)    | `TP` | This papere introduces a table preparation pipeline called Text-to-pipeline which will translated users' input NL queries to data preparation pipelines. Although Autoprep(VLDB'25) can automatically prepare on the Question in QA task, but this paper argues that Autoprep does not support general-purpose pipeline generation. Text-to-pipeline formalizes it as symbolic program generation in a domain-specific language (DSL), which can be compiled into executable backend code such as Pandas or SQL. |
| Arxiv'2507  | Reinforcement Learning-based Feature Generation Algorithm for Scientific Data | |  [paper](https://arxiv.org/abs/2507.03498)    | `FG` | 
| Arxiv'2506  | What to Keep and What to Drop: Adaptive Table Filtering Framework | |  [paper](https://arxiv.org/pdf/2506.23463)    | `` | 
| Arxiv'2501  | TableMaster: A Recipe to Advance Table Understanding with Language Models          | |  [paper](https://arxiv.org/pdf/2501.19378)    | ?    |
| Arxiv'2502  | Towards Question Answering over Large Semi-structured Tables  |   |  [paper](https://arxiv.org/pdf/2502.13422)    | ?    |
| Arxiv'2505  | Weaver: Interweaving SQL and LLM for Table Reasoning | Vivek Gupta |  [paper](https://arxiv.org/pdf/2505.18961)    | `TR` | This paper proposed Weaver system that weave SQL and LLMs for table-based question answering. This system decomposes the reasoning task into four stage: pre-process stage, planning stage, code execution stage and answer extraction stage. Planner decomposes the query into several sequential subtasks, each of them could be done by either SQL or LLM. And a secondary LLM is assigned to verify the initial plan. Following planning, the Weaver executes the plan sequentially, combining SQL queries and LLM-generated prompts. In the final pipeline stage, the intermediate table and user query are inputted to an LLM, which generates a natural language answer.|
| NIPS'24    | TableRAG: Million-Token Table Understanding with Language Models |   |  [paper](https://arxiv.org/pdf/2410.04739)    | ? | 
| ICLR'24    | OpenTab: Advancing Large Language Models as Open-domain Table Reasoners |   |  [paper](https://arxiv.org/pdf/2402.14361)    | ? | 
| ICLR'24    | CABINET: Content Relevance based Noise Reduction for Table Question Answering |   |  [paper](https://arxiv.org/pdf/2402.01155)    | ? | 
| ICLR'24    | Chain-of-Table: Evolving Tables in the Reasoning Chain for Table Understanding | Tomas Pfister  |  [paper](https://arxiv.org/pdf/2401.04398)    | `TP` | CHAIN-OF-TABLE enables LLMs to dynamically plan a chain of operations over a table T in response to a given question Q. It utilizes atomic tool-based operations to construct the table chain. These operations include adding columns, selecting rows or columns, grouping, and sorting, which are common in SQL and DataFrame development. After processing the table, the system will query the reasoning LLM with the last intermediate Table and the NL query. | 
| ICLR'24    | ReMasker: Imputing Tabular Data with Masked Autoencoding | Artem Babenko  |  [paper](https://openreview.net/pdf?id=KI9NqjLVDT)    | ? | 
| ICLR'24    | TabR: Tabular Deep Learning Meets Nearest Neighbors |   |  [paper](https://openreview.net/pdf?id=rhgIgTSSxW)    | ? | 
| ICLR'24    | Making Pre-trained Language Models Great on Tabular Prediction |   Jintai Chen |  [paper](https://openreview.net/pdf?id=anzIzGZuLi)    | ? | 
| ICML'24    | Large Language Models Can Automatically Engineer Features for Few-Shot Tabular Learning |   |  [paper](https://openreview.net/pdf?id=anzIzGZuLi)    | ? |
| ICML'24    | Position: Why Tabular Foundation Models Should Be a Research Priority | Mihaela van der Schaar |  [paper](https://openreview.net/pdf?id=amRSBdZlw9)    | ? |
| ICML'24    | Curated LLM: Synergy of LLMs and Data Curation for tabular augmentation in low-data regimes | Gael Varoquaux |  [paper](https://arxiv.org/pdf/2402.16785)    | ? |
| ICML'24    | CARTE: Pretraining and Transfer for Tabular Learning | Mihaela van der Schaar |  [paper](https://openreview.net/pdf?id=9cG1oRnqNd)    | ? |
| ICML'24    | TabLog: Test-Time Adaptation for Tabular Data Using Logic Rules | Vasant Honavar |  [paper](https://openreview.net/pdf?id=LZeixIvQcB)    | ? |
| ICML'24    | Tabular Insights, Visual Impacts: Transferring Expertise from Tables to Images | De-Chuan Zhan |  [paper](https://openreview.net/forum?id=v7I5FtL2pV)    | ? |
| ICML'24    | Large Language Models Can Automatically Engineer Features for Few-Shot Tabular Learning | Tomas Pfister | [paper](https://openreview.net/forum?id=v7I5FtL2pV)    | ? |
| ICML'24 workshop   | Learning to Reduce: Towards Improving Performance of Large Language Models on Structured Data|   |  [paper](https://arxiv.org/pdf/2407.02750)    | ? | 
| KDD'24      | Unsupervised Generative Feature Transformation via Graph Contrastive Pre-training and Multi-objective Fine-tuning |   |  [paper](https://arxiv.org/pdf/2405.16879)    | `FG` | 
| KDD'24      | Feature selection as deep sequential generative learning. |   |  [paper](https://arxiv.org/pdf/2403.03838)    | `FS` | 
| KDD'24      | Can a Deep Learning Model be a Sure Bet for Tabular Prediction? |  |  [paper](https://dl.acm.org/doi/10.1145/3637528.3671893) | |
| KDD'24      | From Supervised to Generative: A Novel Paradigm for Tabular Deep Learning with Large Language Models | |  [paper](https://arxiv.org/pdf/2310.07338) ||
| SIGMOD'24   | SAGA: A Scalable Framework for Optimizing Data Cleaning Pipelines for Machine Learning Applications | Yin Lou |  [paper](https://dl.acm.org/doi/10.1145/3654942)    | `FS` | 
| SIGMOD'24   | FeatureLTE: Learning to Estimate Feature Importance |   |  [paper](https://dl.acm.org/doi/10.1145/3617338)    | `FS` | 
| SIGMOD'24   | Solo: Data Discovery Using Natural Language Questions Via A Self-Supervised Approach | Raul Castro Fernandez |  [paper](https://arxiv.org/pdf/2301.03560v2)    | `TP` | SOLO introduces a more fine-grained representation by encoding each cell–attribute–cell triplet into a fixeddimensional embedding. For indexing, most methods rely on approximate nearest neighbor (ANN) algorithms, such as IVF-PQ. This work also introduces a training data synthesizing method and a following self-supervised training paradigm. However, it suffers froms from long training time and huge storage use. |
| VLDB'24     | ReAcTable: Enhancing ReAct for Table Question Answering | Jignesh M. Patel  |  [paper](https://arxiv.org/pdf/2310.00815)    | `TP` `TR` | ReAcTable is inspired by ReAct, which combined CoT and Tool-using in one framework. In the ReAcTable Framework, an LLM would break the problem into multiple steps and generate SQL code or Python code for processing the table and generate intermediate table for better reasoning. The framework als0 design some techniques to handle the exception like SQL query requires a column that does not exist in the given table.
| VLDB'24     | Generalizable Data Cleaning of Tabular Data in Latent Space | Carsten Binnig  |  [paper](https://www.vldb.org/pvldb/vol17/p4786-reis.pdf)    | ? | 
| VLDB'24     | AutoTQA: Towards Autonomous Tabular Qestion Answering through Multi-Agent Large Language Models | Qi Liu |  [paper](https://www.vldb.org/pvldb/vol17/p3920-zhu.pdf)    | `TR` | | 
| EMNLP'24    | NormTab: Improving Symbolic Reasoning in LLMs Through Tabular Data Normalization | Davood Rafiei |  [paper](https://arxiv.org/pdf/2406.17961)    | `TP` | 
| EMNLP'24    | ProTrix: Building Models for Planning and Reasoning over Tables with Sentence Context | Yansong Feng |  [paper](https://arxiv.org/pdf/2403.02177) | ? | 
| EMNLP'24    | TAP4LLM: Table Provider on Sampling, Augmenting, and Packing Semi-structured Data for Large Language Model Reasoning |   |  [paper](https://arxiv.org/pdf/2312.09039)    | `TP` | This paper constructed a pipeline to boost LLM's performance of table reasoning by introducing three components, Table Sampling Module, Table Augmentationd Module and Table Packing Module. In each module, the authors designed and compared several common methods under various usage scenarios, aiming to searching for best practices for leveraging LLMs for table reasoning tasks.  |
| EMNLP'24 (Demo)   | OpenT2T: An Open-Source Toolkit for Table-to-Text Generation |   |  [paper](https://aclanthology.org/2024.emnlp-demo.27.pdf)    | ? | 
| ACL'24      | Is Table Retrieval a Solved Problem? Exploring Join-Aware Multi-Table Retrieval | Roth Dan |  [paper](https://arxiv.org/pdf/2404.09889) | ? |
| NAACL'24    | Rethinking Tabular Data Understanding with Large Language Models | |  [paper](https://arxiv.org/pdf/2312.16702) | ? |
| NAACL'24    | TabSQLify: Enhancing Reasoning Capabilities of LLMs Through Table Decomposition |   |  [paper](https://arxiv.org/pdf/2404.10150)    | ? | 
| NAACL'24    | e5: zero-shot hierarchical table analysis using augmented llms via explain, extract, execute, exhibit and extrapolate |   |  [paper](https://aclanthology.org/2024.naacl-long.68.pdf)    | ? |  First, the model is guided by designed prompt to understand the hierarchical structure of the table, including the multi-level headers and their implicit semantic relationships. Then, it generates code to pull out the data rows and columns relevant to the query, along with performing necessary operations like filtering or calculations. Next, an external tool runs this code to get accurate results, preventing the model from making up information. These results are then presented clearly. Finally, the model uses its reasoning ability to analyze these results and derive the final answer to the query. For large tables that exceed token limits, the pipeline first compresses them by identifying and keeping only the most relevant data, while adding back potentially useful information to ensure key details aren’t lost, before proceeding with the above steps.
| CIKM'24    | Reinforcement feature transformation for polymer property performance prediction | |  [paper](https://dl.acm.org/doi/abs/10.1145/3627673.3680105)    | `FG` | 
| ICDM'24    | Feature interaction aware automated data representation transformation. |   |  [paper](https://arxiv.org/pdf/2309.17011)    | `FG` | 
| Arxiv'2411    | Tablegpt2: A large multimodal model with tabular data integration |   |  [paper](https://arxiv.org/pdf/2411.02059)    | | 
| Arxiv'2407    | Talent: A Tabular Analytics and Learning Toolbox | Han-Jia Ye  |  [paper](https://arxiv.org/pdf/2407.04057)    | | 
| SIGMOD'23  | Generation of Training Examples for Tabular Natural Language Inference | Paolo Papotti  |  [paper](https://dl.acm.org/doi/10.1145/3626730)    |  | 
| NIPS'23    | Reinforcement-enhanced autoregressive feature transformation: gradient-steered search in continuous space for postfix expressions |   |  [paper](https://arxiv.org/pdf/2010.08784)    | `FG` `FS` | 
| NIPS'23    | DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with Self-Correction |   |  [paper](https://arxiv.org/pdf/2304.11015)    |  | 
| ICML'23    | OpenFE: Automated Feature Generation with Expert-level Performance |   |  [paper](https://arxiv.org/abs/2211.12507)    | `FG` | 
| ICLR'23    | TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second | Frank Hutter |  [paper](https://openreview.net/pdf?id=cp5PvcI6w8_)    | ? |  
| ICLR'23    | Binding Language Models in Symbolic Languages | Tao Yu |  [paper](https://openreview.net/pdf?id=lH1PV42cbF)    | Tool Using | BINDER, a training-free neural-symbolic framework, extends programming language grammar coverage by binding language model functionalities via a unified API; during execution, it parses programs into abstract syntax trees (ASTs) based on the extended grammar to support nested API calls. For the operations exceeding SQL capabilities, BINDER assigns an LLM to complete the task via APIs, treating it as a special new identifier in grammar and a node in ASTs. The result returned from the LLM is stored as a variable compatible with the standard symbolic language grammar for deriving the final result.
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
| VLDB'21    | Deep Entity Matching with Pre-Trained Language Models | Wang-Chiew Tan  |  [paper](https://arxiv.org/abs/2004.00584)    | `FG` | 
| SIGMOD'20  | Finding related tables in data lakes for interactive data science.  | Zachary G. Ives  |  [paper](https://dl.acm.org/doi/10.1145/3340531.3415927)    | `TP` | Introduced Juneau, a system for table discovering on data lakes. But it needs professional skillset to use, which makes it inconvenient for general users. | 
| ACL'20     | TAPAS: Weakly Supervised Table Parsing via Pre-training |  Julian Martin Eisenschlos  |  [paper](https://arxiv.org/pdf/2004.02349)    | `TP` | 
| CIKM'20    | Tolerant Markov Boundary Discovery for Feature Selection |   |  [paper](https://dl.acm.org/doi/10.1145/3340531.3415927)    | `FS` | 
| ICDM'20    | AutoFS: Automated Feature Selection via Diversity-aware Interactive Reinforcement Learning |   |  [paper](https://arxiv.org/pdf/2008.12001)    | `FS` | 
| ICDM'20    | Simplifying reinforced feature selection via restructured choice strategy of single agent. |   |  [paper](https://arxiv.org/pdf/2009.09230)    | `FS` | 
| KDD'19     | Automating Feature Subspace Exploration via Multi-Agent Reinforcement Learning |   |  [paper](https://dl.acm.org/doi/10.1145/3292500.3330868)    | `FS` | 

### Books, Tutorials and Survey Papers
[Natural Language Interfaces for Tabular DataQuerying and Visualization: A Survey](https://arxiv.org/pdf/2310.17894)

[Large Language Models(LLMs) on Tabular Data: Prediction, Generation, and Understanding - A Survey](https://arxiv.org/pdf/2402.17944v2)

[Principles of Data Wrangling: Practical Techniques for Data Preparation](https://dl.acm.org/doi/book/10.5555/3165161)

[Large Language Models for Tabular Data: Progresses and Future Directions](https://dl.acm.org/doi/abs/10.1145/3626772.3661384)

[A Survey on Table Mining with Large Language Models: Challenges, Advancements and Prospects](https://d197for5662m48.cloudfront.net/documents/publicationstatus/252177/preprint_pdf/3d9c9b7d57481675d0d6e486c8bb7985.pdf)

[Tabular Data-centric AI: Challenges, Techniques and Future Perspectives](https://dl.acm.org/doi/pdf/10.1145/3627673.3679102)

[A Survey on Data-Centric AI: Tabular Learning from Reinforcement Learning and Generative AI Perspective](https://arxiv.org/pdf/2502.08828)

[A Survey of Table Reasoning with Large Language Models](https://arxiv.org/abs/2402.08259)
[Table Meets LLM: Can Large Language Models Understand Structured Table Data? A Benchmark and Empirical Study](https://arxiv.org/pdf/2305.13062)
