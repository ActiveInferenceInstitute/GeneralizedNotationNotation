# PoE-World: Compositional World Modeling with Products of Programmatic Experts

## Overview and Core Innovation

PoE-World represents a groundbreaking approach to world modeling that combines program synthesis with Large Language Models (LLMs) to create compositional world models for complex, non-gridworld domains [1][2]. Developed by researchers from Cornell University, University of Cambridge, The Alan Turing Institute, and Dalhousie University, this work introduces a novel method for learning world models as exponentially-weighted products of programmatic experts synthesized by LLMs [3][2].

The fundamental innovation lies in representing world models not as single monolithic programs, but as collections of small, specialized programs that each capture specific environmental rules [1][4]. This modular approach enables sample-efficient learning from just a few observations while maintaining the ability to generalize to unseen scenarios [2][5].

## Technical Architecture and Methodology

### Core Representation

The PoE-World framework models environmental dynamics using the mathematical formulation:

```
p_θ(o_{t+1}|o_{1:t}, a_{1:t}) ∝ ∏_i p^{expert}_i(o_{t+1}|o_{1:t}, a_{1:t})^{θ_i}
```

where each `p^{expert}_i` represents a programmatic expert and `θ_i` are scalar weights that determine the influence of each expert [2]. This representation enables modularity and compositionality by allowing many small programs to be combined into a comprehensive world model [2].

### Learning Algorithm

The learning process follows a structured approach consisting of four key steps [2]:

1. **Program Synthesis**: LLMs generate programmatic experts from observed trajectory data
2. **Weight Optimization**: Scalar weights for each expert are fitted using gradient-based optimization
3. **Expert Pruning**: Experts with weights below a threshold are removed
4. **Iterative Refinement**: The process repeats as new trajectory data becomes available

Each expert program represents simple causal rules within the environment, such as "if a player touches a skull, then the player dies" or "if an action is LEFT when the player is on a platform, the player's x-axis velocity is -2" [2]. This granular approach allows the system to capture complex environmental dynamics through the composition of simpler rules.

### Integration with Planning

PoE-World is embedded within model-based planning agents that utilize Monte Carlo Tree Search (MCTS) for decision-making [3][6]. The MCTS algorithm enables the agent to simulate potential future outcomes using the learned world model, allowing for strategic planning in complex environments [6][7]. The system employs a hierarchical planning approach that breaks down complex tasks into manageable subgoals, particularly effective for challenging games like Montezuma's Revenge [8].

## Technical Implementation Details

### Development Environment

The PoE-World repository provides a comprehensive implementation built on Python 3.10 with several key dependencies [3]:

- **OpenAI Interface**: Integration with OpenAI's API for LLM-based program synthesis
- **OCAtari Framework**: Object-centric Atari environment wrapper for extracting structured representations [3][9][10]
- **Gymnasium**: Standard reinforcement learning environment interface with Atari support

The system supports multiple execution modes including PoE-World, WorldCoder, ReAct, and PPO baselines, allowing for comprehensive comparative evaluation [3].

### Key Implementation Files

The architecture is organized around several critical components [3]:

- **Agent System** (`agents/agent.py`): Main agent class managing environment interaction and world model updates
- **Planning Module** (`agents/mcts.py`): MCTS implementation for strategic decision-making
- **Learning Framework** (`learners/world_model_learner.py`): Orchestrates the learning of object-specific models
- **Program Synthesis** (`learners/synthesizer.py`): LLM-based generation of programmatic experts
- **Model Architecture** (`learners/models.py`): Weight fitting and expert combination mechanisms

## Experimental Evaluation and Results

### Performance Benchmarks

PoE-World demonstrates remarkable performance across challenging Atari environments, particularly excelling in scenarios requiring strategic planning and sparse reward handling [1][4]. The system was evaluated on both standard games (Pong, Montezuma's Revenge) and modified variants (PongAlt, Montezuma's RevengeAlt) to test generalization capabilities [11].

### Sample Efficiency Achievements

The approach achieves exceptional sample efficiency by learning complex world models from minimal demonstration data [4][11]. In comparative evaluations, PoE-World significantly outperformed baseline methods including standard reinforcement learning algorithms, LLM agents without world models, and previous symbolic approaches like WorldCoder [4][8].

### Generalization Capabilities

A key strength of PoE-World lies in its ability to generalize to novel scenarios that differ from training conditions [11]. The modular nature of the programmatic experts allows the system to recombine learned rules in new ways, enabling adaptation to modified game environments without additional training [12][11].

### Montezuma's Revenge Performance

PoE-World achieved particularly noteworthy results on Montezuma's Revenge, a notoriously difficult exploration problem in reinforcement learning [4][8]. The system was the only method tested that successfully achieved positive rewards when trained on very limited data, demonstrating the effectiveness of the compositional approach for complex planning tasks [8].

## Research Team and Academic Context

### Principal Investigators

The project is led by **Wasu Top Piriyakulkij**, a PhD student at Cornell University specializing in world modeling and program synthesis [13][14]. Under the supervision of **Kevin Ellis**, a prominent researcher in program synthesis and neural-symbolic AI [15][16], the work represents a significant advancement in applying LLMs to structured world model learning.

### Collaborative Research Network

The interdisciplinary team includes researchers from multiple prestigious institutions [2]:

- **Cornell University**: Wasu Top Piriyakulkij, Hao Tang, Kevin Ellis
- **University of Cambridge**: Yichao Liang, Adrian Weller
- **The Alan Turing Institute**: Adrian Weller
- **Dalhousie University**: Marta Kryven

This collaboration brings together expertise in program synthesis, probabilistic inference, machine learning, and cognitive modeling [15][16].

### Academic Impact and Publication

The work was published as a conference paper with the arXiv preprint available since May 2025 [1][5]. The research builds upon Kevin Ellis's extensive work in program synthesis, including influential papers such as "DreamCoder: Bootstrapping Inductive Program Synthesis with Wake-Sleep Library Learning" [16][17].

## Technical Foundations and Related Work

### Program Synthesis Background

PoE-World leverages advances in neural program synthesis, particularly the use of LLMs for code generation [18][19]. Unlike traditional approaches that generate single large programs, the method decomposes world modeling into manageable components that can be individually synthesized and combined [2][18].

### Object-Centric Representation Learning

The system builds upon object-centric approaches to reinforcement learning, utilizing the OCAtari framework for extracting structured representations from Atari games [9][10][20]. This foundation enables the programmatic experts to operate on meaningful object-level abstractions rather than raw pixel data [21].

### Product of Experts Framework

The mathematical foundation draws from the Product of Experts (PoE) paradigm in machine learning, where multiple models are combined multiplicatively rather than additively [22]. This approach ensures that all experts must agree for a prediction to have high probability, creating a natural constraint satisfaction mechanism [22].

### Monte Carlo Tree Search Integration

The planning component utilizes MCTS, a proven algorithm for decision-making in complex domains with large branching factors [6][7][23]. The integration of learned world models with MCTS enables efficient exploration and strategic planning in challenging environments [6].

## Broader Impact and Future Directions

### Advancing Sample-Efficient Learning

PoE-World addresses a fundamental challenge in AI: learning effective policies from limited experience [2][24]. By enabling few-shot learning of complex world models, the approach has implications for robotics, autonomous systems, and other domains where data collection is expensive or dangerous [24].

### Interpretability and Explainability

The programmatic nature of the learned experts provides inherent interpretability, as the rules governing agent behavior are expressed in readable code [12][2]. This transparency is crucial for applications requiring explainable AI, particularly in safety-critical domains.

### Scalability and Modularity

The modular architecture enables incremental learning and knowledge transfer across domains [2]. As new environmental rules are discovered, they can be added to the existing expert pool without disrupting previously learned knowledge, supporting lifelong learning scenarios.

### Limitations and Future Work

Current limitations include dependence on LLM quality for program synthesis and computational overhead from expert combination [2]. Future research directions may explore more efficient expert selection mechanisms, automated hyperparameter tuning, and extension to continuous action spaces [25].

## Conclusion

PoE-World represents a significant advancement in world modeling for artificial intelligence, successfully bridging the gap between traditional deep learning approaches and symbolic reasoning [1][4]. By leveraging LLMs to synthesize modular programmatic experts, the system achieves remarkable sample efficiency while maintaining interpretability and generalization capabilities [2][5]. The work demonstrates the potential of hybrid neural-symbolic approaches for complex sequential decision-making tasks and establishes a foundation for future research in compositional world modeling [4][11].

The open-source nature of the implementation, combined with comprehensive evaluation across challenging benchmarks, positions PoE-World as a valuable contribution to the reinforcement learning and program synthesis communities [3][12]. As the field continues to explore the intersection of large language models and structured reasoning, PoE-World provides a compelling example of how these technologies can be effectively combined to address fundamental challenges in artificial intelligence.

[1] https://topwasu.github.io/poe-world
[2] https://www.youtube.com/watch?v=vkxNy4HXB30
[3] https://github.com/topwasu/poe-world
[4] https://www.youtube.com/watch?v=oV4x8hS51mU
[5] https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
[6] https://gibberblot.github.io/rl-notes/single-agent/mcts.html
[7] https://www.reddit.com/r/MachineLearning/comments/86s1rl/p_monte_carlo_tree_search_beginners_guide/
[8] https://arbs.nzcer.org.nz/predict-observe-explain-poe
[9] https://github.com/k4ntz/OC_Atari
[10] https://openreview.net/forum?id=4PzxLPEGRn
[11] https://www.cs.cornell.edu/~wp237/
[12] https://www.marktechpost.com/2025/06/20/poe-world-outperforms-reinforcement-learning-rl-baselines-in-montezumas-revenge-with-minimal-demonstration-data/
[13] https://openreview.net/pdf/570da99bc96fd9c121ec403723bb49754b11fefb.pdf
[14] https://classes.cornell.edu/browse/roster/FA23/class/CS/6172
[15] https://scholar.google.com/citations?user=tVjxANMAAAAJ
[16] https://papers.nips.cc/paper/5785-unsupervised-learning-by-program-synthesis
[17] https://papers.nips.cc/paper_files/paper/2019/hash/50d2d2262762648589b1943078712aa6-Abstract.html
[18] https://worldmodels.github.io
[19] https://openreview.net/profile?id=~Top_Piriyakulkij1
[20] https://arxiv.org/html/2306.08649v2
[21] https://sites.google.com/view/code-world-models/home
[22] https://arxiv.org/abs/2306.08649
[23] https://evanthebouncy.github.io/program-synthesis-minimal/generation-with-llm/
[24] https://www.pathofexile.com/forum/view-thread/3155590
[25] https://icaps20subpages.icaps-conference.org/wp-content/uploads/2020/10/PRL2020_paper_21.pdf
[26] https://arxiv.org/abs/2505.10819
[27] https://arxiv.org/pdf/2505.10819v1.pdf
[28] https://arxiv.org/abs/2505.10819v2
[29] https://www.youtube.com/watch?v=Fbs4lnGLS8M
[30] https://openreview.net/forum?id=JDa5RiTIC7
[31] http://www.scholarpedia.org/article/Product_of_experts
[32] https://paperswithcode.com/paper/generating-code-world-models-with-large
[33] https://openreview.net/forum?id=EHmjRIA4l2
[34] https://proceedings.neurips.cc/paper_files/paper/2023/file/5647763d4245b23e6a1cb0a8947b38c9-Paper-Conference.pdf
[35] https://www.cs.cornell.edu/~ellisk/
[36] https://podcasts.apple.com/us/podcast/210-hao-tang-beyond-five-stars-the-art-science/id1657265985?i=1000662108621
[37] https://openreview.net/pdf/6caf6e01a8a350a8fd0bf4fe14d7f7d4f54c0ada.pdf
[38] https://papers.nips.cc/paper_files/paper/2015/hash/b73dfe25b4b8714c029b37a6ad3006fa-Abstract.html
[39] https://www.youtube.com/watch?v=qXGAMLS1QsY
[40] https://files.eric.ed.gov/fulltext/EJ1341654.pdf
[41] https://pdfs.semanticscholar.org/d5d6/248eabda02a855441b8ed03bc048df48d214.pdf
[42] https://www.research.ed.ac.uk/files/181350841/Exploration_by_Random_BURDA_DoA211218_AFV.pdf
[43] https://milvus.io/ai-quick-reference/what-are-the-steps-involved-in-implementing-a-fewshot-learning-model
[44] https://arxiv.org/pdf/2403.03997.pdf
[45] https://scalefm.com/2014/11/poe-recap-performance-metrics-analytics/
[46] https://www.wbdg.org/resources/post-occupancy-evaluations
[47] https://www.sciencedirect.com/science/article/pii/S0956713521000487
[48] http://www.arxiv.org/pdf/2502.11537.pdf
[49] https://github.com/jiahangtian/beating-montezuma
[50] https://neptune.ai/blog/model-based-and-model-free-reinforcement-learning-pytennis-case-study
[51] https://x.com/Marktechpost/status/1936125324124897511
[52] https://ai.stackexchange.com/questions/22914/how-to-run-a-monte-carlo-tree-search-mcts-for-stochastic-environment
[53] https://stackoverflow.com/questions/58911784/how-to-understand-the-4-steps-of-monte-carlo-tree-search
[54] https://oc-atari.readthedocs.io
[55] https://www.dfki.de/en/web/research/projects-and-publications/publication/14042
[56] https://x.com/ellisk_kellis?lang=en
[57] https://www.reddit.com/r/pathofexile/comments/1huybnw/new_research_on_increase_item_rarity_with_over/
[58] https://www.reddit.com/r/PathOfExile2/comments/1g0488j/performance_metrics_vs_poe1/
[59] https://towardsdatascience.com/comprehensive-guide-to-ranking-evaluation-metrics-7d10382c1025/