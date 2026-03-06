# Runtime Behavior

Prompt:

> Describe what happens when this GNN model runs and how it would behave in different settings or domains.

Response:

Okay, let’s break down what happens when this “Three-Level Temporal Hierarchy Agent” GNN model runs, and how its behavior would likely manifest across different settings and domains. This is a complex model, so let’s unpack it step-by-step.

**Core Mechanics: Active Inference at Multiple Scales**

At its heart, this GNN implements Active Inference, a framework developed by Friston and colleagues.  Active Inference posits that we constantly generate models of the world to explain our sensory input.  Instead of passively receiving information, we actively *infer* the causes of our perceptions. This model does this across three distinct timescales, which is key to its design.

* **Level 0 (Fast – 100ms): Sensorimotor Control** – This is the most immediate level. It’s responsible for reflexive actions – things like reaching for a glass, adjusting your posture, or blinking. The model predicts the sensory consequences (observations, *o0*) of potential actions (*u0*) and chooses the action that minimizes the expected free energy (EFE, *G0*). This level is driven by a strong prior – a belief about how the world *should* be – and constantly updates this belief based on incoming sensory data. The A0 matrix represents the likelihood of observing the sensory data given the current state, while B0 represents the transition probabilities of the state given an action.

* **Level 1 (Medium – 1s): Tactical Planning** – This level operates on a slightly longer timescale, generating tactical plans. It receives summarized information from Level 0 (the trajectory of the sensory state) and uses this to set goals. It then generates actions to achieve those goals, again minimizing expected free energy. The A1 matrix represents the likelihood of observing the tactical observations given the current tactical state, while B1 represents the transition probabilities of the tactical state given an action.

* **Level 2 (Slow – 10s): Strategic Planning** – This is the highest-level, long-term planning level. It receives summaries of the outcomes from Level 1 and uses this information to adjust its overall strategic goals. It’s essentially managing the long-term consequences of its tactical decisions. The A2 matrix represents the likelihood of observing the strategic observations given the current strategic state, while B2 represents the transition probabilities of the strategic state given an action.


**How the GNN Runs – A Discrete Simulation**

1. **Initialization:** The model starts