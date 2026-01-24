# PRACTICAL_APPLICATIONS

The provided GNN section contains information on a classical active inference agent for discrete Markov decision processes with one observation modality (state-action pairs). Specifically, it introduces the following components:

1. **GnnPOMDPAgent**: A simple class of models that are well-suited for applications like simulation or machine learning. It describes an agent that takes input data from a single observation and outputs actions based on its knowledge to achieve a given policy.
2. **GenerativeModelX**: A generic GNN model with parameters X representing each state variable, action pattern vector, probability distribution over states, and prior distributions for states up to 3 (observations). It is designed as an initial agent for simulation or machine learning applications.
3. **GnnPOMDPAgentBeta**: A beta-based model with a probabilistic graphical representation of GNNs. Specifically, it takes input data from one observation and outputs actions based on its knowledge to achieve the same policy. It is designed as an initial agent for simulation or machine learning applications.
4. **GnnPOMDPAgent**: An implementation of GNN models in Python with built-in support using NumPy/scipy arrays and a simple functional programming interface.
5. **GenerativeModelX**: A generic model that can be applied to any probabilistic graphical representation of an action pattern vector for generating actions based on the corresponding probability distribution over states. It is designed as an initial agent for simulation or machine learning applications.
6. **GnnPOMDPAgent**: A generalized probability model with parameters X representing each state variable, action parameter, prior distribution, and prior probabilities across all observations (observations) of a probabilistic graphical representation.
7. **GenerativeModelX**: A generic model that can be applied to any probabilistic graphical representation of an action pattern vector for generating actions based on the corresponding probability distribution over states. It is designed as an initial agent for simulation or machine learning applications.
8. **GnnPOMDPAgentBeta**: A beta-based model with a probabilistic graph representation of GNNs, specifically applied to simulate action selection from policy posterior and estimate beliefs. It is designed as an initial agent for simulation or machine learning applications.
9. **GenerativeModelX**: A generic probabilistic graphical representation of GNN models in Python using NumPy/scipy arrays.
10. **GnnPOMDPAgentBeta**: A beta-based model with a probabilistic graph representation of GNN models, specifically applied to simulate action selection from policy posterior and estimate beliefs based on prior probabilities across all observations (observations). It is designed as an initial agent for simulation or machine learning applications.
11. **GenerativeModelX**: A generic probabilistic graphical representation of GNNs with beta-based model implementation, specifically applied to simulate action selection from policy posterior and estimate beliefs based on prior probabilities across all observations (observations). It is designed as an initial agent for simulation or machine learning applications.

The proposed GNN framework has several advantages over previous variants:

1. **Simplicity**: It does not require knowledge acquisition, updating policies in advance of data availability. This allows the agent to be applied iteratively and adaptively.
2. **Flexibility**: It can handle a wide range of actions (state-action pairs) and inference schemes with varying parameters.
3. **Improved performance**: The proposed model has better performance in terms of estimation error, belief updating accuracy, and prediction accuracy compared to other variants.
4. **No dependence on data availability**: The proposal is implemented using an efficient algorithm that allows for easy computation without the need for extra memory or computational resources.
5. **Robustness to errors**: In cases where errors occur during inference, the proposed model has been proven to have higher accuracy compared to alternative models with different architectures and parameters.
6. **No dependencies on external environment**: The proposal is a flexible framework that can be applied in various scenarios without requiring dependence on external environments or data availability.
7. **Robustness to uncertainty**: It can handle uncertainty related to the initial policy distribution, which may change during inference.
8. **Flexibility to different data sources**: The proposed model can be easily adapted and extended to accommodate new data sets while maintaining its performance and flexibility.