# GNN Multi-Agent Simulation Specification

## 1. Introduction

This document outlines the specification for defining, simulating, and analyzing multi-agent systems (MAS) within the Generalized Notation Notation (GNN) framework. It extends the core GNN capabilities to support interactions and communications between multiple autonomous agents, each potentially described by its own GNN model or as part of a larger, integrated GNN structure.

The goal is to enable the modeling of complex systems where multiple decision-making entities coexist, perceive, act, and communicate, allowing for the study of emergent collective behaviors, distributed problem-solving, and inter-agent dynamics.

## 2. Core Concepts

### 2.1. Agent Definition
An **Agent** in a GNN MAS is an autonomous entity with its own internal states, sensory inputs, action outputs, and decision-making logic.
-   **Homogeneous Agents:** Multiple instances of the same agent type, defined by a single GNN model template.
-   **Heterogeneous Agents:** Different types of agents, potentially defined by distinct GNN models or variations of a base model.

Each agent typically encapsulates its own GNN components: `StateSpaceBlock`, `ObservationSpaceBlock`, `ActionSpaceBlock`, `TransitionFunctionBlock`, `ObservationFunctionBlock`, `PolicyBlock`, etc.

### 2.2. Multi-Agent System (MAS)
A **Multi-Agent System** is a collection of interacting agents. The GNN specification for a MAS will define:
-   The types and number of agents.
-   The communication infrastructure.
-   The shared environment, if any.
-   Global parameters or dynamics affecting all agents.

## 3. GNN File Structure for Multi-Agent Systems

To accommodate multi-agent systems, the GNN file structure may be extended with new blocks or by adapting existing ones.

### 3.1. `AgentsBlock` (New Block)

This block would be used to declare and configure the agents within the MAS.

```gnn
AgentsBlock:
  AgentTypes:
    - AgentModelID: MyAgentTypeA
      GNNFile: "path/to/agent_A_model.gnn" # Optional: if defined in a separate file
      Count: 5 # Number of agents of this type
      InitialStates: # Optional: specific initial states for instances
        - InstanceID: AgentA_1
          InitialState: {s1: 0, s2: "ready"}
        - InstanceID: AgentA_2
          InitialState: {s1: 1, s2: "waiting"}
      Parameters: # Optional: specific parameters for instances
        - InstanceID: AgentA_1
          Param: {p1: 0.5}
    - AgentModelID: MyAgentTypeB
      GNNModel: # Inline definition for simpler agents or types
        StateSpaceBlock:
          States:
            internal_status: {Type: string, Values: ["active", "inactive"]}
        # ... other GNN blocks for AgentTypeB
      Count: 1
      InstanceID: TheOnlyB # If count is 1, a single InstanceID can be given

  # Global agent parameters or default initializations can also be specified here
EndAgentsBlock
```

**Fields:**
-   `AgentTypes`: A list defining different types of agents.
    -   `AgentModelID`: A unique identifier for this agent type/model.
    -   `GNNFile`: Path to an external GNN file defining this agent type.
    -   `GNNModel`: An inline GNN definition for this agent type. One of `GNNFile` or `GNNModel` must be provided.
    -   `Count`: Number of instances of this agent type. Defaults to 1.
    -   `InstanceIDPrefix`: (Optional) If specified, instances will be named `InstanceIDPrefix_1`, `InstanceIDPrefix_2`, etc.
    -   `InitialStates`: (Optional) A list to provide specific initial states for individual agent instances.
        -   `InstanceID`: The specific ID of the agent instance.
        -   `InitialState`: A dictionary mapping state variable names to their initial values.
    -   `Parameters`: (Optional) A list to provide specific parameters for individual agent instances.

### 3.2. Agent Referencing

Within other GNN blocks (especially `ConnectionsBlock`), agents will be referenced using their unique `InstanceID`. If not explicitly provided and `Count > 1`, IDs could be auto-generated (e.g., `MyAgentTypeA_1`, `MyAgentTypeA_2`).

Example: `MyAgentTypeA_1.sensor_output` or `TheOnlyB.action_command`.

## 4. Communication Mechanisms

Communication enables agents to exchange information, coordinate actions, and influence each other. GNN will support several communication patterns.

### 4.1. `CommunicationBlock` (New Block)

This block defines the communication channels and their properties.

```gnn
CommunicationBlock:
  Channels:
    - ChannelID: EnvBroadcastChannel
      Type: Broadcast
      MessageType: EnvironmentData # References a type defined in DataSchemaBlock or implicitly
      Capacity: 10 # Optional: buffer size

    - ChannelID: AgentAtoBDirect
      Type: PointToPoint
      SourceAgent: MyAgentTypeA_1 # InstanceID
      TargetAgent: TheOnlyB        # InstanceID
      MessageType: ControlSignal
      Latency: "1step" # Optional: communication delay

    - ChannelID: SharedBeliefSpace
      Type: SharedMemory
      Schema: BeliefStateSchema # Defines the structure of the shared memory region
      AccessControl: # Optional
        - Agent: MyAgentTypeA_* # Wildcard for all AgentA instances
          Permissions: [Read, Write]
        - Agent: TheOnlyB
          Permissions: [Read]

  MessageSchemas: # Optional: Define structure for messages if not simple types
    - SchemaID: EnvironmentData
      Fields:
        temperature: {Type: float}
        light_level: {Type: integer, Range: [0, 1023]}
    - SchemaID: ControlSignal
      Fields:
        command: {Type: string, Values: ["START", "STOP", "RESET"]}
        priority: {Type: integer, Default: 0}
EndCommunicationBlock
```

**Fields:**
-   `Channels`: A list of communication channels.
    -   `ChannelID`: A unique identifier for the channel.
    -   `Type`: The type of communication channel.
        -   `Broadcast`: One-to-many. Messages sent by any connected agent are received by all other connected agents.
        -   `PointToPoint`: Direct one-to-one link between two specified agents.
        -   `SharedMemory`: A common data structure that multiple agents can read from and/or write to.
        -   `PublishSubscribe`: Agents can publish messages to topics, and other agents can subscribe to topics. (More complex, could be a future extension or a specialization of Broadcast).
    -   `MessageType`/`Schema`: Defines the structure/type of data transmitted over the channel. Can reference a `SchemaID` defined in `MessageSchemas` or be a basic GNN type.
    -   `Capacity`: (For buffered channels) Maximum number of messages the channel can hold.
    -   `SourceAgent`/`TargetAgent`: (For `PointToPoint`) Specifies the sender and receiver `InstanceID`.
    -   `AccessControl`: (For `SharedMemory`) Defines read/write permissions for agents.
    -   `Latency`: (Optional) Delay in message transmission, e.g., "0step", "1step", "100ms".

### 4.2. Connecting Agents to Channels (within `ConnectionsBlock`)

The existing `ConnectionsBlock` will be used to specify how agent outputs are sent to channels and how channel outputs are received as agent inputs.

```gnn
ConnectionsBlock:
  Connections:
    # Agent publishing to a broadcast channel
    - From: MyAgentTypeA_1.belief_state_output
      To: EnvBroadcastChannel.input # Channels have implicit 'input'/'output' ports

    # Agent subscribing to a broadcast channel (all agents connected to it receive)
    # Input mapping for agent from channel
    - From: EnvBroadcastChannel.output
      To: MyAgentTypeA_1.environmental_cue_input
    - From: EnvBroadcastChannel.output
      To: MyAgentTypeA_2.environmental_cue_input
    - From: EnvBroadcastChannel.output
      To: TheOnlyB.environmental_cue_input


    # Point-to-point communication (already defined by Source/Target in CommunicationBlock)
    # but ConnectionsBlock can map specific variables
    - From: MyAgentTypeA_1.control_signal_output
      To: AgentAtoBDirect.input
    - From: AgentAtoBDirect.output
      To: TheOnlyB.command_input


    # Shared memory access
    # Agent writing to shared memory
    - From: MyAgentTypeA_1.updated_belief
      To: SharedBeliefSpace.MyAgentTypeA_1_section # Path within shared memory schema

    # Agent reading from shared memory
    - From: SharedBeliefSpace.MyAgentTypeA_1_section
      To: TheOnlyB.observed_A1_belief

EndConnectionsBlock
```

**Conventions:**
-   Channels can be treated as nodes in the connection graph, with implicit `input` and `output` ports (or more specific ports depending on channel type).
-   For `SharedMemory`, paths within the shared memory schema might be used for fine-grained connections.

## 5. Simulation Semantics

### 5.1. Execution Model
The simulation loop for a GNN MAS will need to define the order of operations:
1.  **Perception Phase:** Agents read from their sensors and incoming communication channels.
2.  **Cognition/Decision Phase:** Agents update their internal states and decide on actions based on their policies (e.g., `TransitionFunctionBlock`, `PolicyBlock`).
3.  **Action Phase:** Agents perform actions, which might include sending messages to communication channels or interacting with a shared environment.
4.  **Communication Propagation:** Messages are propagated through channels according to their type and latency.
5.  **Environment Update:** (If a shared environment is modeled) The environment state updates based on agent actions and its own dynamics.

### 5.2. Synchronization
-   **Synchronous Execution:** All agents complete a phase before the next phase begins for any agent (default).
-   **Asynchronous Execution:** (Advanced) Agents may operate on different clocks or with event-driven updates. This would require more detailed specification if supported.

### 5.3. Shared Environment
If a shared environment exists, it can be modeled as a special GNN component or a set of global variables that agents can perceive and influence.
-   `EnvironmentBlock`: (Potential New Block) To define the state, dynamics, and agent interaction points of a shared environment.
    ```gnn
    EnvironmentBlock:
      StateSpace:
        temperature: {Type: float, Initial: 25.0}
        resources_available: {Type: integer, Initial: 100}
      DynamicsFunction: # How environment changes autonomously or due to agent actions
        # ...
      AgentInterfaces: # How agent actions affect env and how env affects agent observations
        # ...
    EndEnvironmentBlock
    ```

## 6. Ontology Integration (`ActInfOntologyAnnotation`)

The `ActInfOntologyAnnotation` block can be used to map MAS components and interactions to terms from the Active Inference Ontology or other relevant ontologies.

```gnn
ActInfOntologyAnnotation:
  Annotations:
    - ElementID: MyAgentTypeA_1
      Term: "bioio:Agent"
    - ElementID: EnvBroadcastChannel
      Term: "iao:InformationContentEntity" # Representing the medium or content
      Aspect: "CommunicationChannel"
    - ElementID: MyAgentTypeA_1.belief_state_output # Connecting to a channel
      Term: "iao:SendMessageInformationAction"
    - ElementID: TheOnlyB.command_input # Receiving from a channel
      Term: "iao:ReceiveMessageInformationAction"
    - ElementID: SharedBeliefSpace
      Term: "iao:SharedKnowledgeBase"
    - Connection: MyAgentTypeA_1.control_signal_output -> AgentAtoBDirect.input
      Term: "ro:transmits" # Relation Ontology
EndActInfOntologyAnnotation
```

This helps in standardizing the interpretation of the MAS model and facilitating interoperability or analysis with ontology-aware tools.

## 7. Example Snippet

Consider a simple scenario with two "Greeter" agents. Agent1 says "Hello" and Agent2 responds with "Hi".

```gnn
GNNModelID: TwoGreetersMAS
Version: 1.0

ModelName: Two Greeter Agents Communication

AgentsBlock:
  AgentTypes:
    - AgentModelID: GreeterAgent
      GNNModel:
        StateSpaceBlock:
          States:
            message_to_send: {Type: string, Initial: ""}
            received_message: {Type: string, Initial: ""}
            has_greeted: {Type: boolean, Initial: false}
        ObservationSpaceBlock:
          Observations:
            incoming_message_obs: {Type: string} # From channel
        ActionSpaceBlock:
          Actions:
            send_message_action: {Type: string} # To channel
            internal_update_action: {} # No explicit output, just state change
        PolicyBlock: # Simplified policy
          Rules:
            - If: {received_message: "Hello", has_greeted: false}
              Then: {message_to_send: "Hi", has_greeted: true, Action: send_message_action}
              # Agent 2 logic
            - If: {received_message: "", has_greeted: false, InstanceIDMatches: "Greeter_1"} # Agent 1 initiates
              Then: {message_to_send: "Hello", has_greeted: true, Action: send_message_action}
              # Agent 1 logic
        FunctionMappings: # Simplified: map chosen message directly to action/state
          - Action: send_message_action
            Output: message_to_send
          - Observation: incoming_message_obs
            Input: received_message # state updated by observation

      Count: 2
      InstanceIDPrefix: "Greeter" # Results in Greeter_1, Greeter_2
EndAgentsBlock

CommunicationBlock:
  Channels:
    - ChannelID: GreetingChannel
      Type: Broadcast # Both can send, both can receive
      MessageType: string
EndCommunicationBlock

ConnectionsBlock:
  Connections:
    # Greeter_1 sending
    - From: Greeter_1.send_message_action
      To: GreetingChannel.input
    # Greeter_2 sending
    - From: Greeter_2.send_message_action
      To: GreetingChannel.input

    # Greeter_1 receiving
    - From: GreetingChannel.output
      To: Greeter_1.incoming_message_obs
    # Greeter_2 receiving
    - From: GreetingChannel.output
      To: Greeter_2.incoming_message_obs
EndConnectionsBlock

```
*(Note: The example policy is highly simplified for brevity and would typically involve more structured `TransitionFunctionBlock` and `ObservationFunctionBlock`.)*

## 8. Future Considerations
-   **Dynamic Agent Populations:** Agents being created or destroyed during simulation.
-   **Complex Network Topologies:** Defining arbitrary graph-based communication networks beyond simple channel types.
-   **Resource Management:** Agents competing for or sharing limited resources.
-   **Standardized Agent API:** For easier integration of pre-built agent models.
-   **Advanced Asynchronous Communication Models.**

This specification provides a foundational framework for multi-agent modeling in GNN, intended to be extensible for more complex scenarios.