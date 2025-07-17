# GNN File Structure Specification
**Version: 1.0**

## Sections

| GNNSection | SectionMeaning | ControlledTerms |
|------------|----------------|-----------------|
| ImageFromPaper | Shows the image of the graphical model, if one exists. | NA |
| GNNVersionAndFlags | Describes which specification of GNN is being used, including optional configuration flags. | Specifies GNN release version and optional flags that govern the file's interpretation. |
| ModelName | Gives a descriptive identifier for the model being expressed. | NA |
| ModelAnnotation | Plaintext caption or annotation for the model. This is a free metadata field which does not need to use any formal or controlled language. | Free-text explanation of the model's purpose and context. |
| StateSpaceBlock | Describes all variables in the model, and their state space (dimensionality). | Defines each variable using syntax like X[2,3] to specify dimensions and types. |
| Connections | Describes edges among variables in the graphical model. | Uses directed (>) and undirected (-) edges to specify dependencies between variables. |
| InitialParameterization | Provides the initial parameter values for variables. | Sets starting values for all parameters and variables in the model. |
| Equations | Describes any equations associated with the model, written in LaTeX. These equations are at least rendered for display, and further can actually specify relationships among variables. | LaTeX-rendered formulas defining model dynamics and relationships between variables. |
| Time | Describes the model's treatment of Time. | Static:Is a static model;Dynamic:Is a dynamic model;DiscreteTime=X_t:Specifies X_t as the temporal variable in a discrete time model;ContinuousTime=X_t:Specifies X_t as the temporal variable in a continuous time model;ModelTimeHorizon=X:Specifies X as the time horizon for finite-time modeling. |
| ActInfOntologyAnnotation | Connects the variables to their associated Active Inference Ontology term, for display and model juxtaposition. | Variables in this section are associated with one or more terms from the Active Inference Ontology. Format: VariableName=OntologyTerm (Example: C=Preference) |
| Footer | Closes the file and allows read-in from either end. | Marks the end of the GNN specification. |
| Signature | Cryptographic signature block (can have information regarding the completeness or provenance of the file). | Contains provenance information and optional cryptographic verification. |