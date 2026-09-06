# CatColab: A Comprehensive Summary

**Version**: 0.4 (Robin)  
**Last Updated**: 2026-01-20  
**Platform**: [catcolab.org](https://catcolab.org)

### Executive Overview

CatColab is a pioneering collaborative environment for formal, interoperable, conceptual modeling, developed by the Topos Institute and released as a user-facing application in late 2024. As of January 2026, the project has reached version 0.4 (Robin), establishing itself as a functionally mature platform that democratizes category-theoretic approaches to scientific modeling. The system represents a transformative shift in how interdisciplinary teams can construct, critique, and compose mathematical models of complex systems without requiring specialized expertise in advanced mathematics. [github](https://github.com/ToposInstitute/CatColab)

### Foundational Design Philosophy

CatColab operationalizes three fundamental principles that distinguish it from conventional modeling software:

**Formality**: Models in CatColab are mathematically rigorous objects amenable to formal critique and analysis, not merely visual diagrams or computational scripts. This formality enables systematic reasoning about model correctness, consistency, and equivalence. [epatters](https://www.epatters.org/software/)

**Interoperability**: Rather than constraining users to a single modeling language, CatColab supports multiple domain-specific logics that can be flexibly translated and composed. This architectural choice recognizes that different fields naturally employ different conceptual vocabularies—domain experts in epidemiology reason about stocks and flows, while database engineers think in terms of schemas and instances. [topos](https://topos.institute/blog/2024-10-02-introducing-catcolab/)

**Conceptual Accessibility**: Each domain-specific logic is explicitly designed to reflect the native vocabulary and intuitions of practitioners in that domain. The interface is a "structure editor" rather than either a text-based programming language (which scales poorly with complexity) or a graphical editor (which requires specialized implementation for each logic). [topos](https://topos.institute/blog/2024-10-02-introducing-catcolab/)

### Mathematical Foundation and Double Category Theory

CatColab's theoretical underpinnings rest on recent advances in double category theory, a mathematical framework that adds structure in an orthogonal dimension to classical categories. Mathematically, a **logic** in CatColab is a *double theory*, defined as a small double category within an appropriate *doctrine* (a broader organizational principle for categorical structures). [topos](https://topos.institute/blog/2024-10-02-introducing-catcolab/)

A **model** of a double theory is formally a *lax double functor to the double category of sets, functions, and spans*. This construction is remarkably powerful: it automatically endows models with categorical structure—each object in the theory corresponds not to a mere set but to an entire category, giving models themselves a categorical structure. [topos](https://topos.institute/blog/2024-10-02-introducing-catcolab/)

This mathematical abstraction translates to practical computational power. For example, in causal loop diagrams, the software can identify reinforcing and balancing feedback loops of arbitrary finite length by recognizing them as homomorphisms from abstract loop patterns—a capability that emerges naturally from the theoretical framework rather than requiring ad-hoc implementations. [topos](https://topos.institute/blog/2024-10-02-introducing-catcolab/)

**Instances**, another critical concept still under active research, are special types of bimodules that ground abstract models in concrete data. For simple double theories, categories of instances form presheaf categories; for cartesian theories, they form algebraic categories. [topos](https://topos.institute/blog/2024-10-02-introducing-catcolab/)

### Available Domain-Specific Logics

CatColab currently supports a curated library of domain-specific logics, with the count expanding with each release:

**Ologs** (Ontology Logs): Grounded in Spivak and Kent's 2011 foundational work, ologs represent classes of entities and their relationships as categorical structures. Users can model abstract knowledge domains—including, in one example, ontologies of CatColab's own logics. [github](https://github.com/ToposInstitute/CatColab)

**Schemas**: These formalize database schemas as upgraded ologs, distinguishing tables from columns. Schemas form the basis of acsets (attributed C-sets), a data structure fundamental to the AlgebraicJulia ecosystem. [arxiv](https://arxiv.org/pdf/2205.08373.pdf)

**Regulatory Networks**: Drawn from molecular biology, these model gene expression via positive (promoting) and negative (inhibitory) interactions. Mathematically, they are signed directed graphs with categorical structure. [github](https://github.com/ToposInstitute/CatColab)

**Causal Loop Diagrams**: Originating in systems dynamics, these depict qualitative feedback structures. Notably, CatColab recognizes regulatory networks and causal loop diagrams as mathematically identical at the backend—differences are purely presentational. [github](https://github.com/ToposInstitute/CatColab)

**Stock-and-Flow Diagrams**: The most sophisticated logic currently deployed, stock-and-flow diagrams model population dynamics and epidemiological processes. They encode stocks (population compartments), flows (transitions), and links (dependencies between stocks and flow rates). [arxiv](https://arxiv.org/pdf/2205.08373.pdf)

**Petri Nets**: Added in version 0.4 (Robin), Petri nets support both classical reachability analysis and stochastic mass-action dynamics, enabling chemical reaction and systems biology modeling. [topos](https://topos.institute/blog/2026-01-08-catcolab-0-4-robin/)

Additional logics are in development, including monoidal categories, multicategories, and domain-specific formalisms for programming languages. [topos](https://topos.institute/blog/2024-10-02-introducing-catcolab/)

### Technical Architecture

CatColab's implementation reflects its mathematical sophistication while prioritizing practical usability:

**Core Logic Layer**: The categorical logic library is written in Rust and compiled to WebAssembly (WASM), running entirely in the client browser. This design choice—relocating computational burden from the server to the user's machine—enables responsive interaction and supports offline-first collaboration. [topos](https://topos.institute/blog/2025-05-29-software-team/)

**Frontend**: Implemented in TypeScript with the Solid.js reactive framework, the interface uses a notebook-style presentation inspired by Jupyter and Notion. However, the conceptual underpinnings differ fundamentally: CatColab is a *structure editor* that constrains editing to syntactically valid models within the chosen logic. [topos](https://topos.institute/blog/2024-10-02-introducing-catcolab/)

**Backend**: A Rust server manages persistent storage using PostgreSQL, which stores JSON snapshots of models and analyses. Versioning infrastructure, not yet fully exposed to users, already permits tracking models through their evolution. [topos](https://topos.institute/blog/2024-10-02-introducing-catcolab/)

**Real-Time Collaboration**: Real-time concurrent editing is powered by Automerge, a CRDT (Conflict-free Replicated Data Type) library that automatically merges concurrent edits from multiple users without requiring central coordination. Users can share models via URLs containing secret hashes and collaborate synchronously or asynchronously. [youtube](https://www.youtube.com/watch?v=VJ_GeNfZXrQ)

The technology stack emphasizes modularity: Nix (37.6%), Rust (30.3%), TypeScript (23.9%), Julia (2.6%), CSS (2.0%), and XSLT (1.4%) compose the codebase. [github](https://github.com/ToposInstitute/CatColab)

### Feature Set (Current State)

CatColab's production deployment at catcolab.org provides:

- **Collaborative Notebook Interface**: Users build models through declarative cells, intermixing formal specifications with narrative rich text. [topos](https://topos.institute/work/catcolab/)
- **Domain-Specific Visualization**: Each logic has appropriate visual representations; analyses appear in side panels rather than full-page navigations. [topos](https://topos.institute/blog/2026-01-08-catcolab-0-4-robin/)
- **Motif Analysis**: For causal loop diagrams, the system can exhaustively identify feedback loops of specified types. [github](https://github.com/ToposInstitute/CatColab)
- **Simulation and Dynamics**: Stock-and-flow and Petri net models support numerical simulation with parameterized dynamics. [topos](https://topos.institute/blog/2026-01-08-catcolab-0-4-robin/)
- **Data Import/Export**: JSON serialization enables local storage and programmatic integration. [topos](https://topos.institute/blog/2025-02-05-catcolab-0-2-wren/)
- **Versioning**: Users can branch and merge models, resembling git-like version control. [topos](https://topos.institute/blog/2025-02-05-catcolab-0-2-wren/)
- **Public Profiles**: Users maintain shareable profiles listing their documents with fine-grained permissions. [topos](https://topos.institute/blog/2025-02-05-catcolab-0-2-wren/)
- **Stochastic Analysis**: V0.4 introduces the first stochastic analyses, modeling mass-action dynamics with probabilistic outcomes. [topos](https://topos.institute/blog/2026-01-08-catcolab-0-4-robin/)

### Intellectual Heritage and Ecosystem Integration

CatColab emerges from the AlgebraicJulia project, a suite of Julia packages implementing applied category theory for scientific computing. AlgebraicJulia had demonstrated theoretical power but required steep learning curves in both category theory and Julia programming. CatColab inverts this relationship: users work in intuitive domain-specific languages while the system leverages AlgebraicJulia's sophisticated mathematical machinery internally. [topos](https://topos.institute/blog/2025-05-29-software-team/)

Epidemiology represents a primary use case. John Carlos Baez and collaborators—including Xiaoyan Li and Nate Osgood, who led Canada's COVID modeling—developed compositional frameworks for stock-and-flow diagrams, work now operationalized in CatColab. This integration positions CatColab as a tool for "digital twins" and scenario analysis in public health. [math.ucr](https://math.ucr.edu/home/baez/double.pdf)

### Strategic Roadmap and Future Directions

CatColab is explicitly positioned as work-in-progress with expansive ambitions. The Topos Institute's 2025 hiring announcement indicates recruitment of full-time software engineers to accelerate development toward this vision: [topos](https://topos.institute/blog/2025-05-29-software-team/)

**Compositional Models**: Users will be able to import existing models and fuse them through declarative composition rules, enabling modular construction of complex systems from verified subcomponents. [topos](https://topos.institute/blog/2025-05-29-software-team/)

**Parallel Instance Editing**: Data instances and schemas will be editable simultaneously, offering "spreadsheet fluidity with database type safety." [topos](https://topos.institute/blog/2025-05-29-software-team/)

**Specifications and Verification**: Tools for defining external specifications and verifying that proposed designs satisfy them. [topos](https://topos.institute/blog/2025-05-29-software-team/)

**Cartesian Logics**: Supporting arrows with multiple inputs—a critical step enabling monoidal closed categories and, ultimately, programming language semantics. [topos](https://topos.institute/blog/2025-05-29-software-team/)

**Morphisms and Migrations**: Tools for translating models between logics, leveraging mathematical machinery analogous to categorical database migrations but operating at a higher level. [topos](https://topos.institute/blog/2024-10-02-introducing-catcolab/)

**User-Customizable Logics**: A "logic for logics" would permit users to define domain-specific languages without understanding underlying mathematics. [topos](https://topos.institute/blog/2024-10-02-introducing-catcolab/)

### Deployment and Community Status

**Production**: catcolab.org (tracks latest stable release)  
**Staging**: next.catcolab.org (tracks main development branch)  
**Version History**: 
- v0.1 Hummingbird (Oct 2024): Initial pre-alpha with 5 logics
- v0.2 Wren (Feb 2025): Alpha release with database features, JSON import/export, versioning
- v0.3 Finch (late 2024/early 2025): Intermediate release
- v0.4 Robin (Jan 2026): Petri nets, stochastic analysis, panel-based UI [topos](https://topos.institute/blog/2026-01-08-catcolab-0-4-robin/)

The project has achieved 79 GitHub stars and 20 forks with 15 contributors. As of January 2026, the institute explicitly signals readiness to engage domain experts from diverse disciplines, with ongoing conversations with epidemiologists, systems engineers, and computational scientists. [topos](https://topos.institute/blog/2026-01-08-catcolab-0-4-robin/)

### Implications and Broader Significance

CatColab represents a deliberate effort to operationalize abstract mathematical insights at a user-facing scale. By embedding double category theory directly into an interactive system accessible to working scientists, the project tests whether category-theoretic approaches can genuinely democratize—or whether they remain the province of specialists. [topos](https://topos.institute/blog/2024-10-02-introducing-catcolab/)

The system's architecture—where interface design is "functorial in the language"—suggests that formal mathematical structure can actually reduce implementation burden for domain-specific tools, inverting the usual assumption that rigor and accessibility trade off against each other. [topos](https://topos.institute/blog/2024-10-02-introducing-catcolab/)

For cognitive security, formal modeling systems like CatColab have implications for threat modeling, attack tree analysis, and assumption validation in adversarial settings. The interoperability emphasis enables threat models in one logic to be translated and composed with system architecture models in another, providing novel opportunities for integrated risk assessment. [topos](https://topos.institute/work/catcolab/)

***

### References

 GitHub - ToposInstitute/CatColab repository information and release notes [github](https://github.com/ToposInstitute/CatColab)
 Topos Institute, "Introducing CatColab," October 2024 [topos](https://topos.institute/blog/2024-10-02-introducing-catcolab/)
 Topos Institute, "CatColab - Work Overview," project documentation [epatters](https://www.epatters.org/software/)
 Topos Institute, "CatColab - Concepts and Fundamentals," help documentation [topos](https://topos.institute/work/catcolab/)
 Topos Institute, "Growing the Topos tech team," May 2025 [topos](https://topos.institute/blog/2025-05-29-software-team/)
 Topos Institute, "CatColab v0.4: Robin," January 2026 [topos](https://topos.institute/blog/2026-01-08-catcolab-0-4-robin/)
 Topos Institute, "CatColab 0.2: Wren," February 2025 [topos](https://topos.institute/blog/2025-02-05-catcolab-0-2-wren/)
 Automerge collaborative editing framework documentation [youtube](https://www.youtube.com/watch?v=VJ_GeNfZXrQ)
 Compositional modeling with stock-and-flow diagrams, arXiv preprint [arxiv](https://arxiv.org/pdf/2205.08373.pdf)
 Baez & collaborators, "Double categories of open systems," mathematical foundations [math.ucr](https://math.ucr.edu/home/baez/double.pdf)