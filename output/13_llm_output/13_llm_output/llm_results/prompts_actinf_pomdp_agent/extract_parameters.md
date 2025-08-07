# EXTRACT_PARAMETERS

Here are the structured components for GNNPOMDP agent and examples of parameter breakdowns:

**GNNModelStructureWithParametrsExample2**
```python
class ActiveInferenceContext(Graph):
    def initialize_agent(self, parameters=None):
        """
            Initialize a new instance of the Active Inference POMDP agent with default parameter setups.

        Args:
          parameters (dict): A dictionary representing the initial parameters for the agent.
           Default is an empty dictionary that defaults to `{}` (empty). The function will initialize all parameters in the list at initialization and then assign each parameter from this list to the same value, which should provide a consistent set of values across all instances:
              * For example, if `parameters["x_next"] = {"y": [1.05367842]})
            Then "z_previous" would be returned at time t=1 and "g_prev" is returned at time t=t+1 as well (depending on the type of the current parameter)

            If we initialize with `parameters["x"] = {"y": [0.56739]}, then
            "z_previous" would be returned when updating in 2 steps
                * Current state is now a tuple [[(0, 1), (-1)]], so
                "g_prev" will return the value of current parameter x

                **Initialization:**
                    Current parameters = {"x": {'y': [1.56739]}}

                    **Computation**:
                       * Initialized state = [(0+0):[-(2), (8), ([1])]
                     * Computed parameter = 0.56739*[],[]-([0])...
                
    Args:
      parameters (dict): A dictionary representing the initial parameters for the agent. Default is an empty dictionary that defaults to {} (empty).
          This value will be used as input and initialized in each of the initialization steps, regardless of type parameter values.
        """

        self._init_agents(parameters)

    def _init_agents(self):
        # Initialization process with parameters for example positions
        x = {"x":[("a", [0]), ("b", [])}
        
        # Example position
        y = {("z"): []}
        
    
        # Iteration 1: initialize initial values of the agent
       self.agent1_position(parameters)

    def _init_agents(self):
        # Initialization process with parameter for action choice and prior weights 
        x = {"x":[["a", "b"], ["c", [])}]
        
        y = {("z"): []}
        
    
        # Iteration 2: initialize initial values of the agent
       self.agent1_position(parameters)

        return

    def _init_agents(self):
        # Initialization process with parameter for action choice and prior weights 
        x = {"x":[["a", "b"], ["c", [])}]
        
        y = {("z"): []}
        
    
        self._action_choice()
    
    def act_forward(self, **kwargs):
        """
            Act forward the GNNPOMDP agent's actions given a single observation. The
                action is initialized with values of parameter `x` and `y`.

            Args:
          **kwargs (dict): Dictionary containing arguments for updating parameters to be returned in 
            a different order than when you initialize the parameters using "**".
              * For example, if there are two types of actions, like 'a' and
               'b', then
                - In case there's only one action type.
                   - A tuple will have an integer (type) for each parameter
                  - The following is not a valid sequence in case we update parametrs with "**",
                  but instead, return the next values from the tuple
              * If no value of 'x' or 'y' has been initialized yet and you want to initialize one
                then use the default initialization.
               - In this case
            """

        for (x_current,), y_: 
            if not isinstance(kwargs["x"], type([])) == type([]):  # Check if input is an instance of "dict", "list" or
           "tuple".iteritemspaces()[0]:
              x = kwargs[kwargs]
              
          else:  
            try:,**{'x':{("y":")})   
    def _action_choice(self):
        """
         Return the action choice based on the current parameter's value. The
           actions are initialized with values of "x" and 0-based indexing
             * For example, in case we initialize ``a`` to None
            ** This is not a valid sequence in case there are only one
          type(kwargs)
                # Example: (A*("b"), {})
                
            """
        x = kwargs.get('x', [])
        
        # Iterate through the values of "x" and compute 
        if isinstance(x, dict):  # Check if input is an instance of "dict", "list" or
           "tuple".iteritemspaces()[0]:
              x_current_, y_:
                for key in (
                    ("x":{"a":[("i"), ["l",""],([1])}),
                 ("y"):
                        {"name:", str(key)}
                        
                  } else:  
                  
        try:,**{'f':{},""")   
    def _action_choice(self):
        """
         Return the action choice based on the current parameter's value. The
           actions are initialized with values of "x" and 0-based indexing
             * For example, in case we initialize ``a`` to None
            ** This is not a valid sequence in case there are only one
          type(kwargs)
                # Example: (A*("b"), {})
                
            """
        x = kwargs.get('x', [])
        
        if isinstance(x[0], dict): 
            return {}

        for key, value in x.items():  
            
              if isinstance(value["f"], type([])) == type([]):
                  return value
```