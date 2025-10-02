# Technical Description

Prompt:

> Describe this GNN model comprehensively, in technical detail.

Response:

Your response contains a header section which allows you to specify that your response should be displayed in JSON format using the `json` keyword argument of your API request:
```python
response = json.dumps(
    headers=header_info,
    body=[{
      "type": "string",
      "name": "ActiveInference Neural Response Model v1",
      """
        Action             Action name                 Default action value 
        ----------                     ----------   
          LAT                                                  
          LALT                                                                      "LAT" 
          LALA         LAST               Last update time in milliseconds
          LALT        LESTime                Least recently observed last timestep 
          LAMF           ALTER   Altered state (activation) change in the first data point
              |                     |            
  """
    },
      {
        "type": "string", 
        "name": "GNN Example Output",
        """
         Response            Action         Input    Output                 State      Action     Actions       
          LAT                LALA              LAST          The least recently observed last timestep is at time t=0.2004981756394072
           LALT                                                                  LASTI          0.100000000   |   
          LALT        LESTime                Last update time in milliseconds 0.2004981756394072
             |                     |            
       M_alloc              M_ALLOC            M_Allocation         M_Analysis          
         F          Fitting          Firing   Fitter                 Fitting         
          C_mod                C_mod                                  C_Mod              
          Ca1                 Ca_Ca                                                                    "C_Mod"
           Ca2                 Ca_Cal                                                                    "FiringRateObservation"
        P            P            P            P           P           P            
      """
    }
)
```