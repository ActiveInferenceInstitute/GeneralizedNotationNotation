# EXTRACT_PARAMETERS

Based on the document, here are the key parameters and their corresponding summaries:

**Model Matrices:**

1. **A matrix representing the initial state space dimensions**: `{
  "state_observation": {
    "observations": [
      {"x": 0},
      {"x": 2}],
    ...
  },
  "actions" : {
    "states": [
      {"x": 3},
      {"x": 1}],
    ...
  }
}`

2. **B matrix representing the initial policy prior**: `{
   "policy_prior": {
     "state_observation": {
       "observations": [
         {"x": 0},
         {"x": 2}],
     ...
   },
   "actions" : {
    "states": [
      {"x": 3},
      {"x": 1}],
   }
}`

3. **C matrix representing the initial habit prior**: `{
   "habit_prior": {
     "state_observation": {
       "observations": [
         {"x": 0},
         {"x": 2}],
     ...
   },
   "actions" : {
    "states": [
      {"x": 3},
      {"x": 1}],
   }
}`

4. **D matrix representing the initial policy prior**: `{
   "policy_prior": {
     "state_observation": {
       "observations": [
         {"x": 0},
         {"x": 2}],
     ...
   },
   "actions" : {
    "states": [
      {"x": 3},
      {"x": 1}],
   }
}`

5. **E matrix representing the initial action prior**: `{
   "action_prior": {
     "state_observation": {
       "observations": [
         {"x": 0},
         {"x": 2}],
     ...
   },
   "actions" : {
    "states": [
      {"x": 3},
      {"x": 1}],
   }
}`

6. **F matrix representing the initial policy prior**: `{
   "policy_prior": {
     "state_observation": {
       "observations": [
         {"x": 0},
         {"x": 2}],
     ...
   },
   "actions" : {
    "states": [
      {"x": 3},
      {"