# 🧬 GNN Ontological Annotations Report

## 📊 Summary of Ontology Processing

- **Files Processed:** 4
- **Total Annotations Found:** 88
- **Validations Passed:** 36
- **Validations Failed:** 52
🗓️ Report Generated: 2025-06-20 15:31:34
🎯 GNN Source Directory: `gnn/examples`
📖 Ontology Terms Definition: `src/ontology/act_inf_ontology_terms.json` (Loaded: 60 terms)

---

### Ontological Annotations for `src/gnn/examples/rxinfer_multiagent_gnn.md`
#### Mappings:
- `dt` -> `TimeStep`
- `gamma` -> `ConstraintParameter`
- `nr_steps` -> `TrajectoryLength`
- `nr_iterations` -> `InferenceIterations`
- `nr_agents` -> `NumberOfAgents`
- `softmin_temperature` -> `SoftminTemperature`
- `A` -> `StateTransitionMatrix`
- `B` -> `ControlInputMatrix`
- `C` -> `ObservationMatrix`
- `initial_state_variance` -> `InitialStateVariance`
- `control_variance` -> `ControlVariance`
- `goal_constraint_variance` -> `GoalConstraintVariance`

**Validation Summary**: All ontological terms are recognized.

---

### Ontological Annotations for `src/gnn/examples/self_driving_car_comprehensive.md`
#### Mappings:
- `A_camera_front` -> `LikelihoodMatrixCameraFront` (**INVALID TERM**)
- `A_lidar_objects` -> `LikelihoodMatrixLiDAR` (**INVALID TERM**)
- `A_radar_objects` -> `LikelihoodMatrixRadar` (**INVALID TERM**)
- `A_gps` -> `LikelihoodMatrixGPS` (**INVALID TERM**)
- `A_imu` -> `LikelihoodMatrixIMU` (**INVALID TERM**)
- `B_vehicle_dynamics` -> `TransitionMatrixVehicleDynamics` (**INVALID TERM**)
- `B_traffic` -> `TransitionMatrixTrafficConditions` (**INVALID TERM**)
- `B_other_vehicles` -> `TransitionMatrixOtherVehicles` (**INVALID TERM**)
- `C_collision_avoidance` -> `PreferenceCollisionAvoidance` (**INVALID TERM**)
- `C_lane_keeping` -> `PreferenceLaneKeeping` (**INVALID TERM**)
- `C_fuel_efficiency` -> `PreferenceFuelEfficiency` (**INVALID TERM**)
- `D_vehicle_position` -> `PriorVehiclePosition` (**INVALID TERM**)
- `D_weather` -> `PriorWeatherConditions` (**INVALID TERM**)
- `vehicle_position_x` -> `VehicleGlobalPositionX` (**INVALID TERM**)
- `vehicle_position_y` -> `VehicleGlobalPositionY` (**INVALID TERM**)
- `vehicle_heading` -> `VehicleHeading` (**INVALID TERM**)
- `vehicle_velocity_x` -> `VehicleLongitudinalVelocity` (**INVALID TERM**)
- `vehicle_velocity_y` -> `VehicleLateralVelocity` (**INVALID TERM**)
- `steering_angle` -> `SteeringWheelAngle` (**INVALID TERM**)
- `throttle_position` -> `ThrottlePedalPosition` (**INVALID TERM**)
- `brake_pressure` -> `BrakePedalPressure` (**INVALID TERM**)
- `traffic_density` -> `TrafficDensity` (**INVALID TERM**)
- `weather_condition` -> `WeatherCondition` (**INVALID TERM**)
- `road_surface_condition` -> `RoadSurfaceCondition` (**INVALID TERM**)
- `visibility_range` -> `VisibilityRange` (**INVALID TERM**)
- `other_vehicles_positions` -> `OtherVehiclePositions` (**INVALID TERM**)
- `pedestrians_positions` -> `PedestrianPositions` (**INVALID TERM**)
- `traffic_lights_states` -> `TrafficLightStates` (**INVALID TERM**)
- `action_steering` -> `SteeringAction` (**INVALID TERM**)
- `action_acceleration` -> `AccelerationAction` (**INVALID TERM**)
- `action_braking` -> `BrakingAction` (**INVALID TERM**)
- `behavior_mode` -> `DrivingBehaviorMode` (**INVALID TERM**)
- `maneuver_type` -> `ManeuverType` (**INVALID TERM**)
- `sensor_health_camera` -> `CameraHealthState` (**INVALID TERM**)
- `system_health_engine` -> `EngineHealthState` (**INVALID TERM**)
- `collision_risk_assessment` -> `CollisionRiskAssessment` (**INVALID TERM**)
- `emergency_intervention` -> `EmergencyInterventionSystem` (**INVALID TERM**)
- `gamma_camera` -> `CameraMeasurementPrecision` (**INVALID TERM**)
- `gamma_lidar` -> `LiDARMeasurementPrecision` (**INVALID TERM**)
- `alpha_control` -> `ControlPolicyPrecision` (**INVALID TERM**)
- `alpha_behavior` -> `BehavioralPolicyPrecision` (**INVALID TERM**)

**Validation Summary**: 41 unrecognized ontological term(s) found.

---

### Ontological Annotations for `src/gnn/examples/rxinfer_hidden_markov_model.md`
#### Mappings:
- `A` -> `StateTransitionMatrix`
- `B` -> `ObservationMatrix`
- `A_prior` -> `TransitionMatrixPrior` (**INVALID TERM**)
- `B_prior` -> `ObservationMatrixPrior` (**INVALID TERM**)
- `s_0` -> `InitialStateDistribution` (**INVALID TERM**)
- `s` -> `HiddenStateSequence` (**INVALID TERM**)
- `x` -> `ObservationSequence` (**INVALID TERM**)
- `q_A` -> `PosteriorTransitionMatrix` (**INVALID TERM**)
- `q_B` -> `PosteriorObservationMatrix` (**INVALID TERM**)
- `q_s` -> `PosteriorHiddenStates` (**INVALID TERM**)
- `free_energy` -> `VariationalFreeEnergy`
- `T` -> `TimeHorizon` (**INVALID TERM**)
- `n_states` -> `NumberOfHiddenStates` (**INVALID TERM**)
- `n_obs` -> `NumberOfObservationCategories` (**INVALID TERM**)
- `n_iterations` -> `InferenceIterations`

**Validation Summary**: 11 unrecognized ontological term(s) found.

---



### Ontological Annotations for `src/gnn/examples/pymdp_pomdp_agent.md`
#### Mappings:
- `A_m0` -> `LikelihoodMatrixModality0`
- `A_m1` -> `LikelihoodMatrixModality1`
- `A_m2` -> `LikelihoodMatrixModality2`
- `B_f0` -> `TransitionMatrixFactor0`
- `B_f1` -> `TransitionMatrixFactor1`
- `C_m0` -> `LogPreferenceVectorModality0`
- `C_m1` -> `LogPreferenceVectorModality1`
- `C_m2` -> `LogPreferenceVectorModality2`
- `D_f0` -> `PriorOverHiddenStatesFactor0`
- `D_f1` -> `PriorOverHiddenStatesFactor1`
- `s_f0` -> `HiddenStateFactor0`
- `s_f1` -> `HiddenStateFactor1`
- `s_prime_f0` -> `NextHiddenStateFactor0`
- `s_prime_f1` -> `NextHiddenStateFactor1`
- `o_m0` -> `ObservationModality0`
- `o_m1` -> `ObservationModality1`
- `o_m2` -> `ObservationModality2`
- `π_f1` -> `PolicyVectorFactor1`
- `u_f1` -> `ActionFactor1`
- `G` -> `ExpectedFreeEnergy`

**Validation Summary**: All ontological terms are recognized.

---
