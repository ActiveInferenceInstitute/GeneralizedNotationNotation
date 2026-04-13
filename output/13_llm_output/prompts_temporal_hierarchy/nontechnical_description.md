# Nontechnical Description

Prompt:

> Describe this GNN model comprehensively, in non-technical language suitable for a broad audience.

Response:

You can use the following code to generate a GNN representation of your model:

1. Create an instance of the `GNNModel` class with the specified parameters and annotations, like this:
   ```python
model = GNNModel(
    num_levels=3,
    num_states_l0=4,
    num_actions_l0=3,
    num_states_l1=4,
    num_obs_l0=2,
    num_actions_l0=2,
    timescale_ratio_1_0="fast",
    timescale_ratio_2_1="slow"
)


model.encode(
    data=[
        (
            [
                (
                    (
                        (
                            (
                                A0[3, 4, type=float], "A"), "A")
                            # Action level is fast and reflexive
                            # This action corresponds to the first level of hierarchy
                            A0_action = A0[1:2]

                            # The next level has a higher probability than A0
                            A0_next = A0[3:4]
                        )
                    ),
                (
                    (
                        (
                            (
                                B0[4, 4, type=float], "B")
                            # Action level is medium and reflexive
                            # This action corresponds to the second level of hierarchy
                            B0_action = B0[1:2]

                            # The next level has a higher probability than B0
                            B0_next = B0[3:4]
                        )
                    ),
                (
                    (
                        (
                            C0[3, 4, type=float], "C")
                            # Action level is fast and reflexive
                            # This action corresponds to the third level of hierarchy
                            C0_action = C0[1:2]

                            # The next level has a higher probability than C0
                            C0_next = C0[3:4]
                        )
                    ),
                (
                    (
                        (
                            D0[4, 4, type=float], "D")
                            # Action level is medium and reflexive
                            # This action corresponds to the fourth level of hierarchy
                            D0_action = D0[1:2]

                            # The next level has a higher probability than D0
                            D0_next = D0[3:4]
                        )
                    ),
                (
                    (
                        (
                            A1[3, 4, type=float],