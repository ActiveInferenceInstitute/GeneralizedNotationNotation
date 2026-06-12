// Stan model generated from GNN: Time-Varying Transition Dynamics Agent
// Variables: 5, Connections: 6

data {
  // No observed variables declared
}

parameters {
  matrix[3, 1] D;
  matrix[3, 1] s_t;
  matrix[3, 3] A;
  matrix[3, 1] o_t;
  matrix[2, 1] u_t;
}

model {
  // D → s_t
  s_t ~ normal(D, 1.0);
  // (s_t, u_t) → B_t
  B_t ~ normal((s_t, u_t), 1.0);
  // B_t → s_t+1
  s_t+1 ~ normal(B_t, 1.0);
  // s_t → A
  A ~ normal(s_t, 1.0);
  // A → o_t
  o_t ~ normal(A, 1.0);
  // C → o_t
  o_t ~ normal(C, 1.0);
}