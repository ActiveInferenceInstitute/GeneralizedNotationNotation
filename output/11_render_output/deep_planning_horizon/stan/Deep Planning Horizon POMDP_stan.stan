// Stan model generated from GNN: Deep Planning Horizon POMDP
// Variables: 14, Connections: 33

data {
  matrix[4, 1] o;
  vector[1] u;
  vector[1] t;
}

parameters {
  vector[4] D;
  matrix[4, 1] s;
  matrix[4, 1] s_tau1;
  matrix[4, 1] s_tau2;
  matrix[4, 1] s_tau3;
  matrix[4, 1] s_tau4;
  matrix[4, 1] s_tau5;
  real F;
  vector[4] C;
  array[4] matrix[4, 4] B;
  vector[64] π;
}

model {
  // D → s
  s ~ normal(D, 1.0);
  // s → A
  A ~ normal(s, 1.0);
  // A → o
  o ~ normal(A, 1.0);
  // s → F
  F ~ normal(s, 1.0);
  // o → F
  F ~ normal(o, 1.0);
  // E → π
  π ~ normal(E, 1.0);
  // G → π
  π ~ normal(G, 1.0);
  // s → s_tau1
  s_tau1 ~ normal(s, 1.0);
  // B → s_tau1
  s_tau1 ~ normal(B, 1.0);
  // s_tau1 → s_tau2
  s_tau2 ~ normal(s_tau1, 1.0);
  // B → s_tau2
  s_tau2 ~ normal(B, 1.0);
  // s_tau2 → s_tau3
  s_tau3 ~ normal(s_tau2, 1.0);
  // B → s_tau3
  s_tau3 ~ normal(B, 1.0);
  // s_tau3 → s_tau4
  s_tau4 ~ normal(s_tau3, 1.0);
  // B → s_tau4
  s_tau4 ~ normal(B, 1.0);
  // s_tau4 → s_tau5
  s_tau5 ~ normal(s_tau4, 1.0);
  // A → s_tau1
  s_tau1 ~ normal(A, 1.0);
  // A → s_tau2
  s_tau2 ~ normal(A, 1.0);
  // A → s_tau3
  s_tau3 ~ normal(A, 1.0);
  // A → s_tau4
  s_tau4 ~ normal(A, 1.0);
  // A → s_tau5
  s_tau5 ~ normal(A, 1.0);
  // C → G_tau1
  G_tau1 ~ normal(C, 1.0);
  // C → G_tau2
  G_tau2 ~ normal(C, 1.0);
  // C → G_tau3
  G_tau3 ~ normal(C, 1.0);
  // C → G_tau4
  G_tau4 ~ normal(C, 1.0);
  // C → G_tau5
  G_tau5 ~ normal(C, 1.0);
  // G_tau1 → G
  G ~ normal(G_tau1, 1.0);
  // G_tau2 → G
  G ~ normal(G_tau2, 1.0);
  // G_tau3 → G
  G ~ normal(G_tau3, 1.0);
  // G_tau4 → G
  G ~ normal(G_tau4, 1.0);
  // G_tau5 → G
  G ~ normal(G_tau5, 1.0);
  // G → π
  π ~ normal(G, 1.0);
  // π → u
  u ~ normal(π, 1.0);
}