// Stan model generated from GNN: T-Maze Epistemic Foraging Agent
// Variables: 10, Connections: 19

data {
  vector[1] t;
  vector[1] u;
}

parameters {
  matrix[4, 1] s_loc;
  matrix[2, 1] s_ctx;
  matrix[4, 1] o_loc;
  matrix[3, 1] o_rew;
  vector[3] C_rew;
  array[4] matrix[4, 4] B_loc;
  vector[4] pi;
  real G;
}

model {
  // D_loc → s_loc
  s_loc ~ normal(D_loc, 1.0);
  // D_ctx → s_ctx
  s_ctx ~ normal(D_ctx, 1.0);
  // s_loc → A_loc
  A_loc ~ normal(s_loc, 1.0);
  // A_loc → o_loc
  o_loc ~ normal(A_loc, 1.0);
  // s_loc → A_rew
  A_rew ~ normal(s_loc, 1.0);
  // s_ctx → A_rew
  A_rew ~ normal(s_ctx, 1.0);
  // A_rew → o_rew
  o_rew ~ normal(A_rew, 1.0);
  // s_loc → B_loc
  B_loc ~ normal(s_loc, 1.0);
  // s_ctx → B_ctx
  B_ctx ~ normal(s_ctx, 1.0);
  // C_rew → G_ins
  G_ins ~ normal(C_rew, 1.0);
  // G_epi → G
  G ~ normal(G_epi, 1.0);
  // G_ins → G
  G ~ normal(G_ins, 1.0);
  // G → pi
  pi ~ normal(G, 1.0);
  // pi → u
  u ~ normal(pi, 1.0);
  // B_loc → u
  u ~ normal(B_loc, 1.0);
  // s_loc → F
  F ~ normal(s_loc, 1.0);
  // s_ctx → F
  F ~ normal(s_ctx, 1.0);
  // o_loc → F
  F ~ normal(o_loc, 1.0);
  // o_rew → F
  F ~ normal(o_rew, 1.0);
}