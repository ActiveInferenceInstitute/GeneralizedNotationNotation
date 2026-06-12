// Stan model generated from GNN: Multi Armed Bandit Agent
// Variables: 7, Connections: 10

data {
  vector[1] t;
  matrix[3, 1] o;
  vector[1] u;
}

parameters {
  matrix[3, 3] A;
  matrix[3, 1] s;
  matrix[3, 1] s_prime;
  vector[3] π;
}

model {
  // D → s
  s ~ normal(D, 1.0);
  // s → A
  A ~ normal(s, 1.0);
  // A → o
  o ~ normal(A, 1.0);
  // s → s_prime
  s_prime ~ normal(s, 1.0);
  // s → B
  B ~ normal(s, 1.0);
  // C → G
  G ~ normal(C, 1.0);
  // G → π
  π ~ normal(G, 1.0);
  // π → u
  u ~ normal(π, 1.0);
  // B → u
  u ~ normal(B, 1.0);
  // u → s_prime
  s_prime ~ normal(u, 1.0);
}