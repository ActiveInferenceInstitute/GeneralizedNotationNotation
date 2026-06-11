// Stan model generated from GNN: Simple MDP Agent
// Variables: 11, Connections: 10

data {
  matrix[4, 1] o;
  vector[1] t;
  vector[1] u;
}

parameters {
  array[4] matrix[4, 4] B;
  vector[4] D;
  matrix[4, 1] s;
  matrix[4, 1] s_prime;
  matrix[4, 4] A;
  vector[4] C;
  vector[4] π;
  real G;
}

model {
  // D → s
  s ~ normal(D, 1.0);
  // s → A
  A ~ normal(s, 1.0);
  // s → s_prime
  s_prime ~ normal(s, 1.0);
  // A → o
  o ~ normal(A, 1.0);
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