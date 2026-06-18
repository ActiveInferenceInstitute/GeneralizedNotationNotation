// Stan model generated from GNN: POMDP GridWorld 3x3
// Variables: 12, Connections: 11

data {
  vector[1] t;
  matrix[9, 1] o;
  vector[1] u;
}

parameters {
  matrix[9, 9] A;
  array[9] matrix[9, 5] B;
  vector[9] D;
  matrix[9, 1] s;
  matrix[9, 1] s_prime;
  vector[9] C;
  vector[5] E;
  vector[5] π;
  real G;
}

model {
  // D → s
  s ~ normal(D, 1.0);
  // s → A
  A ~ normal(s, 1.0);
  // A → o
  o ~ normal(A, 1.0);
  // s → B
  B ~ normal(s, 1.0);
  // B → u
  u ~ normal(B, 1.0);
  // u → s_prime
  s_prime ~ normal(u, 1.0);
  // C → G
  G ~ normal(C, 1.0);
  // E → π
  π ~ normal(E, 1.0);
  // G → π
  π ~ normal(G, 1.0);
  // π → u
  u ~ normal(π, 1.0);
  // s → s_prime
  s_prime ~ normal(s, 1.0);
}