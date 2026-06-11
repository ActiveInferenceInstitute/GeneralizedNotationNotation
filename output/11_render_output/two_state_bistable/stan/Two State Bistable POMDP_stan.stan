// Stan model generated from GNN: Two State Bistable POMDP
// Variables: 10, Connections: 11

data {
  vector[1] t;
  matrix[2, 1] o;
  vector[1] u;
}

parameters {
  matrix[2, 2] A;
  vector[2] D;
  matrix[2, 1] s;
  matrix[2, 1] s_prime;
  array[2] matrix[2, 2] B;
  vector[2] E;
  vector[2] π;
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
  // E → π
  π ~ normal(E, 1.0);
  // G → π
  π ~ normal(G, 1.0);
  // π → u
  u ~ normal(π, 1.0);
  // B → u
  u ~ normal(B, 1.0);
  // u → s_prime
  s_prime ~ normal(u, 1.0);
}