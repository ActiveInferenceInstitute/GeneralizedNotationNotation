// Stan model generated from GNN: Simple Markov Chain
// Variables: 6, Connections: 6

data {
  vector[1] t;
  matrix[3, 1] o;
}

parameters {
  vector[3] D;
  matrix[3, 1] s;
  matrix[3, 1] s_prime;
  matrix[3, 3] A;
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
  // B → s_prime
  s_prime ~ normal(B, 1.0);
  // s → B
  B ~ normal(s, 1.0);
}