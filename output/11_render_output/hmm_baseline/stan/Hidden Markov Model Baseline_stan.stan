// Stan model generated from GNN: Hidden Markov Model Baseline
// Variables: 9, Connections: 12

data {
  vector[1] t;
  matrix[6, 1] o;
}

parameters {
  matrix[6, 4] A;
  vector[4] D;
  matrix[4, 1] s;
  matrix[4, 1] s_prime;
  matrix[4, 1] alpha;
  matrix[4, 1] beta;
  matrix[4, 4] B;
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
  // B → s_prime
  s_prime ~ normal(B, 1.0);
  // s → B
  B ~ normal(s, 1.0);
  // s → F
  F ~ normal(s, 1.0);
  // o → F
  F ~ normal(o, 1.0);
  // s → alpha
  alpha ~ normal(s, 1.0);
  // o → alpha
  alpha ~ normal(o, 1.0);
  // alpha → s_prime
  s_prime ~ normal(alpha, 1.0);
  // s_prime → beta
  beta ~ normal(s_prime, 1.0);
}