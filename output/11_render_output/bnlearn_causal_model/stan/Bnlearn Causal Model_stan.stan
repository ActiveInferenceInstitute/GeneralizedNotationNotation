// Stan model generated from GNN: Bnlearn Causal Model
// Variables: 4, Connections: 3

data {
  matrix[2, 1] o;
}

parameters {
  matrix[2, 1] s;
  matrix[2, 1] s_prev;
  matrix[2, 1] a;
}

model {
  // s_prev → s
  s ~ normal(s_prev, 1.0);
  // a → s
  s ~ normal(a, 1.0);
  // s → o
  o ~ normal(s, 1.0);
}