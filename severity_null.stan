data {
    int N;                              // number of claim records
    int K;                              // max age
    matrix[N, K] Inc;                   // "ragged" matrix of incurred
    matrix[N, K] Paid;                  // "ragged" matrix of paid

    array[N] int<lower=-1, upper=K> s;  // settlement lag 1-K. -1 means not yet settled. 0 shouldn't be present.
    array[N] int<lower=-1, upper=K> c;  // closure lag 1-K. -1 means not yet closed. 0 shouldn't be present.
    array[N] int<lower=1, upper=K> age; // age of claim as of evaluation date - grows after settlement
}
transformed data {
    array[N, K] int<lower=0, upper=1> Valid;          // matrix of bools indicating whether cell is valid for modeling or not
    array[N, K] int<lower=0, upper=1> PostSettlement; // matrix of bools indicating whether cell took place after claim closure
    array[N, K] int<lower=0, upper=1> PostClosure;    // matrix of bools indicating whether cell took place after claim closure

    for (n in 1:N) {
        for (k in 1:K) {
            Valid[n, k] = k <= age[n];
            PostSettlement[n, k] = (s[n] != -1) && (k > s[n]);
            PostClosure[n, k] = (c[n] != -1) && (k > c[n]);
        }
    }
}
parameters {
    matrix[2, K] mu;
    vector<lower=0>[2] sigma;
}
model {
    for (n in 1:N) {
        for (k in 1:K) {
            if (Valid[n, k]) {
                Inc[n, k] ~ normal(mu[1, k], sigma[1]);
                Paid[n, k] ~ normal(mu[2, k], sigma[2]);
            }
        }
    }
}