data {
    int N;                                        // number of policies
    int E_cols;                                   // number of exposure types
    int K_class;                                  // number of distinct class codes
    int K_state;                                  // number of distinct risk states
    matrix<lower=0>[N, E_cols] E;                 // annualized exposure matrix
    vector<lower=1e-5>[N] t;                      // policy term length
    vector<lower=1e-5>[N] policy_age;             // max possible report lag for policy as of evaluation
    array[N] int<lower=1, upper=K_class> g_class; // categorical variable: policy's class code
    array[N] int<lower=1, upper=K_state> g_state; // categorical variable: policy's risk state

    int C_rows;                                      // number of claim records
    array[C_rows] int<lower=1, upper=N> idx_policy;  // which policy does the claim belong to
    vector<lower=1e-5>[C_rows] R;                    // report lag

    int<lower=0, upper=1> prior_only;
}
transformed data {
    real epsilon = 1e-5;

    array[N] int<lower=0> C_R = zeros_int_array(N);  // reported claims by policy
    vector<lower=epsilon>[C_rows] R_trunc;           // truncation point for each claim
    for (c in 1:C_rows) {
        C_R[idx_policy[c]] += 1;
        R_trunc[c] = policy_age[idx_policy[c]];
    }
}
parameters {
    real mu;
    real<lower=0> sigma;
}
model {
    // priors
    mu ~ normal(0, 10);
    sigma ~ exponential(1 / 10.0);

    // likelihood
    if (!prior_only) {
        C_R ~ normal(mu, sigma);
    }
}
generated quantities {
    array[N] real C_R_post_pred = normal_rng(rep_vector(mu, N), sigma);
    vector[N] C_R_log_lik;
    for (n in 1:N) {
        C_R_log_lik[n] = normal_lpdf(C_R[n] | mu, sigma);
    }
}