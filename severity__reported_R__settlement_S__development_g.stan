// gamma severity with beta-thinning to approximate reporting pattern
// let ultimate severity X ~ gamma(alpha, beta).
// let settlement lag S ~ exponential(lambda).
// let T be the age of a claim.
// let "percent reported" g be beta-distributed with
//     ... mean parameter := CDF @ T truncated at S
//     ... total count := X's alpha.
//     ... we're using the Weibull distribution's CDF
// then through beta-thinning, we can decompose X into a reported portion
// R ~ gamma(E[g] * alpha, beta) and unobserved & unmodeled IBNR portion.
// admittedly, we're only assuming a beta distribution for g to be able
// to use this handy decomposition.

functions {
    int count_positives(array[] int x) {
        int pos = 0;
        for (xi in x) {
            if (xi > 0) pos += 1;
        }
        return pos;
    }
    // this will serve as our log mean percent reported function, ln E[g_S(T)]
    // note that we're always truncating at the scale itself, which allows us
    // to subtract a constant "given" probability to account for truncation
    vector truncated_weibull_lcdf_elemwise(vector T, real weibull_shape, vector weibull_scale) {
        int M = size(T);
        vector[M] output;
        real weibull_lcdf_S = log1m_exp(-1);
        for (m in 1:M) {
            output[m] = weibull_lcdf(T[m] | weibull_shape, weibull_scale[m]);
        }
        return output - weibull_lcdf_S;
    }
}
data {
    int N;                                  // number of claim records
    vector<lower=0>[N] R;                   // reported claims
    vector<lower=0>[N] S;                   // settlement lag for all claims - denoted 0 for unsettled claims
    vector<lower=0>[N] T;                   // current age of claim
    array[N] int<lower=0, upper=1> settled; // indicator showing if claim is settled or not yet
}
transformed data {
    int N_settled = sum(settled);
    int N_unsettled = N - N_settled;

    vector<lower=0>[N_settled] S_known;
    vector<lower=0>[N_unsettled] S_unknown_lb;
    vector<lower=0>[N_settled] R_settled;
    vector<lower=0>[N_unsettled] R_unsettled;
    array[N] int<lower=0, upper=N_unsettled> claim_idx_to_unsettled_idx;
    int counter_settled = 0;
    int counter_unsettled = 0;
    for (n in 1:N) {
        if (settled[n]) {
            counter_settled += 1;
            S_known[counter_settled] = S[n];
            R_settled[counter_settled] = R[n];
            claim_idx_to_unsettled_idx[n] = 0;
        } else {
            counter_unsettled += 1;
            S_unknown_lb[counter_unsettled] = T[n];
            R_unsettled[counter_unsettled] = R[n];
            claim_idx_to_unsettled_idx[n] = counter_unsettled;
        }
    }
}
parameters {
    // settlement lag log rate
    real S_exponential_log_lambda;

    // ultimate severity parameters
    real X_gamma_log_alpha;
    real X_gamma_log_beta;

    // development function parameters
    real<lower=0> g_weibull_alpha;

    // latent variables representing remaining time to settlement
    vector<lower=0>[N_unsettled] S_unknown_minus_age;
}
transformed parameters {
    real<lower=0> S_exponential_lambda = exp(S_exponential_log_lambda);
    real<lower=0> X_gamma_alpha = exp(X_gamma_log_alpha);
    real<lower=0> X_gamma_beta = exp(X_gamma_log_beta);

    vector<lower=0>[N_unsettled] S_unknown = S_unknown_lb + S_unknown_minus_age;
    // if we can marginalize g_mu to it doesn't depend on the latent
    // S_unknown directly, that should improve sampling
    vector[N_unsettled] log_g_mu = truncated_weibull_lcdf_elemwise(S_unknown_lb, g_weibull_alpha, S_unknown);
    vector<lower=0>[N_unsettled] X_reported_gamma_alpha = exp(X_gamma_log_alpha + log_g_mu);
}
model {
    // priors
    S_exponential_log_lambda ~ normal(0, 0.1);
    X_gamma_log_alpha ~ normal(-1, 3);
    X_gamma_log_beta ~ normal(-5, 5);
    g_weibull_alpha ~ lognormal(0.0625, 0.25); // mode at 1.0

    S_known ~ exponential(S_exponential_lambda);
    // conditional density of the excess-over-T already accounts for S > T via memorylessness;
    // no separate exponential_lccdf(T | lambda) term is needed (would double-count)
    S_unknown_minus_age ~ exponential(S_exponential_lambda);

    // main likelihoods of model
    // second statement exploits beta-decomposition of gamma variable,
    // implicitly assuming g ~ beta_proportion(g_mu, X_gamma_alpha)
    R_settled ~ gamma(X_gamma_alpha, X_gamma_beta);
    R_unsettled ~ gamma(X_reported_gamma_alpha, X_gamma_beta);
}
generated quantities {
    vector<lower=0>[N] R_post_pred;
    vector[N] R_log_lik;
    for (n in 1:N) {
        if (settled[n]) {
            R_post_pred[n] = gamma_rng(X_gamma_alpha, X_gamma_beta);
            R_log_lik[n] = gamma_lpdf(R[n] | X_gamma_alpha, X_gamma_beta);
        } else {
            int unsettled_idx = claim_idx_to_unsettled_idx[n];
            R_post_pred[n] = gamma_rng(X_reported_gamma_alpha[unsettled_idx], X_gamma_beta);
            R_log_lik[n] = gamma_lpdf(R[n] | X_reported_gamma_alpha[unsettled_idx], X_gamma_beta);
        }
    }
}
