functions {
    vector weibull_lcdf_elemwise(vector x, real shape, real scale) {
        int n = size(x);
        vector[n] output;
        for (i in 1:n) {
            output[i] = weibull_lcdf(x[i] | shape, scale);
        }
        return output;
    }
    real neg_binomial_2_log_zero_inflated_lpmf(array[] int y, real theta, vector eta, real phi) {
        real log_prob = 0;
        real log_theta = log(theta);
        real log1m_theta = log1m(theta);
        for (n in 1:size(y)) {
            if (y[n] == 0) {
                log_prob += log_sum_exp(
                    log_theta,
                    log1m_theta + neg_binomial_2_log_lpmf(0 | eta[n], phi)
                );
            } else {
                log_prob += log1m_theta + neg_binomial_2_log_lpmf(y[n] | eta[n], phi);
            }
        }
        return log_prob;
    }
    array[] int neg_binomial_2_log_zero_inflated_rng(real theta, vector eta, real phi) {
        array[rows(eta)] int output = neg_binomial_2_log_rng(eta, phi);
        array[rows(eta)] int force_zero = bernoulli_rng(rep_vector(theta, rows(eta)));
        for (n in 1:rows(eta)) {
            if (force_zero[n] == 1) {
                output[n] = 0;
            }
        }
        return output;
    }
}
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
    real<lower=0> lambda_0; // an intercept
    vector<lower=0>[E_cols] lambda_E;
    real<lower=epsilon> R_shape;
    real<lower=epsilon> R_scale;
    real<lower=epsilon> C_R_phi; // NB2 dispersion parameter
    real<lower=0, upper=1> theta_zero; // zero-inflation parameter

    // cross-classified - non-centered parameterization
    real<lower=epsilon> sigma_class;
    real<lower=epsilon> sigma_state;
    vector[K_class] z_class;
    vector[K_state] z_state;
}
transformed parameters {
    // cross-classified - recover factors from non-centered parameterization
    vector[K_class] log_beta_class = z_class * sigma_class;
    vector[K_state] log_beta_state = z_state * sigma_state;

    // frequency log mean
    vector[N] log_lambda_ultimate = log(lambda_0 + E * lambda_E);
    vector[N] log_C_ultimate = log(t) + log_lambda_ultimate;
    vector[N] log_theta_R = weibull_lcdf_elemwise(policy_age, R_shape, R_scale);
    vector[N] log_C_R_mean_raw = log_C_ultimate + log_theta_R;
    vector[N] log_C_R_mean;
    for (n in 1:N) {
        log_C_R_mean[n] = (
            log_C_R_mean_raw[n]
            + log_beta_class[g_class[n]]
            + log_beta_state[g_state[n]]
        );
    }

}
model {
    // priors
    lambda_0 ~ exponential(10); // close to 0 so E[C] ~ exposure
    lambda_E ~ exponential(20);
    R_shape ~ normal(0.8, 0.2);
    R_scale ~ normal(1, 1);
    C_R_phi ~ exponential(1);
    theta_zero ~ beta(1, 1);

    // cross-classified - level 2 hyperpriors
    // I want fairly heavy regularization here
    sigma_class ~ exponential(50);
    sigma_state ~ exponential(50);

    // cross-classified - level 1 priors
    z_class ~ std_normal();
    z_state ~ std_normal();

    // likelihood
    if (!prior_only) {
        // for reported frequency... Binomial(N, p) where N ~ Poisson(lambda) is just Poisson(p * lambda). whoa!
        // https://math.stackexchange.com/q/3967690 - "poisson thinning identity"
        // can also use Binomial(N, p) where N ~ NegBinom() -> NegBinom(mu=N*p)
        C_R ~ neg_binomial_2_log_zero_inflated(theta_zero, log_C_R_mean, C_R_phi);

        // for report lag (truncated by policy age)
        for (c in 1:C_rows) {
            R[c] ~ weibull(R_shape, R_scale) T[, R_trunc[c]];
        }
    }
}
generated quantities {
    array[N] int<lower=0> C_R_post_pred = neg_binomial_2_log_zero_inflated_rng(theta_zero, log_C_R_mean, C_R_phi);
    vector[N] C_R_log_lik;
    for (n in 1:N) {
        // slightly weird arguments to get them into the format required for single function signature
        C_R_log_lik[n] = neg_binomial_2_log_zero_inflated_lpmf({C_R[n]} | theta_zero, log_C_R_mean[n:n], C_R_phi);
    }
}