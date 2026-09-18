functions {
    vector log_sum_exp_by_row(matrix x) {
        int output_length = rows(x);
        vector[output_length] output;
        for (i in 1:output_length) {
            output[i] = log_sum_exp(x[i]);
        }
        return output;
    }

    vector cross_classify(
        real mu,
        real g1_sigma, vector g1_z, array[] int g1_idx
    ) {
        int M = size(g1_idx);
        vector[M] output;
        for (m in 1:M) {
            output[m] = (
                mu
                + g1_sigma * g1_z[g1_idx[m]]
            );
        }
        return output;
    }
    vector cross_classify(
        real mu,
        real g1_sigma, vector g1_z, array[] int g1_idx,
        real g2_sigma, vector g2_z, array[] int g2_idx,
        real g3_sigma, vector g3_z, array[] int g3_idx
    ) {
        int M = size(g1_idx);
        vector[M] output;
        for (m in 1:M) {
            output[m] = (
                mu
                + g1_sigma * g1_z[g1_idx[m]]
                + g2_sigma * g2_z[g2_idx[m]]
                + g3_sigma * g3_z[g3_idx[m]]
            );
        }
        return output;
    }
}
data {
    int N;                                         // number of policies
    int E_cols;                                    // number of exposure types
    int K_class;                                   // number of distinct class codes
    int K_state;                                   // number of distinct risk states
    int K_rnlob;                                   // number of distinct products
    matrix<lower=0>[N, E_cols] E;                  // annualized exposure matrix
    vector<lower=1e-5>[N] t;                       // policy term length
    vector<lower=1e-5>[N] policy_age;              // max possible report lag for policy as of evaluation
    array[N] int<lower=1, upper=K_class> g_class;  // categorical variable: policy's class code
    array[N] int<lower=1, upper=K_state> g_state;  // categorical variable: policy's risk state
    array[N] int<lower=1, upper=K_rnlob> g_rnlob;  // categorical variable: policy's LOB
    array[N] int<lower=0, upper=1> ind_occurrence; // binary indicator - occurrence coverage form

    // important! percent of claims reported, theta_R, is given here instead of modeled as data
    vector<lower=1e-5>[N] theta_R;                 // percent of claims reported for a given policy - output from another model

    int N_claims;                                      // number of claim records
    array[N_claims] int<lower=1, upper=N> idx_policy;  // which policy does the claim belong to
    vector<lower=0>[N_claims] R;                       // report lag for a given claim

    int<lower=0, upper=1> prior_only;
}
transformed data {
    real epsilon = 0.0001;
    matrix[N, E_cols] log_E = log(E + epsilon);   // logged exposure - shifted slightly to handle 0 cases
    array[N] int<lower=0> C = zeros_int_array(N); // reported claims by policy
    vector<lower=0>[N_claims] R_trunc;            // truncation point for each claim
    for (c in 1:N_claims) {
        C[idx_policy[c]] += 1;
        R_trunc[c] = policy_age[idx_policy[c]];
    }

    vector[N] log_theta_R = log(theta_R);
    vector[N] log_theta_IBNR = log1m(theta_R);
}
parameters {
    // global frequency rates
    real log_lambda_0;           // an intercept
    vector[E_cols] log_lambda_E; // actual frequencies per exposure rates (logged)
    real C_log_phi_mu;           // NB2 dispersion parameter

    // cross-classified - non-centered parameterization
    real<lower=0.0> C_eta_sigma_class;
    real<lower=0.0> C_eta_sigma_state;
    real<lower=0.0> C_eta_sigma_rnlob;
    vector[K_class] C_eta_class_z;
    vector[K_state] C_eta_state_z;
    vector[K_rnlob] C_eta_rnlob_z;

    real<lower=0.0> C_log_phi_sigma_rnlob;
    vector[K_rnlob] C_log_phi_rnlob_z;
}
transformed parameters {
    vector[N] C_eta_delta = cross_classify(
        0, // will add in mean later with exposure rates
        C_eta_sigma_class, C_eta_class_z, g_class,
        C_eta_sigma_state, C_eta_state_z, g_state,
        C_eta_sigma_rnlob, C_eta_rnlob_z, g_rnlob
    );
    vector[N] C_log_phi = cross_classify(
        C_log_phi_mu,
        C_log_phi_sigma_rnlob, C_log_phi_rnlob_z, g_rnlob
    );
    vector[N] C_phi = exp(C_log_phi);

    // frequency log mean
    vector[N] log_lambda_ultimate = log_sum_exp_by_row(
        // not storing the E * lambda intermediate step for the sake of memory consumption
        log_E
        + rep_matrix(log_lambda_E', N)
        + log_lambda_0
    );
    vector[N] C_eta_ultimate = (
        log_lambda_ultimate // global mean
        + log(t)            // de-annualization
        + C_eta_delta       // cross-classification impact
    );
    vector[N] C_eta_reported = (
        C_eta_ultimate // ultimate mean
        + log_theta_R  // only the reported portion
    );
}
model {
    // priors
    log_lambda_0 ~ normal(-3, 1);
    log_lambda_E ~ normal(-2, 4);
    C_log_phi_mu ~ normal(0, 3);

    C_eta_sigma_class ~ exponential(1);
    C_eta_sigma_state ~ exponential(1);
    C_eta_sigma_rnlob ~ exponential(1);
    C_eta_class_z ~ std_normal();
    C_eta_state_z ~ std_normal();
    C_eta_rnlob_z ~ std_normal();

    C_log_phi_sigma_rnlob ~ exponential(1);
    C_log_phi_rnlob_z ~ std_normal();

    // likelihood
    if (!prior_only) {
        C ~ neg_binomial_2_log(C_eta_reported, C_phi);
    }
}
generated quantities {
    vector[N] C_eta_unreported = (
        C_eta_ultimate   // ultimate mean
        + log_theta_IBNR // only the unreported portion
    );
    // can use the same phi for both:
    // binomial thinning doesn't impact the dispersion parameter
    array[N] int<lower=0> C_reported_post_pred = neg_binomial_2_log_rng(C_eta_reported, C_phi);
    array[N] int<lower=0> C_IBNR_post_pred = neg_binomial_2_log_rng(C_eta_unreported, C_phi);
    vector[N] C_reported_log_lik;
    for (n in 1:N) {
        C_reported_log_lik[n] = neg_binomial_2_log_lpmf(C[n] | C_eta_reported[n], C_phi[n]);
    }
}