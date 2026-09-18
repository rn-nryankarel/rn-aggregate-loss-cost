functions {
    vector weibull_lcdf_elemwise(vector x, real shape, real scale) {
        int n = size(x);
        vector[n] output;
        for (i in 1:n) {
            output[i] = weibull_lcdf(x[i] | shape, scale);
        }
        return output;
    }
    vector weibull_lcdf_elemwise(vector x, vector shape, vector scale) {
        int n = size(x);
        vector[n] output;
        for (i in 1:n) {
            output[i] = weibull_lcdf(x[i] | shape[i], scale[i]);
        }
        return output;
    }

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

    int N_claims;                                      // number of claim records
    array[N_claims] int<lower=1, upper=N> idx_policy;  // which policy does the claim belong to
    vector<lower=1e-5>[N_claims] R;                    // report lag for a given claim
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
}
parameters {
    // global frequency rates
    real log_lambda_0;           // an intercept
    vector[E_cols] log_lambda_E; // actual frequencies per exposure rates (logged)
    real C_log_phi_mu;           // NB2 dispersion parameter

    // reporting pattern parameterizations
    real R_log_shape_mu;
    real R_log_scale_mu;

    real<lower=0.0> R_log_shape_sigma_rnlob;
    vector[K_rnlob] R_log_shape_rnlob_z;
    real R_log_shape_occurrence_delta;

    real<lower=0.0> R_log_scale_sigma_rnlob;
    vector[K_rnlob] R_log_scale_rnlob_z;
    real R_log_scale_occurrence_delta;

    // cross-classified - non-centered parameterization
    real<lower=0.0> C_log_mu_sigma_class;
    real<lower=0.0> C_log_mu_sigma_state;
    real<lower=0.0> C_log_mu_sigma_rnlob;
    vector[K_class] C_log_mu_class_z;
    vector[K_state] C_log_mu_state_z;
    vector[K_rnlob] C_log_mu_rnlob_z;

    real<lower=0.0> C_log_phi_sigma_rnlob;
    vector[K_rnlob] C_log_phi_rnlob_z;
}
transformed parameters {
    vector[N] C_eta_delta = cross_classify(
        0, // will add in mean later with exposure rates
        C_log_mu_sigma_class, C_log_mu_class_z, g_class,
        C_log_mu_sigma_state, C_log_mu_state_z, g_state,
        C_log_mu_sigma_rnlob, C_log_mu_rnlob_z, g_rnlob
    );
    vector[N] C_log_phi = cross_classify(
        C_log_phi_mu,
        C_log_phi_sigma_rnlob, C_log_phi_rnlob_z, g_rnlob
    );
    vector[N] C_phi = exp(C_log_phi);

    vector[N] R_log_shape = (
        cross_classify(
            R_log_shape_mu,
            R_log_shape_sigma_rnlob, R_log_shape_rnlob_z, g_rnlob
        )
        + R_log_shape_occurrence_delta * to_vector(ind_occurrence)
    );
    vector[N] R_log_scale = (
        cross_classify(
            R_log_scale_mu,
            R_log_scale_sigma_rnlob, R_log_scale_rnlob_z, g_rnlob
        )
        + R_log_scale_occurrence_delta * to_vector(ind_occurrence)
    );
    vector[N] R_shape = exp(R_log_shape);
    vector[N] R_scale = exp(R_log_scale);

    // frequency log mean
    matrix[N, E_cols] log_E_lambda = (
        log_E
        + rep_matrix(log_lambda_E', N)
        + log_lambda_0
    );
    vector[N] log_lambda_ultimate = log_sum_exp_by_row(log_E_lambda);
    vector[N] log_theta_R = weibull_lcdf_elemwise(policy_age, R_shape, R_scale);
    vector[N] C_eta_ultimate = (
        log_lambda_ultimate // global mean
        + log(t)            // annualization
        + C_eta_delta       // cross-classification impact
    );
    vector[N] C_eta_reported = (
        C_eta_ultimate // ultimate mean
        + log_theta_R  // only the reported portion
    );
}
model {
    // priors
    log_lambda_0 ~ normal(-5, 1);
    log_lambda_E ~ normal(-3, 2);
    R_log_shape_mu ~ normal(0, 0.1);
    R_log_scale_mu ~ normal(0, 1);
    C_log_phi_mu ~ normal(1, 0.1);

    R_log_shape_sigma_rnlob ~ exponential(20);
    R_log_shape_rnlob_z ~ std_normal();
    R_log_shape_occurrence_delta ~ normal(-0.5, 1);

    R_log_scale_sigma_rnlob ~ exponential(20);
    R_log_scale_rnlob_z ~ std_normal();
    R_log_scale_occurrence_delta ~ normal(0.5, 1);

    C_log_mu_sigma_class ~ exponential(5);
    C_log_mu_sigma_state ~ exponential(5);
    C_log_mu_sigma_rnlob ~ exponential(5);
    C_log_mu_class_z ~ std_normal();
    C_log_mu_state_z ~ std_normal();
    C_log_mu_rnlob_z ~ std_normal();

    C_log_phi_sigma_rnlob ~ exponential(10);
    C_log_phi_rnlob_z ~ std_normal();


    // likelihood
    C ~ neg_binomial_2_log(C_eta_reported, C_phi);

    // for report lag (truncated by policy age)
    // let's try removing this for now to see if we can fit the model without these - right now it's not training at all
    // for (c in 1:N_claims) {
    //     int p = idx_policy[c]; // policy index
    //     R[c] ~ weibull(R_shape[p], R_scale[p]) T[, R_trunc[c]];
    // }
}
generated quantities {
    vector[N] log_theta_IBNR = log1m_exp(log_theta_R);
    vector[N] C_eta_unreported = (
        C_eta_ultimate   // ultimate mean
        + log_theta_IBNR // only the unreported portion
    );
    // can use the same phi for both:
    // binomial thinning doesn't impact this particular dispersion parameter
    array[N] int<lower=0> C_reported_post_pred = neg_binomial_2_log_rng(C_eta_reported, C_phi);
    array[N] int<lower=0> C_IBNR_post_pred = neg_binomial_2_log_rng(C_eta_unreported, C_phi);
    vector[N] C_reported_log_lik;
    for (n in 1:N) {
        C_reported_log_lik[n] = neg_binomial_2_log_lpmf(C[n] | C_eta_reported[n], C_phi[n]);
    }
}