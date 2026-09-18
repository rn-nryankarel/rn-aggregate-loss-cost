// it was becoming too challenging to model both reported frequency and
// theta_R = % of claims reported as of evaluation simultaneously.
// instead, let's model R in an isolated fashion, estimate theta_R by
// policy, and use the a posteriori mean estimate of theta_R in the
// frequency model.
functions {
    real truncated_weibull_rng(real alpha, real sigma, real upper_bound) {
        real p = weibull_cdf(upper_bound | alpha, sigma);  // cdf for upper_bound
        real u = uniform_rng(0, p);                        // unif in bounds
        real y = sigma * (-log1m(u))^inv(alpha);           // inverse cdf
        return y;
    }
    real truncated_hurdled_weibull_rng(real alpha, real sigma, real p0_logit, real upper_bound) {
        real p = weibull_cdf(upper_bound | alpha, sigma);  // cdf for upper_bound
        real u = uniform_rng(0, p);                        // unif in bounds
        real y = sigma * (-log1m(u))^inv(alpha);           // inverse cdf
        real hurdle_gate = 1 - bernoulli_logit_rng(p0_logit);
        return y * hurdle_gate;
    }


    real hurdle_weibull_lpdf(vector y, vector alpha, vector sigma, vector p0_logit) {
        int size_y = size(y);
        real lp = 0;
        for (i in 1:size_y) {
            real y_i = y[i];
            if (y_i == 0) {
                lp += log_inv_logit(p0_logit[i]);
            } else {
                lp += log1m_inv_logit(p0_logit[i]) + weibull_lpdf(y_i | alpha[i], sigma[i]);
            }
        }
        return lp;
    }
    real hurdle_weibull_lpdf(real y, real alpha, real sigma, real p0_logit) {
        return hurdle_weibull_lpdf(
            [y]' |
            [alpha]',
            [sigma]',
            [p0_logit]'
        );
    }
    real hurdle_weibull_lcdf(vector y, vector alpha, vector sigma, vector p0_logit) {
        int size_y = size(y);
        real lp = 0;
        for (i in 1:size_y) {
            real y_i = y[i];
            real log_p0 = log_inv_logit(p0_logit[i]);
            if (y_i == 0) {
                lp += log_p0;
            } else {
                // F(y) = p0 + (1-p0) * weibull_cdf(y), combined on the log scale
                lp += log_sum_exp(
                    log_p0,
                    log1m_inv_logit(p0_logit[i]) + weibull_lcdf(y_i | alpha[i], sigma[i])
                );
            }
        }
        return lp;
    }
    real hurdle_weibull_lcdf(real y, real alpha, real sigma, real p0_logit) {
        return hurdle_weibull_lcdf(
            [y]' |
            [alpha]',
            [sigma]',
            [p0_logit]'
        );
    }
    real hurdle_weibull_lccdf(vector y, vector alpha, vector sigma, vector p0_logit) {
        int size_y = size(y);
        real lp = 0;
        for (i in 1:size_y) {
            real y_i = y[i];
            real log_1m_p0 = log1m_inv_logit(p0_logit[i]);
            if (y_i == 0) {
                lp += log_1m_p0;
            } else {
                // 1-F(y) = (1-p0) * weibull_ccdf(y)
                lp += log_1m_p0 + weibull_lccdf(y_i | alpha[i], sigma[i]);
            }
        }
        return lp;
    }
    real hurdle_weibull_cdf(vector y, vector alpha, vector sigma, vector p0_logit) {
        return exp(hurdle_weibull_lcdf(y | alpha, sigma, p0_logit));
    }
    real hurdle_weibull_ccdf(vector y, vector alpha, vector sigma, vector p0_logit) {
        return exp(hurdle_weibull_lccdf(y | alpha, sigma, p0_logit));
    }
    real hurdle_weibull_cdf(real y, real alpha, real sigma, real p0_logit) {
        return hurdle_weibull_cdf(
            [y]' |
            [alpha]',
            [sigma]',
            [p0_logit]'
        );
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
    int E_cols;                                    // number of exposure types - not used in this model, but leaving for consistency
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
    vector<lower=0>[N_claims] R;                       // report lag for a given claim

    int<lower=0, upper=1> prior_only;
}
transformed data {
    vector[N_claims] R_trunc = policy_age[idx_policy];
    for (c in 1:N_claims) {
        if (R[c] >= R_trunc[c]) {
            reject("R[c]=", R[c], " >= R_trunc[c]=", R_trunc[c], " at c=", c);
        }
    }
}
parameters {
    // global frequency rates
    // reporting pattern parameterizations
    real R_log_shape_mu;
    real R_log_scale_mu;
    real R_p0_logit_mu;

    real<lower=0.0> R_log_shape_sigma_class;
    real<lower=0.0> R_log_shape_sigma_state;
    real<lower=0.0> R_log_shape_sigma_rnlob;
    vector[K_class] R_log_shape_class_z;
    vector[K_state] R_log_shape_state_z;
    vector[K_rnlob] R_log_shape_rnlob_z;
    real R_log_shape_occurrence_delta;

    real<lower=0.0> R_log_scale_sigma_class;
    real<lower=0.0> R_log_scale_sigma_state;
    real<lower=0.0> R_log_scale_sigma_rnlob;
    vector[K_class] R_log_scale_class_z;
    vector[K_state] R_log_scale_state_z;
    vector[K_rnlob] R_log_scale_rnlob_z;
    real R_log_scale_occurrence_delta;

    // real<lower=0.0> R_p0_logit_sigma_class;
    // real<lower=0.0> R_p0_logit_sigma_state;
    real<lower=0.0> R_p0_logit_sigma_rnlob;
    // vector[K_class] R_p0_logit_class_z;
    // vector[K_state] R_p0_logit_state_z;
    vector[K_rnlob] R_p0_logit_rnlob_z;
    real R_p0_logit_occurrence_delta;
}
transformed parameters {
    vector[N] R_log_shape = (
        cross_classify(
            R_log_shape_mu,
            R_log_shape_sigma_class, R_log_shape_class_z, g_class,
            R_log_shape_sigma_state, R_log_shape_state_z, g_state,
            R_log_shape_sigma_rnlob, R_log_shape_rnlob_z, g_rnlob
        )
        + R_log_shape_occurrence_delta * to_vector(ind_occurrence)
    );
    vector[N] R_log_scale = (
        cross_classify(
            R_log_scale_mu,
            R_log_scale_sigma_class, R_log_scale_class_z, g_class,
            R_log_scale_sigma_state, R_log_scale_state_z, g_state,
            R_log_scale_sigma_rnlob, R_log_scale_rnlob_z, g_rnlob
        )
        + R_log_scale_occurrence_delta * to_vector(ind_occurrence)
    );
    vector[N] R_p0_logit = (
        cross_classify(
            R_p0_logit_mu,
            // R_p0_logit_sigma_class, R_p0_logit_class_z, g_class,
            // R_p0_logit_sigma_state, R_p0_logit_state_z, g_state,
            R_p0_logit_sigma_rnlob, R_p0_logit_rnlob_z, g_rnlob
        )
        + R_p0_logit_occurrence_delta * to_vector(ind_occurrence)
    );
    vector[N] R_shape = exp(R_log_shape);
    vector[N] R_scale = exp(R_log_scale);

    vector[N_claims] R_shape_by_claim = R_shape[idx_policy];
    vector[N_claims] R_scale_by_claim = R_scale[idx_policy];
    vector[N_claims] R_p0_logit_by_claim = R_p0_logit[idx_policy];
}
model {
    // priors
    R_log_shape_mu ~ normal(0, 1);
    R_log_scale_mu ~ normal(-1, 1);
    R_p0_logit_mu ~ normal(-1, 3);

    R_log_shape_sigma_class ~ exponential(10);
    R_log_shape_sigma_state ~ exponential(10);
    R_log_shape_sigma_rnlob ~ exponential(2);
    R_log_shape_class_z ~ std_normal();
    R_log_shape_state_z ~ std_normal();
    R_log_shape_rnlob_z ~ std_normal();
    R_log_shape_occurrence_delta ~ normal(0, 0.5);

    R_log_scale_sigma_class ~ exponential(10);
    R_log_scale_sigma_state ~ exponential(10);
    R_log_scale_sigma_rnlob ~ exponential(2);
    R_log_scale_class_z ~ std_normal();
    R_log_scale_state_z ~ std_normal();
    R_log_scale_rnlob_z ~ std_normal();
    R_log_scale_occurrence_delta ~ normal(3, 3);

    // R_p0_logit_sigma_class ~ exponential(10);
    // R_p0_logit_sigma_state ~ exponential(10);
    R_p0_logit_sigma_rnlob ~ exponential(2);
    // R_p0_logit_class_z ~ std_normal();
    // R_p0_logit_state_z ~ std_normal();
    R_p0_logit_rnlob_z ~ std_normal();
    R_p0_logit_occurrence_delta ~ normal(-5, 3);

    // for report lag (truncated by policy age), vectorized over claims
    // less legible than the for loop with its T[,] notation, but
    // vectorized and faster
    if (!prior_only) {
        target += (
            hurdle_weibull_lpdf(
                R |
                R_shape_by_claim,
                R_scale_by_claim,
                R_p0_logit_by_claim
            )
            - hurdle_weibull_lcdf(
                R_trunc |
                R_shape_by_claim,
                R_scale_by_claim,
                R_p0_logit_by_claim
            )
        );
    }
}
generated quantities {
    // sample from R distribution to ensure we're fitting this one appropriately
    // we'll want to pay special attention to claims-made policies to ensure that
    // we're capturing the more extreme ends of the distribution properly - eg
    // fast reporting up front, then potential unloading near the end of the term.
    array[N_claims] real R_post_pred;
    vector[N_claims] R_log_lik;
    for (c in 1:N_claims) {
        R_post_pred[c] = truncated_hurdled_weibull_rng(
            R_shape_by_claim[c],
            R_scale_by_claim[c],
            R_p0_logit_by_claim[c],
            R_trunc[c]
        );
        R_log_lik[c] = (
            hurdle_weibull_lpdf(
                R[c] |
                R_shape_by_claim[c],
                R_scale_by_claim[c],
                R_p0_logit_by_claim[c]
            )
            - hurdle_weibull_lcdf(
                R_trunc[c] |
                R_shape_by_claim[c],
                R_scale_by_claim[c],
                R_p0_logit_by_claim[c]
            )
        );
    }

    vector[N] theta_R_posterior;
    for (p in 1:N) {
        theta_R_posterior[p] = hurdle_weibull_cdf(policy_age[p] | R_shape[p], R_scale[p], R_p0_logit[p]);
    }

}