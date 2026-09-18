// gamma severity with beta-thinning to approximate reporting pattern
// let ultimate severity X ~ gamma(alpha, beta).
// let settlement lag S ~ exponential(lambda).
// let T be the age of a claim.
// let "percent reported" g be beta-distributed with
//     ... mean parameter := CDF @ T truncated at S
//     ... total count := X's alpha.
//     ... we're using the Weibull distribution's CDF
// then through beta-thinning, we can decompose X into a reported portion
// R ~ gamma(E[g] * alpha, beta) and unobserved IBNER portion.
// admittedly, we're only assuming a beta distribution for g to be able
// to use this handy decomposition.

functions {
    // this will serve as our log mean percent reported function, ln E[g_S(T)]
    // note that we're always truncating at the scale itself, which allows us
    // to subtract a constant "given" probability to account for truncation
    real truncated_weibull_lcdf(real T, real weibull_shape, real weibull_scale) {
        real weibull_lcdf_S = log1m_exp(-1);
        return weibull_lcdf(T | weibull_shape, weibull_scale) - weibull_lcdf_S;
    }

    // hurdle-gamma distribution
    real hurdle_gamma_lpdf(vector y, vector alpha, vector beta, vector p0_logit) {
        // p0 = hurdle probability of 0
        real lp = 0;
        for (i in 1:size(y)) {
            real y_i = y[i];
            if (y_i == 0) {
                lp += log_inv_logit(p0_logit[i]);
            } else {
                lp += log1m_inv_logit(p0_logit[i]) + gamma_lpdf(y_i | alpha[i], beta[i]);
            }
        }
        return lp;
    }
    real hurdle_gamma_lpdf(real y, real alpha, real beta, real p0_logit) {
        return hurdle_gamma_lpdf([y]' | [alpha]', [beta]', [p0_logit]');
    }
    real hurdle_gamma_rng(real alpha, real beta, real p0_logit) {
        return (1 - bernoulli_logit_rng(p0_logit)) * gamma_rng(alpha, beta);
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
        real g2_sigma, vector g2_z, array[] int g2_idx
    ) {
        int M = size(g1_idx);
        vector[M] output;
        for (m in 1:M) {
            output[m] = (
                mu
                + g1_sigma * g1_z[g1_idx[m]]
                + g2_sigma * g2_z[g2_idx[m]]
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
    vector cross_classify(
        real mu,
        real g1_sigma, vector g1_z, array[] int g1_idx,
        real g2_sigma, vector g2_z, array[] int g2_idx,
        real g3_sigma, vector g3_z, array[] int g3_idx,
        real g4_sigma, vector g4_z, array[] int g4_idx
    ) {
        int M = size(g1_idx);
        vector[M] output;
        for (m in 1:M) {
            output[m] = (
                mu
                + g1_sigma * g1_z[g1_idx[m]]
                + g2_sigma * g2_z[g2_idx[m]]
                + g3_sigma * g3_z[g3_idx[m]]
                + g4_sigma * g4_z[g4_idx[m]]
            );
        }
        return output;
    }
    vector cross_classify(
        real mu,
        real g1_sigma, vector g1_z, array[] int g1_idx,
        real g2_sigma, vector g2_z, array[] int g2_idx,
        real g3_sigma, vector g3_z, array[] int g3_idx,
        real g4_sigma, vector g4_z, array[] int g4_idx,
        real g5_sigma, vector g5_z, array[] int g5_idx
    ) {
        int M = size(g1_idx);
        vector[M] output;
        for (m in 1:M) {
            output[m] = (
                mu
                + g1_sigma * g1_z[g1_idx[m]]
                + g2_sigma * g2_z[g2_idx[m]]
                + g3_sigma * g3_z[g3_idx[m]]
                + g4_sigma * g4_z[g4_idx[m]]
                + g5_sigma * g5_z[g5_idx[m]]
            );
        }
        return output;
    }
}
data {
    int N;                                  // number of claim records
    vector<lower=0>[N] R;                   // reported claims
    vector<lower=0>[N] S;                   // settlement lag for all claims - denoted 0 for unsettled claims
    vector<lower=0>[N] T;                   // current age of claim
    array[N] int<lower=0, upper=1> settled; // indicator showing if claim is settled or not yet

    int K_class;                                   // number of distinct class codes
    int K_state;                                   // number of distinct risk states
    int K_rnlob;                                   // number of distinct RN LOB
    int K_svrty;                                   // number of distinct severity codes
    int K_cause;                                   // number of distinct causes of loss
    array[N] int<lower=1, upper=K_class> g_class;  // categorical variable: claim's class code
    array[N] int<lower=1, upper=K_state> g_state;  // categorical variable: claim's risk state
    array[N] int<lower=1, upper=K_rnlob> g_rnlob;  // categorical variable: claim's rnlob
    array[N] int<lower=1, upper=K_svrty> g_svrty;  // categorical variable: claim's severity code
    array[N] int<lower=1, upper=K_cause> g_cause;  // categorical variable: claim's cause of loss
}
transformed data {
    int N_settled = sum(settled);
    int N_unsettled = N - N_settled;

    vector<lower=0>[N_settled] S_known;
    vector<lower=0>[N_unsettled] S_unknown_lb;
    vector<lower=0>[N_settled] R_settled;
    vector<lower=0>[N_unsettled] R_unsettled;
    array[N] int<lower=0, upper=N_unsettled> claim_idx_to_unsettled_idx;
    array[N] int<lower=0, upper=N_settled> claim_idx_to_settled_idx;
    int counter_settled = 0;
    int counter_unsettled = 0;
    for (n in 1:N) {
        if (settled[n]) {
            counter_settled += 1;
            S_known[counter_settled] = S[n];
            R_settled[counter_settled] = R[n];
            claim_idx_to_unsettled_idx[n] = 0;
            claim_idx_to_settled_idx[n] = counter_settled;
        } else {
            counter_unsettled += 1;
            S_unknown_lb[counter_unsettled] = T[n];
            R_unsettled[counter_unsettled] = R[n];
            claim_idx_to_unsettled_idx[n] = counter_unsettled;
            claim_idx_to_settled_idx[n] = 0;
        }
    }
}
parameters {
    // settlement lag log rate - only varying by LOB and severity code
    real S_log_lambda_mu;
    real<lower=0> S_log_lambda_sigma_rnlob;
    real<lower=0> S_log_lambda_sigma_svrty;

    // ultimate severity parameters
    real X_log_alpha_mu;
    real X_log_beta_mu;
    real<lower=0> X_log_alpha_sigma_class;
    real<lower=0> X_log_beta_sigma_class;
    real<lower=0> X_log_alpha_sigma_state;
    real<lower=0> X_log_alpha_sigma_rnlob;
    real<lower=0> X_log_beta_sigma_rnlob;
    real<lower=0> X_log_beta_sigma_svrty;
    real<lower=0> X_log_alpha_sigma_cause;
    real<lower=0> X_log_beta_sigma_cause;

    // hurdle rate parameters
    real X_p0_logit_mu;
    real<lower=0> X_p0_logit_sigma_rnlob;
    real<lower=0> X_p0_logit_sigma_svrty;

    // development function parameters
    real g_log_alpha_mu;
    real<lower=0> g_log_alpha_sigma_rnlob;

    // parameters for scale trend - these will be subtracted from the
    // Gamma log-beta parameter (after scaling by claim age).
    // will only vary by LOB for now.
    real X_log_scale_trend_mu;
    real<lower=0> X_log_scale_trend_sigma_rnlob;

    // latent variables representing remaining time to settlement
    // TODO: we should still aim to marginalize this out if possible
    vector<lower=0>[N_unsettled] S_unknown_minus_age;

    // non-centered parameterization of cross-classified variables
    vector[K_class] X_log_alpha_class_z;
    vector[K_state] X_log_alpha_state_z;
    vector[K_rnlob] X_log_alpha_rnlob_z;
    vector[K_cause] X_log_alpha_cause_z;
    vector[K_class] X_log_beta_class_z;
    vector[K_rnlob] X_log_beta_rnlob_z;
    vector[K_svrty] X_log_beta_svrty_z;
    vector[K_cause] X_log_beta_cause_z;

    vector[K_rnlob] X_p0_logit_rnlob_z;
    vector[K_svrty] X_p0_logit_svrty_z;

    vector[K_rnlob] S_log_lambda_rnlob_z;
    vector[K_svrty] S_log_lambda_svrty_z;
    vector[K_rnlob] g_log_alpha_rnlob_z;
    vector[K_rnlob] X_log_scale_trend_rnlob_z;
}
transformed parameters {
    vector<lower=0>[N_unsettled] S_unknown = S_unknown_lb + S_unknown_minus_age;

    vector[N] X_log_alpha = cross_classify(
        X_log_alpha_mu,
        X_log_alpha_sigma_class, X_log_alpha_class_z, g_class,
        X_log_alpha_sigma_state, X_log_alpha_state_z, g_state,
        X_log_alpha_sigma_rnlob, X_log_alpha_rnlob_z, g_rnlob,
        X_log_alpha_sigma_cause, X_log_alpha_cause_z, g_cause
    );
    vector[N] X_log_beta = cross_classify(
        X_log_beta_mu,
        X_log_beta_sigma_class, X_log_beta_class_z, g_class,
        X_log_beta_sigma_rnlob, X_log_beta_rnlob_z, g_rnlob,
        X_log_beta_sigma_svrty, X_log_beta_svrty_z, g_svrty,
        X_log_beta_sigma_cause, X_log_beta_cause_z, g_cause
    );
    vector[N] X_p0_logit = cross_classify(
        X_p0_logit_mu,
        X_p0_logit_sigma_rnlob,  X_p0_logit_rnlob_z, g_rnlob,
        X_p0_logit_sigma_svrty,  X_p0_logit_svrty_z, g_svrty
    );
    vector[N] S_log_lambda = cross_classify(
        S_log_lambda_mu,
        S_log_lambda_sigma_rnlob,  S_log_lambda_rnlob_z, g_rnlob,
        S_log_lambda_sigma_svrty,  S_log_lambda_svrty_z, g_svrty
    );
    vector[N] g_log_alpha = cross_classify(
        g_log_alpha_mu,
        g_log_alpha_sigma_rnlob,  g_log_alpha_rnlob_z, g_rnlob
    );
    vector[N] X_log_scale_trend = cross_classify(
        X_log_scale_trend_mu,
        X_log_scale_trend_sigma_rnlob,  X_log_scale_trend_rnlob_z, g_rnlob
    );

    vector<lower=0>[N] S_lambda = exp(S_log_lambda);
    vector<lower=0>[N] g_alpha = exp(g_log_alpha);
    // reasoning on adding the trend variable:
    // * beta is an inverse scale parameter
    // * trend is for *detrending* back from today's cost level
    //   back to past claim's cost level
    // * intuitively, adding a positive trend value to
    //   log_beta will increase the inverse scale and thus reduce
    //   the scale, which is what we'd want for detrending
    vector<lower=0>[N] X_beta = exp(X_log_beta + X_log_scale_trend .* T);
    vector<lower=0>[N] X_alpha = exp(X_log_alpha);
    vector<lower=0>[N] X_alpha_reported;
    vector<lower=0>[N] X_alpha_unreported;
    vector<lower=0>[N_settled] S_lambda_settled;
    vector<lower=0>[N_unsettled] S_lambda_unsettled;

    // beta-thinning multiplicand (called "epsilon" in this paper:
    // https://www.jmlr.org/papers/volume25/23-0446/23-0446.pdf, p. 6)
    // if we can marginalize epsilon to it doesn't depend on the latent
    // S_unknown directly, that should improve sampling
    vector[N] log_epsilon = zeros_vector(N);
    for (n in 1:N) {
        if (settled[n]) {
            S_lambda_settled[claim_idx_to_settled_idx[n]] = S_lambda[n];
        } else {
            int idx_unsettled = claim_idx_to_unsettled_idx[n];
            log_epsilon[n] = truncated_weibull_lcdf(
                S_unknown_lb[idx_unsettled] |
                g_alpha[n],
                S_unknown[idx_unsettled]
            );
            S_lambda_unsettled[idx_unsettled] = S_lambda[n];
        }
    }
    X_alpha_reported = exp(X_log_alpha + log_epsilon);
    X_alpha_unreported = X_alpha - X_alpha_reported;
}
model {
    // level 2 priors
    S_log_lambda_mu ~ normal(0, 0.2);
    S_log_lambda_sigma_rnlob ~ exponential(10);
    S_log_lambda_sigma_svrty ~ exponential(10);

    X_log_alpha_mu ~ normal(-1, 1);
    X_log_beta_mu ~ normal(-5, 5);
    X_log_alpha_sigma_class ~ exponential(10);
    X_log_beta_sigma_class ~ exponential(10);
    X_log_alpha_sigma_state ~ exponential(10);
    X_log_alpha_sigma_rnlob ~ exponential(1);
    X_log_beta_sigma_rnlob ~ exponential(1);
    X_log_beta_sigma_svrty ~ exponential(0.5);
    X_log_alpha_sigma_cause ~ exponential(1);
    X_log_beta_sigma_cause ~ exponential(1);

    X_p0_logit_mu ~ normal(0, 1);
    X_p0_logit_sigma_rnlob ~ exponential(1);
    X_p0_logit_sigma_svrty ~ exponential(5);

    g_log_alpha_mu ~ normal(0.0, 0.05);
    g_log_alpha_sigma_rnlob ~ exponential(10);

    X_log_scale_trend_mu ~ normal(0.06, 0.03);
    X_log_scale_trend_sigma_rnlob ~ exponential(100);

    // level 1 priors
    X_log_alpha_class_z ~ std_normal();
    X_log_alpha_state_z ~ std_normal();
    X_log_alpha_rnlob_z ~ std_normal();
    X_log_alpha_cause_z ~ std_normal();
    X_log_beta_class_z ~ std_normal();
    X_log_beta_rnlob_z ~ std_normal();
    X_log_beta_svrty_z ~ std_normal();
    X_log_beta_cause_z ~ std_normal();
    X_p0_logit_rnlob_z ~ std_normal();
    X_p0_logit_svrty_z ~ std_normal();
    S_log_lambda_rnlob_z ~ std_normal();
    S_log_lambda_svrty_z ~ std_normal();
    g_log_alpha_rnlob_z ~ std_normal();
    X_log_scale_trend_rnlob_z ~ std_normal();

    S_known ~ exponential(S_lambda_settled);
    // conditional density of the excess-over-T already accounts for S > T via memorylessness;
    // no separate exponential_lccdf(T | lambda) term is needed (would double-count)
    S_unknown_minus_age ~ exponential(S_lambda_unsettled);

    // main likelihoods of model
    // exploits beta-decomposition of gamma variable for unsettled claims
    R ~ hurdle_gamma(X_alpha_reported, X_beta, X_p0_logit);
}
generated quantities {
    array[N] real<lower=0> R_post_pred;
    array[N] real<lower=0> IBNER_post_pred = zeros_array(N);
    vector[N] R_log_lik;
    for (n in 1:N) {
        R_post_pred[n] = hurdle_gamma_rng(X_alpha_reported[n], X_beta[n], X_p0_logit[n]);
        R_log_lik[n] = hurdle_gamma_lpdf(R[n] | X_alpha_reported[n], X_beta[n], X_p0_logit[n]);
        if (X_alpha_unreported[n] > 0) {
            IBNER_post_pred[n] = hurdle_gamma_rng(X_alpha_unreported[n], X_beta[n], X_p0_logit[n]);
        }
    }
}
