// estimate claim grouping given policy details.
// ie, given class, state, and rnlob, estimate model svrty and cause codes
// likelihoods.
//
// let's try a basic logistic regression using one hot encoded versions of the
// predictors. svrty and cause are modeled separately but their per-predictor
// coefficients are drawn from a shared multivariate prior, inducing
// correlation (partial pooling) between the two outcomes at the predictor level.

data {
    int N;                                         // number of claim records

    int K_class;                                   // number of distinct class codes
    int K_state;                                   // number of distinct risk states
    int K_rnlob;                                   // number of distinct RN LOB
    int K_svrty;                                   // number of distinct severity codes
    int K_cause;                                   // number of distinct causes of loss
    array[N] int<lower=1, upper=K_class> g_class;  // categorical variable: claim's class code
    array[N] int<lower=1, upper=K_state> g_state;  // categorical variable: claim's risk state
    array[N] int<lower=1, upper=K_rnlob> g_rnlob;  // categorical variable: claim's rnlob
    array[N] int<lower=1, upper=K_svrty> g_svrty;  // target categorical variable: claim's severity code
    array[N] int<lower=1, upper=K_cause> g_cause;  // target categorical variable: claim's cause of loss

    real<lower=0> correlation_regularization_strength; // for the LKJ corr factor prior
}
transformed data {
    int num_predictors = K_class + K_state + K_rnlob;
    matrix[N, num_predictors] X;
    for (n in 1:N) {
        row_vector[K_class] ohe_class = one_hot_row_vector(K_class, g_class[n]);
        row_vector[K_state] ohe_state = one_hot_row_vector(K_state, g_state[n]);
        row_vector[K_rnlob] ohe_rnlob = one_hot_row_vector(K_rnlob, g_rnlob[n]);
        X[n, :] = append_col(append_col(ohe_class, ohe_state), ohe_rnlob);
    }
    int M = K_svrty + K_cause;
}
parameters {
    // two sets of intercepts: one for svrty code and one for cause of loss code
    vector[K_svrty] alpha_svrty;
    vector[K_cause] alpha_cause;

    // non-centered joint coefficients: each predictor's (svrty, cause) loading
    // vector is drawn from a shared MVN(0, Sigma) prior, Sigma = diag(tau) Omega diag(tau)
    matrix[num_predictors, M] z;
    cholesky_factor_corr[M] L_Omega; // correlation between svrty/cause loadings
    vector<lower=0>[M] tau;          // per-dimension loading scale
}
transformed parameters {
    matrix[num_predictors, M] beta_joint = z * diag_pre_multiply(tau, L_Omega)';
    matrix[num_predictors, K_svrty] beta_svrty = beta_joint[, 1:K_svrty];
    matrix[num_predictors, K_cause] beta_cause = beta_joint[, (K_svrty + 1):M];
}
model {
    alpha_svrty ~ std_normal();
    alpha_cause ~ std_normal();

    to_vector(z) ~ std_normal();
    tau ~ exponential(1);
    L_Omega ~ lkj_corr_cholesky(correlation_regularization_strength);

    g_svrty ~ categorical_logit_glm(X, alpha_svrty, beta_svrty);
    g_cause ~ categorical_logit_glm(X, alpha_cause, beta_cause);
}
generated quantities {
    array[N] int<lower=1, upper=K_svrty> k_svrty_post_pred;
    array[N] int<lower=1, upper=K_cause> k_cause_post_pred;
    array[N, 2] int<lower=1, upper=max(K_svrty, K_cause)> k_joint_post_pred;

    vector[N] k_svrty_log_lik;
    vector[N] k_cause_log_lik;
    vector[N] k_joint_log_lik;

    matrix[N, K_svrty] logit_svrty = rep_matrix(alpha_svrty', N) + X * beta_svrty;
    matrix[N, K_cause] logit_cause = rep_matrix(alpha_cause', N) + X * beta_cause;

    for (n in 1:N) {
        k_svrty_post_pred[n] = categorical_logit_rng(logit_svrty[n]');
        k_cause_post_pred[n] = categorical_logit_rng(logit_cause[n]');
        k_svrty_log_lik[n] = categorical_logit_lpmf(g_svrty[n] | logit_svrty[n]');
        k_cause_log_lik[n] = categorical_logit_lpmf(g_cause[n] | logit_cause[n]');
    }

    // correlation comes from the shared beta_joint prior, so outcomes are
    // conditionally independent given beta_joint (no extra latent term to sum over)
    k_joint_post_pred[, 1] = k_svrty_post_pred;
    k_joint_post_pred[, 2] = k_cause_post_pred;
    k_joint_log_lik = k_svrty_log_lik + k_cause_log_lik;
}
