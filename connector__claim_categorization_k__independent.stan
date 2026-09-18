// estimate claim grouping given policy details.
// ie, given class, state, and rnlob, estimate model svrty and cause codes
// likelihoods.
//
// let's try a basic logistic regression using one hot encoded versions of the
// predictors. this will treat svrty and cause as independent, which we'll
// compare with alternative models that model them as dependent variables.

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
    // true target: joint distribution
    array[N, 2] int<lower=1, upper=max(K_svrty, K_cause)> g_joint;
    g_joint[, 1] = g_svrty;
    g_joint[, 2] = g_cause;

    // although it won't be used in the independent model, let's also place the "concatenated"
    // target here. this will be a new 1-d array that maps the known combos of the individual
    // target to new integer groups.
    int K_joint_max = K_svrty * K_cause;
    array[N] int<lower=1, upper=K_joint_max> g_joint_1d_extra; // "extra" bc has more options than necessary
    int K_joint_counter = 0;
    array[K_joint_max] int hash_to_joint_code_lookup = zeros_int_array(K_joint_max);
    for (n in 1:N) {
        int hash = (g_svrty[n] - 1) * K_cause + g_cause[n];
        if (hash_to_joint_code_lookup[hash] == 0) {
            // code combination not previously found
            K_joint_counter += 1;
            hash_to_joint_code_lookup[hash] = K_joint_counter;
        }
        g_joint_1d_extra[n] = hash_to_joint_code_lookup[hash];
    }
    int<upper=K_joint_max> K_joint = K_joint_counter;
    array[N] int<lower=1, upper=K_joint> g_joint_1d = g_joint_1d_extra; // no longer "extra": only has the appropriate K_joint options

    // recover reverse mapping
    array[K_joint, 2] int joint_code_to_original_groups;
    int K_joint_counter_2 = 0;
    for (gz in 1:K_svrty) {
        for (gc in 1:K_cause) {
            int hash = (gz - 1) * K_cause + gc;
            int joint_code = hash_to_joint_code_lookup[hash];
            if (joint_code > 0) {
                joint_code_to_original_groups[joint_code, ] = {gz, gc};
                K_joint_counter_2 += 1;
            }
        }
    }
    if (K_joint_counter_2 != K_joint_counter) {
        reject("Unable to recreate mapping of joint code to original codes.");
    }
}
parameters {
    // two sets: one for svrty code and one for cause of loss code
    vector[K_svrty] alpha_svrty;
    vector[K_cause] alpha_cause;

    matrix[num_predictors, K_svrty] beta_svrty;
    matrix[num_predictors, K_cause] beta_cause;
}
model {
    alpha_svrty ~ std_normal();
    alpha_cause ~ std_normal();
    to_vector(beta_svrty) ~ std_normal();
    to_vector(beta_cause) ~ std_normal();

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

    // the true target: assuming independence
    k_joint_post_pred[, 1] = k_svrty_post_pred;
    k_joint_post_pred[, 2] = k_cause_post_pred;
    k_joint_log_lik = k_svrty_log_lik + k_cause_log_lik;
}
