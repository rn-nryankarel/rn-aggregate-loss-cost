functions {
    real double_lomax_lpdf(real y, real p_plus, real lambda_plus, real alpha_plus, real lambda_minus, real alpha_minus) {
        if (y > 0) {
            return log(p_plus) + pareto_type_2_lpdf(y | 0, lambda_plus, alpha_plus);
        } else {
            return log1m(p_plus) + pareto_type_2_lpdf(-y | 0, lambda_minus, alpha_minus);
        }
    }
    real double_lomax_lpdf(
        vector y,
        real p_plus,
        real lambda_plus,
        real alpha_plus,
        real lambda_minus,
        real alpha_minus
    ) {
        // Exactly 1 for y > 0, and 0 otherwise, including y == 0.
        vector[num_elements(y)] positive =
            ceil(fmin(fmax(y, 0.0), 1.0));
        vector[num_elements(y)] negative = 1.0 - positive;

        vector[num_elements(y)] scale =
            lambda_plus * positive + lambda_minus * negative;
        vector[num_elements(y)] shape =
            alpha_plus * positive + alpha_minus * negative;

        real n_positive = sum(positive);
        real n_negative = num_elements(y) - n_positive;
        real lp = pareto_type_2_lpdf(abs(y) | 0, scale, shape);

        if (n_positive > 0) {
            lp += n_positive * log(p_plus);
        }
        if (n_negative > 0) {
            lp += n_negative * log1m(p_plus);
        }
        return lp;
    }
    real double_lomax_lpdf(
        vector y,
        real p_plus,
        real lambda_plus,
        real alpha_plus,
        real lambda_minus,
        real alpha_minus,
        vector positive
    ) {
        // with this signature, you provide the positive 1 vs. 0 vector so it doesn't need
        // to be recomputed each time
        vector[num_elements(y)] negative = 1.0 - positive;

        vector[num_elements(y)] scale =
            lambda_plus * positive + lambda_minus * negative;
        vector[num_elements(y)] shape =
            alpha_plus * positive + alpha_minus * negative;

        real n_positive = sum(positive);
        real n_negative = num_elements(y) - n_positive;
        real lp = pareto_type_2_lpdf(abs(y) | 0, scale, shape);

        if (n_positive > 0) {
            lp += n_positive * log(p_plus);
        }
        if (n_negative > 0) {
            lp += n_negative * log1m(p_plus);
        }
        return lp;
    }
    real double_lomax_rng(real p_plus, real lambda_plus, real alpha_plus, real lambda_minus, real alpha_minus) {
        int positive = bernoulli_rng(p_plus);
        if (positive) {
            return pareto_type_2_rng(0, lambda_plus, alpha_plus);
        } else {
            return -pareto_type_2_rng(0, lambda_minus, alpha_minus);
        }
    }
}
data {
    int<lower=1> C;                           // number of claims
    int<lower=0> N;                           // number of jump transactions
    vector<lower=-1>[C] t_settled;            // -1 = null
    vector<lower=0>[C] t_trunc;               // truncation point (evaluation - report)

    array[N] int<lower=1, upper=C> idx_claim; // jump transaction's claim
    vector<lower=0>[N] t;                     // jump transaction's time since claim report
    vector[N] x_t;                            // jump transaction's incremental incurred

    int<lower=0> C_out;                       // out of sample predictions; 0 disables prediction
    vector<lower=-1>[C_out] t_settled_out;    // retained for input compatibility; not used in prediction
    vector<lower=0>[C_out] t_trunc_out;       // truncation point (evaluation - report)

    real posterior_predictive_grain;          // inference: for discretization
}
transformed data {
    vector[C] t_max;
    if (!(posterior_predictive_grain > 0) || is_inf(posterior_predictive_grain)) {
        reject("posterior_predictive_grain must be positive and finite.");
    }
    for (c in 1:C) {
        if (is_inf(t_trunc[c])) {
            reject("t_trunc must be finite; claim ", c);
        }
        if (t_settled[c] != -1
            && (t_settled[c] < 0 || t_settled[c] > t_trunc[c])) {
            reject("Settlement must be -1 or within [0, t_trunc]; claim ", c);
        }
        t_max[c] = t_settled[c] == -1 ? t_trunc[c] : t_settled[c];
    }

    array[C] int n_jumps = zeros_int_array(C);
    for (n in 1:N) {
        // Equality allows transactions and settlement recorded at the same rounded time.
        if (t[n] > t_max[idx_claim[n]]) {
            reject("Transaction after settlement or evaluation; transaction ", n);
        }
        if (t_max[idx_claim[n]] == 0) {
            reject("Transaction with zero open exposure; transaction ", n);
        }
        n_jumps[idx_claim[n]] += 1;
    }

    vector[N] positive_incurred = ceil(fmin(fmax(x_t, 0.0), 1.0));

    // parameters that govern posterior predictive time series output
    vector[C_out] t_max_out = t_trunc_out;
    for (c in 1:C_out) {
        if (is_inf(t_trunc_out[c])) {
            reject("t_trunc_out must be finite; claim ", c);
        }
    }
    int projection_width = C_out > 0
        ? to_int(ceil(max(t_max_out) / posterior_predictive_grain))
        : 0;
}
parameters {
    // Constant transaction intensity while open, zero after settlement.
    real log_lambda;
    real log_settlement_rate;

    real<lower=0, upper=1> p_plus;
    real<lower=0> lambda_plus;
    real<lower=0> lambda_minus;
    // Shapes above one ensure finite conditional means on both sides.
    real<lower=1> alpha_plus;
    real<lower=1> alpha_minus;
}
transformed parameters {
    real<lower=0> lambda = exp(log_lambda);
    real<lower=0> settlement_rate = exp(log_settlement_rate);
}
model {
    log_lambda ~ normal(0.8, 1); // of completed series, about 2 transactions / year
    log_settlement_rate ~ normal(0, 1); // rate per unit of t; prior median rate is one

    for (c in 1:C) {
        if (t_settled[c] == -1) {
            target += exponential_lccdf(t_trunc[c] | settlement_rate);
        } else {
            target += exponential_lpdf(t_settled[c] | settlement_rate);
        }
    }

    p_plus ~ beta(1, 1);
    lambda_plus ~ normal(0, 100);
    lambda_minus ~ normal(0, 100);
    alpha_plus ~ lognormal(0, 1);
    alpha_minus ~ lognormal(0, 1);

    // Integrated transaction intensity uses only exposure before settlement.
    target += sum(n_jumps) * log_lambda - sum(t_max) * lambda;

    // now let's bring on the mark portion
    x_t ~ double_lomax(
        p_plus,
        lambda_plus,
        alpha_plus,
        lambda_minus,
        alpha_minus,
        positive_incurred
    );
}
generated quantities {
    // for fear of needing ragged arrays, we'll sample the variable counts and
    // continuously placed points and corresponding incurreds, then create a
    // discrete time series at grain posterior_predictive_grain that shows the
    // discretized incurred path
    matrix[C_out, projection_width] X;
    vector[C_out] t_settled_post_pred;

    // Each new path gets an independent settlement time, even beyond its output horizon.
    array[C_out] int n_jumps_out = zeros_int_array(C_out);

    for (c in 1:C_out) {
        real open_exposure;
        t_settled_post_pred[c] = exponential_rng(settlement_rate);
        open_exposure = fmin(t_max_out[c], t_settled_post_pred[c]);
        n_jumps_out[c] = poisson_rng(lambda * open_exposure);
        int n_jumps_c = n_jumps_out[c];
        row_vector[projection_width] X_c_incremental = zeros_row_vector(projection_width);
        for (i in 1:n_jumps_c) {
            real location = uniform_rng(0, open_exposure);
            // then sample x_t, the incurred amount
            real incurred = double_lomax_rng(
                p_plus,
                lambda_plus,
                alpha_plus,
                lambda_minus,
                alpha_minus
            );
            int discretized_location = max(
                1,
                to_int(ceil(location / posterior_predictive_grain))
            );
            X_c_incremental[discretized_location] += incurred;
        }
        X[c] = cumulative_sum(X_c_incremental);
    }
}