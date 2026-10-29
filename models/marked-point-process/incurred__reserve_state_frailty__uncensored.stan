// Reserve-state marked point process for cumulative incurred.
//
// Each claim alternates between two states:
//   1 = unreserved (cumulative incurred is zero): the next transaction sets a
//       positive lognormal reserve;
//   2 = reserved (cumulative incurred L > 0): the next transaction either
//       closes the reserve to exactly zero, with logit probability linear in
//       log(L), or moves it to log(L_new) = log(L) + a + b * z + Laplace error,
//       where z is centered log(L) and b < 0 gives mean reversion.
// Transaction intensity in state s at claim age u is
//   frailty_c * exp(log_lambda[s] - beta[s] * u),
// and is zero after settlement. The claim frailty ~ gamma(phi, phi) is
// integrated out of the likelihood. Settlement time is Weibull.
functions {
    // Integral of exp(-beta * u) over [a, b].
    real decay_integral(real beta, real a, real b) {
        real w = b - a;
        if (w <= 0) {
            return 0;
        }
        if (abs(beta * w) < 1e-6) {
            return exp(-beta * a) * w * (1 - 0.5 * beta * w);
        }
        return -exp(-beta * a) * expm1(-beta * w) / beta;
    }

    // Next event after time s for intensity exp(log_rate - beta * u); +inf if none.
    real next_event_time_rng(real s, real log_rate, real beta) {
        // Solve integral_s^tau exp(log_rate - beta * u) du = E, E ~ exponential(1).
        real q = exp(log(exponential_rng(1)) - log_rate + beta * s);
        real bq = beta * q;
        if (is_inf(q)) {
            return positive_infinity();
        }
        if (abs(bq) < 1e-6) {
            return s + q * (1 + 0.5 * bq);
        }
        if (bq >= 1) {
            return positive_infinity();
        }
        return s - log1m(bq) / beta;
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
    // Relative tolerance for treating a revision as closing the reserve to zero.
    real closure_tolerance = 1e-9;

    if (!(posterior_predictive_grain > 0) || is_inf(posterior_predictive_grain)) {
        reject("posterior_predictive_grain must be positive and finite.");
    }

    vector[C] t_max;
    int n_settled = 0;
    for (c in 1:C) {
        if (is_inf(t_trunc[c])) {
            reject("t_trunc must be finite; claim ", c);
        }
        if (t_settled[c] != -1
            && (t_settled[c] <= 0 || t_settled[c] > t_trunc[c])) {
            reject("Settlement must be -1 or within (0, t_trunc]; claim ", c);
        }
        t_max[c] = t_settled[c] == -1 ? t_trunc[c] : t_settled[c];
        n_settled += t_settled[c] != -1;
    }
    vector[n_settled] settled_times;
    vector[C - n_settled] censored_times;
    {
        int i_settled = 1;
        int i_censored = 1;
        for (c in 1:C) {
            if (t_settled[c] == -1) {
                censored_times[i_censored] = t_trunc[c];
                i_censored += 1;
            } else {
                settled_times[i_settled] = t_settled[c];
                i_settled += 1;
            }
        }
    }

    array[C] int n_jumps = zeros_int_array(C);
    for (n in 1:N) {
        if (is_inf(t[n]) || is_nan(x_t[n]) || is_inf(x_t[n])) {
            reject("Transaction times and amounts must be finite; transaction ", n);
        }
        // Equality allows transactions and settlement recorded at the same rounded time.
        if (t[n] > t_max[idx_claim[n]]) {
            reject("Transaction after settlement or evaluation; transaction ", n);
        }
        n_jumps[idx_claim[n]] += 1;
    }

    // Order transactions by claim, then by time; ties keep input order.
    array[C] int claim_start;
    array[N] int order;
    {
        array[C] int filled = zeros_int_array(C);
        claim_start[1] = 1;
        for (c in 2:C) {
            claim_start[c] = claim_start[c - 1] + n_jumps[c - 1];
        }
        for (n in 1:N) {
            int c = idx_claim[n];
            order[claim_start[c] + filled[c]] = n;
            filled[c] += 1;
        }
        for (c in 1:C) {
            for (i in (claim_start[c] + 1):(claim_start[c] + n_jumps[c] - 1)) {
                int key = order[i];
                int j = i - 1;
                while (j >= claim_start[c] && t[order[j]] > t[key]) {
                    order[j + 1] = order[j];
                    j -= 1;
                }
                order[j + 1] = key;
            }
        }
    }

    // Replay each claim's reserve path to classify transactions and exposure.
    int S = N + C;                            // each claim has n_jumps + 1 exposure segments
    array[S] int segment_claim;
    array[S] int segment_state;
    vector[S] segment_start;
    vector[S] segment_end;
    array[N] int event_state;
    vector[N] event_time;

    int N_initial = 0;
    int N_revision = 0;
    array[N] int is_initial;
    array[N] int is_closure = zeros_int_array(N);
    vector[N] log_level_before = zeros_vector(N);
    vector[N] log_level_after = zeros_vector(N);
    real mark_jacobian = 0;
    {
        int s = 1;
        for (c in 1:C) {
            real level = 0;
            real previous_time = 0;
            for (i in claim_start[c]:(claim_start[c] + n_jumps[c] - 1)) {
                int n = order[i];
                int state = level > 0 ? 2 : 1;
                segment_claim[s] = c;
                segment_state[s] = state;
                segment_start[s] = previous_time;
                segment_end[s] = t[n];
                s += 1;

                event_state[i] = state;
                event_time[i] = t[n];
                if (state == 1) {
                    if (!(x_t[n] > 0)) {
                        reject(
                            "A transaction from zero cumulative incurred must be positive; transaction ",
                            n
                        );
                    }
                    is_initial[i] = 1;
                    N_initial += 1;
                    level = x_t[n];
                } else {
                    real relative_level = (level + x_t[n]) / level;
                    is_initial[i] = 0;
                    N_revision += 1;
                    log_level_before[i] = log(level);
                    if (relative_level < -closure_tolerance) {
                        reject(
                            "Cumulative incurred must stay non-negative; transaction ",
                            n
                        );
                    }
                    if (relative_level <= closure_tolerance) {
                        is_closure[i] = 1;
                        level = 0;
                    } else {
                        log_level_after[i] = log(level + x_t[n]);
                        // The model is on log(L + x), so log |d log(L + x) / dx| = -log(L + x).
                        mark_jacobian -= log_level_after[i];
                        level += x_t[n];
                    }
                }
                previous_time = t[n];
            }
            segment_claim[s] = c;
            segment_state[s] = level > 0 ? 2 : 1;
            segment_start[s] = previous_time;
            segment_end[s] = t_max[c];
            s += 1;
        }
    }

    int N_closure = sum(is_closure);
    int N_continue = N_revision - N_closure;

    vector[N_initial] initial_reserve;
    array[N_revision] int revision_closes;
    vector[N_revision] revision_log_level;
    vector[N_continue] continue_log_level;
    vector[N_continue] continue_log_new_level;
    {
        int i_initial = 1;
        int i_revision = 1;
        int i_continue = 1;
        for (i in 1:N) {
            if (is_initial[i]) {
                initial_reserve[i_initial] = x_t[order[i]];
                i_initial += 1;
            } else {
                revision_closes[i_revision] = is_closure[i];
                revision_log_level[i_revision] = log_level_before[i];
                i_revision += 1;
                if (!is_closure[i]) {
                    continue_log_level[i_continue] = log_level_before[i];
                    continue_log_new_level[i_continue] = log_level_after[i];
                    i_continue += 1;
                }
            }
        }
    }
    // Revision covariates use log reserve level centered at its training mean.
    real log_level_center = N_revision > 0 ? mean(revision_log_level) : 0;
    vector[N_revision] revision_z = revision_log_level - log_level_center;
    vector[N_continue] continue_z = continue_log_level - log_level_center;
    vector[C] n_jumps_vector = to_vector(n_jumps);

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
    // Intensity by state: 1 = unreserved, 2 = reserved.
    vector[2] log_lambda;
    vector[2] beta;                          // decay rate in claim age; negative means increasing
    real log_phi;                            // gamma frailty shape and rate; variance is 1 / phi

    real log_settlement_shape;
    real log_settlement_scale;

    // Initial (or reopening) reserve size: lognormal.
    real initial_log_mean;
    real<lower=0> initial_log_sd;

    // Revisions while reserved, as functions of z = log(L) - log_level_center.
    real closure_intercept;                  // logit P(closure to zero)
    real closure_slope;
    real revision_intercept;                 // log(new L) = log(L) + intercept + slope * z + e
    real revision_slope;                     // negative values mean reversion
    // Laplace scale below one keeps E[exp(e)] finite.
    real<lower=0, upper=1> revision_scale;
}
transformed parameters {
    real<lower=0> phi = exp(log_phi);
    real<lower=0> settlement_shape = exp(log_settlement_shape);
    real<lower=0> settlement_scale = exp(log_settlement_scale);
}
model {
    log_lambda ~ normal(0, 1.5);
    beta ~ normal(0, 2);
    log_phi ~ normal(0, 1.5);
    log_settlement_shape ~ normal(0, 1);
    log_settlement_scale ~ normal(0, 1);
    initial_log_mean ~ normal(log(20), 2);
    initial_log_sd ~ normal(0, 2);
    closure_intercept ~ normal(-2, 1.5);
    closure_slope ~ normal(0, 1);
    revision_intercept ~ normal(0, 1);
    revision_slope ~ normal(0, 0.5);
    revision_scale ~ normal(0, 1);

    // Settlement, right-censored at t_trunc for open claims.
    target += weibull_lpdf(settled_times | settlement_shape, settlement_scale);
    target += weibull_lccdf(censored_times | settlement_shape, settlement_scale);

    // Event-time intensities, then the gamma-frailty-integrated compensator.
    if (N > 0) {
        target += sum(log_lambda[event_state] - beta[event_state] .* event_time);
    }
    {
        vector[C] compensator = zeros_vector(C);
        vector[2] lambda = exp(log_lambda);
        for (s in 1:S) {
            int state = segment_state[s];
            compensator[segment_claim[s]] += lambda[state] * decay_integral(
                beta[state],
                segment_start[s],
                segment_end[s]
            );
        }
        target += sum(lgamma(phi + n_jumps_vector)) - C * lgamma(phi)
            + C * phi * log_phi
            - dot_product(phi + n_jumps_vector, log(phi + compensator));
    }

    // Marks.
    target += lognormal_lpdf(initial_reserve | initial_log_mean, initial_log_sd);
    if (N_revision > 0) {
        target += bernoulli_logit_lpmf(
            revision_closes | closure_intercept + closure_slope * revision_z
        );
    }
    if (N_continue > 0) {
        target += double_exponential_lpdf(
            continue_log_new_level
            | continue_log_level + revision_intercept + revision_slope * continue_z,
            revision_scale
        );
    }
    target += mark_jacobian;
}
generated quantities {
    matrix[C_out, projection_width] X;
    vector[C_out] t_settled_post_pred;
    vector[C_out] frailty_post_pred;
    array[C_out] int n_jumps_out = zeros_int_array(C_out);

    for (c in 1:C_out) {
        row_vector[projection_width] X_c_incremental = zeros_row_vector(projection_width);
        real level = 0;
        real now = 0;
        real horizon;

        // Each new path gets an independent settlement time, even beyond its output horizon.
        t_settled_post_pred[c] = weibull_rng(settlement_shape, settlement_scale);
        frailty_post_pred[c] = gamma_rng(phi, phi);
        horizon = fmin(t_max_out[c], t_settled_post_pred[c]);

        while (frailty_post_pred[c] > 0) {
            int state = level > 0 ? 2 : 1;
            real increment;
            now = next_event_time_rng(
                now,
                log_lambda[state] + log(frailty_post_pred[c]),
                beta[state]
            );
            if (now > horizon) {
                break;
            }
            if (state == 1) {
                increment = lognormal_rng(initial_log_mean, initial_log_sd);
            } else {
                real z = log(level) - log_level_center;
                if (bernoulli_logit_rng(closure_intercept + closure_slope * z)) {
                    increment = -level;
                } else {
                    increment = level * expm1(double_exponential_rng(
                        revision_intercept + revision_slope * z,
                        revision_scale
                    ));
                }
            }
            level += increment;
            n_jumps_out[c] += 1;
            if (n_jumps_out[c] > 100000) {
                reject("More than 100000 simulated transactions for output claim ", c);
            }
            X_c_incremental[max(1, to_int(ceil(now / posterior_predictive_grain)))] += increment;
        }
        X[c] = cumulative_sum(X_c_incremental);
    }
}
