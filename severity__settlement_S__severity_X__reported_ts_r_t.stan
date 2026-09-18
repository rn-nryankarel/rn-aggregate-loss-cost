functions {
    vector hurdle_lognormal_lpdf_elemwise(vector x, vector p0, vector mu, vector sigma) {
        vector[size(x)] lp;
        for (n in 1:size(x)) {
            real p0_n = p0[n];
            real mu_n = mu[n];
            real sigma_n = sigma[n];
            if (x[n] == 0) {
                lp[n] = log(p0_n);
            } else {
                lp[n] = log1m(p0_n) + lognormal_lpdf(x[n] | mu_n, sigma_n);
            }
        }
        return lp;
    }
    vector hurdle_lognormal_lpdf_elemwise(vector x, real p0, real mu, real sigma) {
        int n_obs = size(x);
        return hurdle_lognormal_lpdf_elemwise(x, rep_vector(p0, n_obs), rep_vector(mu, n_obs), rep_vector(sigma, n_obs));
    }
    real hurdle_lognormal_lpdf(vector x, vector p0, vector mu, vector sigma) {
        return sum(hurdle_lognormal_lpdf_elemwise(x, p0, mu, sigma));
    }
    real hurdle_lognormal_lpdf(vector x, real p0, real mu, real sigma) {
        int n_obs = size(x);
        return hurdle_lognormal_lpdf(x | rep_vector(p0, n_obs), rep_vector(mu, n_obs), rep_vector(sigma, n_obs));
    }
    int count_positives(array[] int x) {
        int pos = 0;
        for (xi in x) {
            if (xi > 0) pos += 1;
        }
        return pos;
    }

    real development_pattern_g(real t, real settlement_S) {
        if (t < 0) {
            reject("Negative t not permissible.");
        }
        return fmin(t, settlement_S) / settlement_S;
    }
    real incremental_g(real t, real settlement_S) {
        return development_pattern_g(t, settlement_S) - development_pattern_g(t - 1, settlement_S);
    }
}
data {
    int N;                              // number of claim records
    int K;                                     // max age
    matrix[N, K] Inc;                   // "ragged" matrix of loss incurred
    matrix[N, K] Paid;                  // "ragged" matrix of loss paid

    array[N] int<lower=-1, upper=K> s;  // settlement lag 1-K. -1 means not yet settled. 0 shouldn't be present.
    array[N] int<lower=-1, upper=K> c;  // closure lag 1-K. -1 means not yet closed. 0 shouldn't be present.
    array[N] int<lower=1, upper=K> age; // age of claim as of evaluation date - grows after settlement

    // don't know if this is necessary at this point, but it feels important
    // to pass in the data in matrix form and then melt it to long form. I
    // still need to validate it against my observed paids and incurreds,
    // though, so I'm considering passing in my python-computed long-form
    // paids and incurreds to validate that the two series align.
    int expected_N_long;
    array[expected_N_long] real expected_long_paid;
    array[expected_N_long] real expected_long_incurred;
}
transformed data {
    // identify records for which we have (near) full data;
    // ie claims which have been marked as "settled"
    int N_settled = count_positives(s);
    int N_not_settled = N - N_settled;
    vector[N_settled] X_known;
    array[N_settled] int<lower=1, upper=K> S_known;
    int idx_settled = 0;
    int idx_not_settled = 0;
    array[N] int<lower=0, upper=1> ind_settled;
    array[N_settled] int<lower=1, upper=N> ii_obs;
    array[N_not_settled] int<lower=1, upper=N> ii_mis;
    array[N] int<lower=-1, upper=N_settled> settled_vector_lookup = rep_array(-1, N);
    for (n in 1:N) {
        if (c[n] == 0) {
            reject("Unexpected c[n] == 0 at n=", n);
        }
        if (s[n] == 0) {
            reject("Unexpected s[n] == 0 at n=", n);
        }
        else if (s[n] == -1) {
            ind_settled[n] = 0;
            idx_not_settled += 1;
            ii_mis[idx_not_settled] = n;
        } else {
            ind_settled[n] = 1;
            idx_settled += 1;
            ii_obs[idx_settled] = n;
            X_known[idx_settled] = sum(Paid[n, 1:s[n]]);
            S_known[idx_settled] = s[n];
            settled_vector_lookup[n] = idx_settled;
        }
    }

    // turn data into long form
    // N * K less unapplicable cells
    // this includes post-settlement evaluations of a claim up to its age
    int N_long = 0;
    for (n in 1:N) {
        N_long += age[n];
    }
    array[N_long] int<lower=1, upper=K> t;                // time index for claim
    array[N_long] int<lower=0, upper=1> s_state;          // boolean indicating whether claim was settled or not by this point
    array[N_long] int<lower=0, upper=1> c_state;          // boolean indicating whether claim file was closed or not by this point
    array[N_long] int<lower=1, upper=N> claim;     // integer tracking claim groups
    array[N_long] real paid;                              // incremental paid
    array[N_long] real incurred;                          // incremental incurred
    int i = 0; // counter to iterate through every value of N_long
    for (n in 1:N) {
        int age_n = age[n];
        for (k in 1:K) {
            if (k > age_n) {
                break;
            }
            i += 1;
            t[i] = k;
            s_state[i] = s[n] > 0 && k >= s[n];
            c_state[i] = c[n] > 0 && k >= c[n];
            claim[i] = n;
            paid[i] = Paid[n, k];
            incurred[i] = Inc[n, k];
        }
    }
    if (i != N_long) {
        reject("Unexpected number of iterations to create long-form data.");
    }
    // validate that the long form looks as expected
    if (N_long != expected_N_long) {
        reject("N_long=", N_long, " but expected_N_long=", expected_N_long);
    }
    for (n in 1:N_long) {
        if (abs(paid[n] - expected_long_paid[n]) > 0.01) {
            reject("n=", n, ": paid[n]=", paid[n], " but expected_long_paid[n]=", expected_long_paid[n]);
        }
        if (abs(incurred[n] - expected_long_incurred[n]) > 0.01) {
            reject("n=", n, ": incurred[n]=", incurred[n], " but expected_long_incurred[n]=", expected_long_incurred[n]);
        }
    }
}
parameters {
    // model settlement lag as survival analysis
    real<lower=0> S_weibull_shape;
    real<lower=0> S_weibull_scale;

    // ultimate severity upon settlement
    real<lower=0, upper=1> X_hurdle_p0;
    real X_lognorm_mu;
    real<lower=0> X_lognorm_sigma;

    // incremental incurred time series { r_t }
    // for my initial g = uniform, I may not need any additional parameters for
    // this particular development pattern, but we will still need parameters
    // for the likelihood of each r in the series.
    // will also differentiate between pre-S and post-S
    // no mu for ante-S: this will be gleaned from X
    // real<lower=0> r_ante_S_student_t_sigma;
    // real<lower=0> r_post_S_student_t_sigma;
    // real<lower=0> r_post_S_student_t_mu;

    // treat the unknown X and S as latent parameters
    // S must be continuous because it's a parameter.
    // subtracting age from this base parameter so we can
    // have varying lower bound based on age in the true
    // S_unknown array
    vector<lower=0>[N_not_settled] S_unknown_minus_age;
    vector<lower=0>[N_not_settled] X_unknown;
}
transformed parameters {
    // will have r's student t nu hardcoded here
    // real<lower=0> r_ante_S_student_t_nu = 5;
    // real<lower=0> r_post_S_student_t_nu = 25;

    // full X and S = known and unknown
    vector<lower=0>[N] X;
    vector<lower=0>[N] S;
    X[ii_obs] = X_known;
    X[ii_mis] = X_unknown;
    S[ii_obs] = to_vector(S_known);
    S[ii_mis] = S_unknown_minus_age + to_vector(age[ii_mis]); // this looks weird, but its forcing the correct lower constraint on S

    // reported incurred time series parameters
    // vector<lower=0>[N_long] r_student_t_nu;
    // vector[N_long] r_student_t_mu;
    // vector<lower=0>[N_long] r_student_t_sigma;
    // for (n in 1:N_long) {
    //     int t_n = t[n];
    //     int claim_n = claim[n];
    //     int pre_settle = t_n <= S[claim_n];
    //     real r_t_mean = incremental_g(t_n, S[claim_n]) * X[claim_n];
    //     r_student_t_mu[n] = pre_settle ? r_t_mean : r_post_S_student_t_mu;
    //     r_student_t_nu[n] = pre_settle ? r_ante_S_student_t_nu : r_post_S_student_t_nu;
    //     r_student_t_sigma[n] = pre_settle ? r_ante_S_student_t_sigma : r_post_S_student_t_sigma;
    // }
}
model {
    // priors
    S_weibull_scale ~ normal(4, 2.5);
    S_weibull_shape ~ normal(1, 0.2);

    X_hurdle_p0 ~ beta(1, 4);
    X_lognorm_mu ~ normal(6, 3);
    X_lognorm_sigma ~ normal(1, 1);

    // r_ante_S_student_t_sigma ~ normal(10, 20);
    // r_post_S_student_t_sigma ~ normal(1, 3);
    // r_post_S_student_t_mu ~ normal(0, 1);

    // likelihoods

    // settlement lag
    S ~ weibull(S_weibull_shape, S_weibull_scale);

    // settlement
    X ~ hurdle_lognormal(X_hurdle_p0, X_lognorm_mu, X_lognorm_sigma);

    // reported incurred time series
    // incurred ~ student_t(r_student_t_nu, r_student_t_mu, r_student_t_sigma);
}
generated quantities {
    // array[N_long] real incurred_post_pred = student_t_rng(
    //     r_student_t_nu,
    //     r_student_t_mu,
    //     r_student_t_sigma
    // );
    // vector[N_long] incurred_log_lik;
    // for (n in 1:N_long) {
    //     incurred_log_lik[n] = student_t_lpdf(
    //         incurred[n] |
    //         r_student_t_nu[n],
    //         r_student_t_mu[n],
    //         r_student_t_sigma[n]
    //     );
    // }
}