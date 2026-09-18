functions {

    // hurdled probability statements
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

    vector hurdle_gumbel_lpdf_elemwise(vector x, vector p0, vector mu, vector beta) {
        vector[size(x)] lp;
        for (n in 1:size(x)) {
            real p0_n = p0[n];
            real mu_n = mu[n];
            real beta_n = beta[n];
            if (x[n] == 0) {
                lp[n] = log(p0_n);
            } else {
                lp[n] = log1m(p0_n) + gumbel_lpdf(x[n] | mu_n, beta_n);
            }
        }
        return lp;
    }
    vector hurdle_gumbel_lpdf_elemwise(vector x, real p0, real mu, real beta) {
        int n_obs = size(x);
        return hurdle_gumbel_lpdf_elemwise(x, rep_vector(p0, n_obs), rep_vector(mu, n_obs), rep_vector(beta, n_obs));
    }
    real hurdle_gumbel_lpdf(vector x, vector p0, vector mu, vector beta) {
        return sum(hurdle_gumbel_lpdf_elemwise(x, p0, mu, beta));
    }
    real hurdle_gumbel_lpdf(vector x, real p0, real mu, real beta) {
        int n_obs = size(x);
        return hurdle_gumbel_lpdf(x | rep_vector(p0, n_obs), rep_vector(mu, n_obs), rep_vector(beta, n_obs));
    }
    real hurdle_gumbel_rng(real p0, real mu, real beta) {
        if (bernoulli_rng(p0) == 1) {
            return 0;
        }
        return gumbel_rng(mu, beta);
    }

    int count_positives(array[] int x) {
        int pos = 0;
        for (xi in x) {
            if (xi > 0) pos += 1;
        }
        return pos;
    }

    vector decayed_beta(vector t, real min_beta, real beta_range, real decay_rate) {
        vector[size(t)] decay_factor = exp(-decay_rate * (t - 1));
        return min_beta + beta_range * decay_factor;
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
    real eulers_constant = -digamma(1);

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
    vector[N_long] paid;                              // incremental paid
    vector[N_long] incurred;                          // incremental incurred
    vector[N_long] prior_cumulative_incurred;         // cumulative incurred up to k-1
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
            prior_cumulative_incurred[i] = (
                k == 1
                ? 0
                : sum(Inc[n, 1:(k-1)])
            );
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
    real<lower=0> S_exponential_rate;

    // ultimate severity upon settlement
    real<lower=0, upper=1> X_hurdle_p0;
    real X_lognorm_mu;
    real<lower=0> X_lognorm_sigma;

    // incurred time series { r_t }
    real<lower=0, upper=1> r_hurdle_p0;
    // real<lower=0> r_min_beta;
    // real<lower=0> r_beta_range;
    // real<lower=0> r_beta_decay_rate;
    real<lower=0> r_gumbel_beta;

    // treat the unknown X and S as latent parameters
    // S must be continuous because it's a parameter.
    // subtracting age from this base parameter so we can
    // have varying lower bound based on age in the true
    // S_unknown array
    vector<lower=0>[N_not_settled] S_unknown_minus_age;
    vector[N_not_settled] z_X_unknown;
}
transformed parameters {
    // full X and S = known and unknown
    vector<lower=0>[N] X;
    vector<lower=0>[N] S;
    X[ii_obs] = X_known;
    X[ii_mis] = exp(X_lognorm_mu + X_lognorm_sigma * z_X_unknown);
    S[ii_obs] = to_vector(S_known);
    S[ii_mis] = S_unknown_minus_age + to_vector(age[ii_mis]);

    // next define submartingale increment parameters
    // vector<lower=0>[N_long] r_gumbel_beta = decayed_beta(to_vector(t), r_min_beta, r_beta_range, r_beta_decay_rate);
    vector[N_long] r_gumbel_mu = X[claim] - prior_cumulative_incurred - r_gumbel_beta * eulers_constant;
}
model {
    // priors
    S_exponential_rate ~ lognormal(log(0.25), 1);

    X_hurdle_p0 ~ beta(1, 4);
    X_lognorm_mu ~ normal(6, 3);
    X_lognorm_sigma ~ normal(1, 1);

    r_hurdle_p0 ~ beta(2, 2);
    // r_min_beta ~ lognormal(log(10), 1);
    // r_beta_range ~ lognormal(log(100), 1);
    // r_beta_decay_rate ~ lognormal(log(0.1), 1);
    r_gumbel_beta ~ lognormal(log(100), 1);

    // likelihoods

    // settlement lag
    S_known ~ exponential(S_exponential_rate);
    S_unknown_minus_age ~ exponential(S_exponential_rate);

    // settlement
    z_X_unknown ~ std_normal();
    X_known ~ hurdle_lognormal(X_hurdle_p0, X_lognorm_mu, X_lognorm_sigma);

    // reported incurred time series
    incurred ~ hurdle_gumbel(rep_vector(r_hurdle_p0, N_long), r_gumbel_mu, rep_vector(r_gumbel_beta, N_long));
}
generated quantities {
    array[N_long] real r_post_pred;
    vector[N_long] r_log_lik;
    for (n in 1:N_long) {
        r_post_pred[n] = hurdle_gumbel_rng(r_hurdle_p0, r_gumbel_mu[n], r_gumbel_beta);
        r_log_lik[n] = hurdle_gumbel_lpdf(
            [incurred[n]]' | r_hurdle_p0, r_gumbel_mu[n], r_gumbel_beta
        );
    }
}
