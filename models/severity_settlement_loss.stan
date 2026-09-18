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
}
data {
    int N;                              // number of claim records
    int K;                              // max age
    matrix[N, K] Inc;                   // "ragged" matrix of loss incurred
    matrix[N, K] Paid;                  // "ragged" matrix of loss paid

    array[N] int<lower=-1, upper=K> s;  // settlement lag 1-K. -1 means not yet settled. 0 shouldn't be present.
    array[N] int<lower=-1, upper=K> c;  // closure lag 1-K. -1 means not yet closed. 0 shouldn't be present.
    array[N] int<lower=1, upper=K> age; // age of claim as of evaluation date - grows after settlement
}
transformed data {
    array[N, K] int<lower=0, upper=1> Valid;          // matrix of bools indicating whether cell is valid for modeling or not
    array[N, K] int<lower=0, upper=1> PostSettlement; // matrix of bools indicating whether cell took place after claim closure
    array[N, K] int<lower=0, upper=1> PostClosure;    // matrix of bools indicating whether cell took place after claim closure

    for (n in 1:N) {
        for (k in 1:K) {
            Valid[n, k] = k <= age[n];
            PostSettlement[n, k] = (s[n] != -1) && (k > s[n]);
            PostClosure[n, k] = (c[n] != -1) && (k > c[n]);
        }
    }

    int N_settled = count_positives(s);
    vector[N_settled] loss_settlement;
    int idx_settled = 0;
    for (n in 1:N) {
        if (s[n] == -1) continue;
        idx_settled += 1;
        loss_settlement[idx_settled] = sum(Paid[n, 1:s[n]]);
        if (is_nan(loss_settlement[idx_settled])) {
            print("NaN settlement loss; Paid row: ", Paid[n], "; s[n]: ", s[n], "; attempted slice: ", Paid[n, 1:s[n]]);
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
}
model {

    // priors
    S_weibull_scale ~ normal(4, 2.5);
    S_weibull_shape ~ normal(1, 0.2);

    X_hurdle_p0 ~ beta(1, 4);
    X_lognorm_mu ~ normal(10, 3);
    X_lognorm_sigma ~ normal(1, 1);


    // likelihoods

    // settlement lag
    for (n in 1:N) {
        if (s[n] == -1) {
            // censored observation
            target += weibull_lccdf(age[n] | S_weibull_shape, S_weibull_scale);
        } else {
            s[n] ~ weibull(S_weibull_shape, S_weibull_scale);
        }
    }

    // known settlement values
    loss_settlement ~ hurdle_lognormal(X_hurdle_p0, X_lognorm_mu, X_lognorm_sigma);
}