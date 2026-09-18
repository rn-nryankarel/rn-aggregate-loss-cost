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
    int N_claims;                              // number of claim records
    int K;                                     // max age
    matrix[N_claims, K] Inc;                   // "ragged" matrix of loss incurred
    matrix[N_claims, K] Paid;                  // "ragged" matrix of loss paid

    array[N_claims] int<lower=-1, upper=K> s;  // settlement lag 1-K. -1 means not yet settled. 0 shouldn't be present.
    array[N_claims] int<lower=-1, upper=K> c;  // closure lag 1-K. -1 means not yet closed. 0 shouldn't be present.
    array[N_claims] int<lower=1, upper=K> age; // age of claim as of evaluation date - grows after settlement

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
    int N_settled = count_positives(s);
    vector[N_settled] loss_settlement;
    int idx_settled = 0;
    for (n in 1:N_claims) {
        if (s[n] == -1) continue;
        idx_settled += 1;
        loss_settlement[idx_settled] = sum(Paid[n, 1:s[n]]);
        if (is_nan(loss_settlement[idx_settled])) {
            print("NaN settlement loss; Paid row: ", Paid[n], "; s[n]: ", s[n], "; attempted slice: ", Paid[n, 1:s[n]]);
        }
    }

    // turn data into long form
    int N_long = 0; // N_claims * K less unapplicable cells
    for (n in 1:N_claims) {
        N_long += age[n];
    }
    array[N_long] int<lower=1, upper=K> t;                // time index for claim
    array[N_long] int<lower=0, upper=1> s_state;          // boolean indicating whether claim was settled or not by this point
    array[N_long] int<lower=0, upper=1> c_state;          // boolean indicating whether claim file was closed or not by this point
    array[N_long] int<lower=1, upper=N_claims> claim;     // integer tracking claim groups
    array[N_long] real paid;                              // incremental paid
    array[N_long] real incurred;                          // incremental incurred
    int i = 0; // counter to iterate through every value of N_long
    for (n in 1:N_claims) {
        int age_n = age[n];
        for (k in 1:K) {
            if (k > age_n) {
                break;
            }
            i += 1;
            t[i] = k;
            s_state[i] = k >= s[n];
            c_state[i] = k >= c[n];
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
        if (abs(paid[n] - expected_long_paid[n]) < 0.01) {
            reject("n=", n, ": paid[n]=", paid[n], " but expected_long_paid[n]=", expected_long_paid[n]);
        }
        if (abs(incurred[n] < expected_long_incurred[n]) < 0.01) {
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

    // need some parameters to define time series of reported r_t
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
    for (n in 1:N_claims) {
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