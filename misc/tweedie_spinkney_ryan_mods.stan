functions {
  int num_non_zero_fun(array[] real y) {
    int A = 0;
    int N = num_elements(y);

    for (n in 1 : N) {
      if (y[n] != 0) {
        A += 1;
      }
    }
    return A;
  }

  array[] int non_zero_index_fun(array[] real y) {
    int N = num_elements(y);
    int A = num_non_zero_fun(y);
    array[A] int non_zero_index;
    int counter = 0;
    for (n in 1 : N) {
      if (y[n] != 0) {
        counter += 1;
        non_zero_index[counter] = n;
      }
    }
    return non_zero_index;
  }

  array[] int zero_index_fun(array[] real y) {
    int N = num_elements(y);
    int Z = N - num_non_zero_fun(y);
    array[Z] int zero_index;
    int counter = 0;
    for (n in 1 : N) {
      if (y[n] == 0) {
        counter += 1;
        zero_index[counter] = n;
      }
    }
    return zero_index;
  }

  void check_tweedie(real mu, real phi, real theta) {
    if (mu < 0) {
      reject("mu must be >= 0; found mu =", mu);
    }
    if (phi < 0) {
      reject("phi must be >= 0; found phi =", phi);
    }
    if (theta < 1 || theta > 2) {
      reject("theta must be in [1, 2]; found theta =", theta);
    }
  }

  void check_tweedie(vector mu, real phi, real theta) {
    int N = num_elements(mu);
    if (phi < 0) {
      reject("phi must be >= 0; found phi =", phi);
    }
    if (theta < 1 || theta > 2) {
      reject("theta must be in [1, 2]; found theta =", theta);
    }

    for (n in 1 : N) {
      if (mu[n] < 0) {
        reject("mu must be >= 0; found mu =", mu[n], "on element n=", n);
      }
    }
  }

  vector positive_filter(array[] real y) {
    array[A] int non_zero_index = zero_index_fun(y);
    vector[A] y_non_zero;
    matrix[A, M] ps;
    for (n in 1 : A) {
      y_non_zero[n] = y[non_zero_index[n]];
    }
  }
  vector positive_filter(vector y) {
    array[A] int non_zero_index = zero_index_fun(y);
    vector[A] y_non_zero;
    matrix[A, M] ps;
    for (n in 1 : A) {
      y_non_zero[n] = y[non_zero_index[n]];
    }
  }

  real tweedie_lpdf(array[] real y, int M, real mu, real phi, real theta) {
    check_tweedie(mu, phi, theta);
    int N = num_elements(y);
    int A = num_elements(non_zero_index);
    int NmA = N - A;
    real lambda = 1 / phi * mu ^ (2 - theta) / (2 - theta);
    real alpha = (2 - theta) / (theta - 1);
    real beta = 1 / phi * mu ^ (1 - theta) / (theta - 1);
    real lp = -NmA * lambda;


    for (m in 1 : M) {
      real shape = m * alpha;
      ps[, m] = poisson_lpmf(m | lambda)
                + (shape - 1) * log(y_non_zero)
                - beta * y_non_zero
                + shape * log(beta)
                - lgamma(shape);
    }
    for (n in 1 : A) {
      lp += log_sum_exp(ps[n, ]');
    }
    return lp;
  }

  real tweedie_lpdf(array[] real y, array[] int non_zero_index, int M, real mu, real phi, real theta) {
    check_tweedie(mu, phi, theta);
    int N = num_elements(y);
    int A = num_non_zero_fun(y);
    int NmA = N - A;
    real lambda = 1 / phi * mu ^ (2 - theta) / (2 - theta);
    real alpha = (2 - theta) / (theta - 1);
    real beta = 1 / phi * mu ^ (1 - theta) / (theta - 1);
    real lp = -NmA * lambda;

    for (n in 1 : A) {
      vector[M] ps;
      for (m in 1 : M) {
        ps[m] = poisson_lpmf(m | lambda) + gamma_lpdf(y[non_zero_index[n]] | m * alpha, beta);
      }
      lp += log_sum_exp(ps);
    }
    return lp;
  }

  real tweedie_lpdf(real y, int M, real mu, real phi, real theta) {
    check_tweedie(mu, phi, theta);
    int N = num_elements(y);
    int A = num_elements(non_zero_index);
    int NmA = N - A;
    real lambda = 1 / phi * mu ^ (2 - theta) / (2 - theta);
    real alpha = (2 - theta) / (theta - 1);
    real beta = 1 / phi * mu ^ (1 - theta) / (theta - 1);
    real lp = -NmA * lambda;

    for (n in 1 : A) {
      vector[M] ps;
      for (m in 1 : M) {
        ps[m] = poisson_lpmf(m | lambda) + gamma_lpdf(y[non_zero_index[n]] | m * alpha, beta);
      }
      lp += log_sum_exp(ps);
    }
    return lp;
  }

  // vector version for mu is untested
  real tweedie_lpdf(array[] real y, array[] int non_zero_index, array[] int zero_index, int M, vector mu, real phi, real theta) {
    check_tweedie(mu, phi, theta);
    int N = num_elements(y);
    int A = num_elements(non_zero_index);
    int NmA = N - A;
    vector[N] lambda = 1 / phi * mu ^ (2 - theta) / (2 - theta);
    real alpha = (2 - theta) / (theta - 1);
    vector[N] beta = 1 / phi * mu ^ (1 - theta) / (theta - 1);
    real lp = -sum(lambda[zero_index]);

    for (n in 1 : A) {
      vector[M] ps;
      for (m in 1 : M) {
        ps[m] = poisson_lpmf(m | lambda[n]) + gamma_lpdf(y[non_zero_index[n]] | m * alpha, beta[n]);
      }
      lp += log_sum_exp(ps);
    }
    return lp;
  }

  real tweedie_rng(real mu, real phi, real theta) {
    check_tweedie(mu, phi, theta);

    real lambda = 1 / phi * mu ^ (2 - theta) / (2 - theta);
    real alpha = (2 - theta) / (theta - 1);
    real beta = 1 / phi * mu ^ (1 - theta) / (theta - 1);

    int N = poisson_rng(lambda);
    real tweedie_val;

    if (theta == 1) {
      return phi * poisson_rng(mu / phi);
    }
    if (theta == 2) {
      return gamma_rng(1 / phi, beta);
    }
    if (N * alpha == 0) {
      return 0.;
    }

    return gamma_rng(N * alpha, beta);
  }
}
data {
  int N;
  int M;
  array[N] real<lower=0> Y;
}
transformed data {
  int R = 20;
  array[N_zero] int zero_index = zero_index_fun(Y);
  array[N_non_zero] int non_zero_index = non_zero_index_fun(Y);
}
parameters {
  real<lower=0> mu;
  real<lower=0> phi;
  real<lower=1, upper=2> theta;
}
model {
  mu ~ cauchy(0, 5);
  phi ~ cauchy(0, 5);

  Y ~ tweedie(non_zero_index, M, mu, phi, theta);
  // mu is a vector then
  // Y ~ tweedie(non_zero_index, zero_index, M, mu, phi, theta);

}
generated quantities {
  vector[R] r_tweedie;
  vector[R] r_log_lik;
  // below has mu as a real
  // if mu is a vector you need to
  // generate values for each mu
  for (r in 1 : R) {
    r_tweedie[r] = tweedie_rng(mu, phi, theta);
    r_log_lik[r] = tweedie_lpdf()
  }
}