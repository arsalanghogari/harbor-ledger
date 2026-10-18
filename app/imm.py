"""Interacting Multiple Model estimator: four linear Kalman models over [x, y, vx, vy] at 1 Hz with
position-only measurements, mixed each cycle by how well each explains the last innovation.
Depth runs as a separate 1D filter."""
import numpy as np

DT = 1.0
R_VAR = 4.0  # measurement noise variance, m^2 (sonar grade)
H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
R = R_VAR * np.eye(2)
I4 = np.eye(4)


def cv_F(dt=DT):
    return np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)


def ct_F(w, dt=DT):
    """Coordinated turn at fixed rate w (rad/s); linear in [x, y, vx, vy]."""
    s, c = np.sin(w * dt), np.cos(w * dt)
    return np.array([[1, 0, s / w, -(1 - c) / w], [0, 1, (1 - c) / w, s / w], [0, 0, c, -s], [0, 0, s, c]])


def cv_Q(q, dt=DT):
    return q * np.array([[dt**4 / 4, 0, dt**3 / 2, 0], [0, dt**4 / 4, 0, dt**3 / 2],
                         [dt**3 / 2, 0, dt**2, 0], [0, dt**3 / 2, 0, dt**2]])


class KalmanModel:
    def __init__(self, name, F, Q):
        self.name, self.F, self.Q = name, F, Q

    def predict(self, x, P):
        return self.F @ x, self.F @ P @ self.F.T + self.Q

    def update(self, x, P, z):
        nu = z - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        return x + K @ nu, (I4 - K @ H) @ P, nu, S


MODELS = [KalmanModel("CV-quiet", cv_F(), cv_Q(0.02)),
          KalmanModel("CV-agile", cv_F(), cv_Q(0.5)),
          KalmanModel("CT+", ct_F(0.08), cv_Q(0.02)),
          KalmanModel("CT-", ct_F(-0.08), cv_Q(0.02))]
N = len(MODELS)
STAY = 0.95
MU0 = np.array([0.7, 0.1, 0.1, 0.1])  # prior mode mix: most things in a harbor drift or transit quietly
PI = np.full((N, N), (1 - STAY) / (N - 1)) + np.eye(N) * (STAY - (1 - STAY) / (N - 1))  # rows sum to 1


def gauss_pdf(nu, S):
    return float(np.exp(-0.5 * nu @ np.linalg.solve(S, nu)) / (2 * np.pi * np.sqrt(np.linalg.det(S))))


class IMMTrack:
    def __init__(self, z, depth=0.0, v0_var=4.0):
        x0 = np.array([z[0], z[1], 0.0, 0.0])
        P0 = np.diag([R_VAR, R_VAR, v0_var, v0_var])
        self.xs = [x0.copy() for _ in MODELS]
        self.Ps = [P0.copy() for _ in MODELS]
        self.mu = MU0.copy()
        self.depth, self.depth_var = float(depth), 1.0
        self._combine()

    def _combine(self):
        self.x = sum(m * x for m, x in zip(self.mu, self.xs))
        self.P = sum(m * (P + np.outer(x - self.x, x - self.x)) for m, x, P in zip(self.mu, self.xs, self.Ps))

    def predict(self):
        """Mix the model states by conditional mode probabilities, then predict each model."""
        c = PI.T @ self.mu                        # c_j = sum_i p_ij mu_i
        w = PI * self.mu[:, None] / c[None, :]    # w[i, j] = mu_{i|j}
        xs0 = [sum(w[i, j] * self.xs[i] for i in range(N)) for j in range(N)]
        Ps0 = [sum(w[i, j] * (self.Ps[i] + np.outer(self.xs[i] - xs0[j], self.xs[i] - xs0[j])) for i in range(N))
               for j in range(N)]
        for j, m in enumerate(MODELS):
            self.xs[j], self.Ps[j] = m.predict(xs0[j], Ps0[j])
        self.mu = c
        self.depth_var += 0.01
        self._combine()

    def update(self, z, depth=None):
        L = np.empty(N)
        for j, m in enumerate(MODELS):
            self.xs[j], self.Ps[j], nu, S = m.update(self.xs[j], self.Ps[j], z)
            L[j] = gauss_pdf(nu, S)
        mu = self.mu * np.maximum(L, 1e-300)
        self.mu = mu / mu.sum()
        if depth is not None:
            k = self.depth_var / (self.depth_var + 0.25)
            self.depth += k * (depth - self.depth)
            self.depth_var *= 1 - k
        self._combine()

    def step(self, z=None, depth=None):
        """One cycle. z=None coasts. Returns (combined state, covariance, mode probabilities)."""
        self.predict()
        if z is not None:
            self.update(np.asarray(z, float), depth)
        return self.x, self.P, self.mu

    def mahalanobis2(self, z):
        """Squared Mahalanobis distance of z from the combined prediction."""
        nu = np.asarray(z, float) - H @ self.x
        S = H @ self.P @ H.T + R
        return float(nu @ np.linalg.solve(S, nu))

    def predict_ahead(self, seconds):
        """Combined estimate propagated under the current mode mix: [x, y] per second, index 0 = now."""
        xs = [self.x.copy() for _ in MODELS]
        path = [self.x[:2].copy()]
        for _ in range(int(seconds)):
            xs = [m.F @ x for m, x in zip(MODELS, xs)]
            path.append(sum(mu * x[:2] for mu, x in zip(self.mu, xs)))
        return np.array(path)

    @property
    def mode(self):
        return MODELS[int(np.argmax(self.mu))].name

    @property
    def modes(self):
        return {m.name: round(float(p), 3) for m, p in zip(MODELS, self.mu)}

    @property
    def maneuvering(self):
        """Largest single probability on CV-agile or either CT model. Not the sum: with a
        0.95-diagonal chain three indistinguishable alternatives share 0.75 by symmetry, so the
        sum would call every slow drifter a maneuver."""
        return float(self.mu[1:].max())
