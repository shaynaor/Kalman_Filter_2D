import numpy as np


def predict_X_k(F: np.ndarray, X_k_minus1: np.ndarray, B=0, u_k=0) -> np.ndarray:
    """
    predict the next state at time k.
    X_k = F*X_k-1 + B*u_k.
    :param F: also called A. takes every point in our original estimate and moves it to a new predicted location.
    :param X_k_minus1: the current state (at time k-1)
    :param B: the control matrix
    :param u_k: control vector
    :return: X_k
    """
    return np.add(np.dot(F, X_k_minus1), np.dot(B, u_k))


def get_init_cov_mat(var_pos: float, var_vel: float) -> np.ndarray:
    """
    :return: the initial process covariance matrix (P_k-1)
    """
    return np.array([[var_pos, 0], [0, var_vel]])


def calc_predicted_cov_mat(F: np.ndarray, P_k_minus1: np.ndarray, Q_k=0) -> np.ndarray:
    """
    P_k = F*P_k-1*F^T + Q_k
    :param F:  also called A. takes every point in our original estimate and moves it to a new predicted location.
    :param P_k_minus1: last covariance matrix
    :param Q_k: additional uncertainty from the environment
    :return: the predicted process covariance matrix
    """
    return np.multiply(np.add(np.dot(np.dot(F, P_k_minus1), np.transpose(F)), Q_k), np.eye(2))


def calc_kalman_gain(P_k: np.ndarray, H: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    calculating the kalman gain
    K = (P_k*H^T) * (H*P_k*H^T + R)^-1
    :param P_k: the covariance matrix
    :param H: sensor model matrix
    :param R: sensor noise
    :return: the kalman gain
    """
    return np.dot(np.dot(P_k, np.transpose(H)), np.linalg.inv(np.add(np.dot(np.dot(H, P_k), np.transpose(H)), R)))


def calc_cur_state(X_k: np.ndarray, K: np.ndarray, Z: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    X_real = X_k + K(Z - H*X_k)
    :param X_k: predicted state
    :param K: kalman gain
    :param Z: sensor measurement
    :param H: sensor model matrix
    :return: our new best estimate
    """
    return np.add(X_k, np.dot(K, np.subtract(Z, np.dot(H, X_k))))


def update_cov_mat(P_k: np.ndarray, K: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    P_real = (I - K*H)*P_k
    :param P_k: the covariance matrix
    :param K: kalman gain
    :param H: sensor model matrix
    :return: updated covariance error.
    """
    return np.dot(np.subtract(np.eye(2), np.dot(K, H)), P_k)


def kalman_update(F: np.ndarray, P: np.ndarray, X: np.ndarray, B=0, u_k=0) -> (np.ndarray, np.ndarray):
    """
    time update ("predict")
    :param F: also called A. takes every point in our original estimate and moves it to a new predicted location.
    :param P: covariance matrix
    :param X: state
    :param B: the control matrix
    :param u_k: control vector
    :return: (X_k, P_k) predicted state and covariance matrix.
    """
    return predict_X_k(F, X, B, u_k), calc_predicted_cov_mat(F, P)


def kalman_measurement(P_k: np.ndarray, H: np.ndarray, R: np.ndarray, X_k: np.ndarray, Z: np.ndarray) -> \
        (np.ndarray, np.ndarray):
    """
    measurement update ("correct")
    :param P_k: covariance matrix
    :param H: sensor model matrix
    :param R: sensor noise
    :param X_k: predicted state
    :param Z: measurement
    :return: (X_real, P_real): the new vector state and the new uncertainty
    """
    K = calc_kalman_gain(P_k, H, R)
    return calc_cur_state(X_k, K, Z, H), update_cov_mat(P_k, K, H)


def calc_kalman_step(H: np.ndarray, F: np.ndarray, P: np.ndarray, X: np.ndarray, Z: np.ndarray,
                     R: np.ndarray, B=0, u_k=0) -> (np.ndarray, np.ndarray):
    """
    predict and correct.
    :param H: sensor model matrix
    :param F: also called A. takes every point in our original estimate and moves it to a new predicted location.
    :param P: covariance matrix
    :param X: state
    :param Z: measurement
    :param R: sensor noise
    :param B: the control matrix
    :param u_k: control vector
    :return: (X_real, P_real): the new vector state and the new uncertainty
    """
    X_k, P_k = kalman_update(F, P, X, B, u_k)  # time update ("predict")
    return kalman_measurement(P_k, H, R, X_k, Z)  # measurement update ("correct")
