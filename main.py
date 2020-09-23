from kalman_2d import *


def main():
    """
    this example taken from Michel van Biezen course:
    https://www.youtube.com/watch?v=krdA3yyOfgI&list=PLX2gX-ftPVXU3oUFNATxGXY90AULiqnWT&index=35
    """
    delta_t = 1  # difference in time
    # init values
    pos = 4000
    vel = 280

    # Standard Deviations
    pos_std = 20
    vel_std = 5

    # variance
    pos_var = np.square(pos_std)
    vel_var = np.square(vel_std)

    a = np.array([[2]])  # acceleration
    F = np.array([[1, delta_t],
                  [0, 1]])

    X = np.array([[pos],
                  [vel]])

    B = np.array([[np.multiply(1 / 2, np.square(delta_t))],
                  [delta_t]])

    P_k_minus1 = get_init_cov_mat(pos_var, vel_var)

    # sensor of vel and pos
    H = np.array([[1, 0],
                  [0, 1]])

    # Observation Errors
    error_pos = 25  # Uncertainty in the measurement
    error_vel = 6

    R = np.array([[np.square(error_pos), 0],
                  [0, np.square(error_vel)]])


    Z = np.array([[4260.],
                  [282.]])


    x_real, p_real = calc_kalman_step(H, F, P_k_minus1, X, Z, R, B, a)
    print("x_real: \n{}\n".format(x_real))
    print("p_real: \n{}\n".format(p_real))



if __name__ == '__main__':
    main()
