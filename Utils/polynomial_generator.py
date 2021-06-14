import numpy as np


def generate_polynomial(theta_values):
    # join the coeff and intercept array
    # using numpy poly function, i.e generating a polynomial from the theta values
    generated_polynomial = np.poly1d(theta_values)
    print(generated_polynomial)
    return generated_polynomial



my_polynomial = generate_polynomial()

my_polynomial(-3)
my_polynomial(-2)
my_polynomial(-1)
my_polynomial(0)
my_polynomial(1)
my_polynomial(2)
my_polynomial(3)
