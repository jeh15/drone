import numpy

# 3 Design Variables:
H = numpy.zeros((3, 3), dtype=float)
f = numpy.zeros((3,), dtype=float)
H_block = numpy.zeros((2, 2), dtype=float)
f_block = numpy.zeros((2,), dtype=float)

# Data Point Locations:
xd = numpy.array([0, 0.5, 1.7, 3], dtype=float)
yd = numpy.array([1, 0.8, 0.7, 0.5], dtype=float)
x = numpy.linspace(xd[0], xd[-1], 3)

# Compute Hessian:
j = 0
for i in range(len(xd)):
    if xd[i] >= x[j+1]:
        H[j:j+2, j:j+2] = H[j:j+2, j:j+2] + H_block
        f[j:j+2] = f[j:j+2] + f_block
        j = j + 1
        H_block[:, :] = 0.0
        f_block[:] = 0.0

    if xd[i] == x[j]:
        H_block[0, 0] = 2.0
        f_block[0] = -2 * yd[i]
    else:
        span = x[j] - x[j+1]
        span_sq = span ** 2
        upper_span = xd[i] - x[j+1]
        lower_span = xd[i] - x[j]

        H_block[0, 0] = H_block[0, 0] + 2 * upper_span ** 2 / span_sq
        H_block[1, 0] = H_block[1, 0] + 2 * -lower_span * lower_span / span_sq
        H_block[1, 1] = H_block[1, 1] + 2 * -lower_span ** 2 / span_sq

        f_block[0] = f_block[0] + 2 * yd[i] * (-upper_span) / span
        f_block[1] = f_block[1] + 2 * yd[i] * (lower_span) / span

        # H_block[0, 0] = H_block[0, 0] + 2 * (xd[i] - x[j+1]) ** 2 / (x[j] - x[j+1]) ** 2
        # H_block[0, 1] = H_block[0, 1] + 2 * (x[j] - xd[i]) * (xd[i] - x[j]) / (x[j] - x[j+1]) ** 2
        # H_block[1, 1] = H_block[1, 1] + 2 * (x[j] - xd[i]) ** 2 / (x[j] - x[j+1]) ** 2

# Add End Point:
H[-1, -1] = H[-1, -1] + 2.0
f[-1] = f[-1] - 2.0 * yd[-1]

H_ = H.T + H
numpy.fill_diagonal(H_, numpy.diag(H))

