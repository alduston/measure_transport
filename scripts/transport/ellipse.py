import numpy as np
import matplotlib.pyplot as plt


def ellipse_arc(a, b, theta, n):
    """Cumulative arc length of ellipse with given dimensions"""

    # Divide the interval [0 , theta] into n steps at regular angles
    t = np.linspace(0, theta, n)

    # Using parametric form of ellipse, compute ellipse coord for each t
    x, y = np.array([a * np.cos(t), b * np.sin(t)])

    # Compute vector distance between each successive point
    x_diffs, y_diffs = x[1:] - x[:-1], y[1:] - y[:-1]

    cumulative_distance = [0]
    c = 0

    # Iterate over the vector distances, cumulating the full arc
    for xd, yd in zip(x_diffs, y_diffs):
        c += np.sqrt(xd**2 + yd**2)
        cumulative_distance.append(c)
    cumulative_distance = np.array(cumulative_distance)

    # Return theta-values, distance cumulated at each theta,
    # and total arc length for convenience
    return t, cumulative_distance, c


def theta_from_arc_length_constructor(a, b, theta=2*np.pi, n=100):
    """
    Inverse arc length function: constructs a function that returns the
    angle associated with a given cumulative arc length for given ellipse."""

    # Get arc length data for this ellipse
    t, cumulative_distance, total_distance = ellipse_arc(a, b, theta, n)

    # Construct the function
    def f(s):
        assert np.all(s <= total_distance), "s out of range"
        # Can invert through interpolation since monotonic increasing
        return np.interp(s, cumulative_distance, t)

    # return f and its domain
    return f, total_distance


def rand_ellipse(a=2, b=0.5, size=50, presicion=1000):
    """
    Returns uniformly distributed random points from perimeter of ellipse.
    """
    theta_from_arc_length, domain = theta_from_arc_length_constructor(a, b, theta=2*np.pi, n=presicion)
    s = np.random.rand(size) * domain
    t = theta_from_arc_length(s)
    x, y = np.array([a * np.cos(t), b * np.sin(t)])
    return np.asarray([x, y]).reshape((2,len(x)))


