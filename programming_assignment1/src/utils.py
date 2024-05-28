import numpy as np
import matplotlib.pyplot as plt

def plot_contours_with_path(obj_func, points, objectives):
    # Define the range for the plot
    x_min, x_max = -5, 5
    y_min, y_max = -5, 5

    # Generate a grid of points
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)

    # Evaluate the function at each point on the grid
    Z = obj_func(np.array([X, Y]))[0]  # Only use the function value (f)

    # Plot the contour lines of the function
    plt.contour(X, Y, Z, levels=50)

    # Plot the path taken by the optimization algorithm
    plt.plot([point[0] for point in points], [point[1] for point in points], marker='o', markersize=5, linestyle='-', color='r')

    # Annotate each point with its objective value
    for i, point in enumerate(points):
        plt.annotate(f'{objectives[i]:.2f}', (point[0], point[1]), textcoords="offset points", xytext=(-10,10), ha='center')

    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Contour Plot with Optimization Path')
    plt.grid(True)

    # Show the plot
    plt.colorbar(label='Function Value')
    plt.show()