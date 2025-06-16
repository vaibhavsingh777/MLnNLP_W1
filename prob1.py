import numpy as np

# Generate a 2D NumPy array of shape (5, 4) filled with random integers between 1 and 50
array = np.random.randint(1, 51, size=(5, 4))

print("Generated Array:\n", array)

# Extract elements along the anti-diagonal (top-right to bottom-left)
anti_diagonal = [array[i, -1 - i] for i in range(min(array.shape))]
print("\nAnti-diagonal Elements:", anti_diagonal)

# Compute and print the maximum value in each row
max_in_each_row = np.max(array, axis=1)
print("\nMaximum Value in Each Row:", max_in_each_row)

# Create a new array containing only the elements <= overall mean of the array
mean_value = np.mean(array)
elements_leq_mean = array[array <= mean_value]
print("\nOverall Mean of the Array:", mean_value)
print("Elements <= Overall Mean:\n", elements_leq_mean)

# Function to perform boundary traversal of a 2D NumPy array (clockwise)
def numpy_boundary_traversal(matrix):
    rows, cols = matrix.shape
    boundary = []

    # Top row (left to right)
    boundary.extend(matrix[0, :])

    # Right column (top to bottom, excluding first element)
    if rows > 1:
        boundary.extend(matrix[1:rows-1, -1])

    # Bottom row (right to left)
    if rows > 1:
        boundary.extend(matrix[-1, ::-1])

    # Left column (bottom to top, excluding first and last element)
    if cols > 1 and rows > 2:
        boundary.extend(matrix[-2:0:-1, 0])

    return boundary

# Display the boundary traversal
boundary_elements = numpy_boundary_traversal(array)
print("\nBoundary Traversal (Clockwise):", boundary_elements)
