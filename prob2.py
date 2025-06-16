import numpy as np

# Generate a 1D NumPy array of 20 random floats between 0 and 10
array = np.random.uniform(0, 10, size=20)

print("Generated Array (Original, Full Precision):\n", array)

# Round all elements to two decimal places
rounded_array = np.round(array, 2)
print("\nRounded Array (2 Decimal Places):\n", rounded_array)

# Calculate minimum, maximum, and median of the array
min_val = np.min(rounded_array)
max_val = np.max(rounded_array)
median_val = np.median(rounded_array)

print(f"\nMinimum Value: {min_val}")
print(f"Maximum Value: {max_val}")
print(f"Median Value: {median_val}")

# Replace all elements less than 5 with their squares
modified_array = rounded_array.copy()
modified_array[modified_array < 5] = np.square(modified_array[modified_array < 5])
print("\nModified Array (Elements < 5 Replaced by Their Squares):\n", modified_array)

# Function to sort array in alternating order: smallest, largest, 2nd smallest, 2nd largest, etc.
def numpy_alternate_sort(arr):
    """
    Sorts a 1D NumPy array in an alternating pattern:
    smallest, largest, second smallest, second largest, and so on.
    
    Parameters:
        arr (np.ndarray): Input 1D NumPy array.
        
    Returns:
        np.ndarray: New array sorted in alternating order.
    """
    sorted_arr = np.sort(arr)
    result = []
    left, right = 0, len(sorted_arr) - 1

    # Alternate appending from smallest and largest
    while left <= right:
        result.append(sorted_arr[left])
        left += 1
        if left <= right:
            result.append(sorted_arr[right])
            right -= 1

    return np.array(result)

# Generate the alternating sorted array
alternating_sorted_array = numpy_alternate_sort(rounded_array)
print("\nArray Sorted in Alternating Smallest-Largest Order:\n", alternating_sorted_array)
