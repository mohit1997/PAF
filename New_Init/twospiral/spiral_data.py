"""
	The spiral dataset is generated based on Matt White's algorithm
	
	Author         : Kelvin Yin
	Python Version : 3.5.2
"""

# Import libraries
from math import pi, cos, sin

def generate_spiral_dataset(density=1, maxRadius=6.5, c=0):
	""" To generate spiral dataset

	It generates one spirals dataset with x and y coordinates

	This generator generates points, each with 96 * density + 1 data points
	(3 revolutions of 32 times the density plus one end point).

	Args:
		density (int)  : Density of the points
		maxRadius (float) : Maximum radius of the spiral
		c (int) : Class of this spiral

	Returns:
		array: Return spiral data and its class

	"""
	
	# Spirals data
	spirals_data = []
	
	# Spirals class
	spirals_class = []

	# Number of interior data points to generate
	points = 96 * density 

	# Generate points
	for i in range(0, points):
		# Angle is based on the iteration * PI/16, divided by point density
		angle = (i * pi) / (16 * density)

		# Radius is the maximum radius * the fraction of iterations left
		radius = maxRadius * ((104 * density) - i) / (104 * density)

		# Get x and y coordinates
		x = radius * cos(angle)
		y = radius * sin(angle)
		
		# Format: 8.5f
		x = float(format(x, '8.5f'))
		y = float(format(y, '8.5f'))

		spirals_data.append([x, y])
		spirals_class.append([c])

	return spirals_data, spirals_class

def generate_two_spirals_dataset(density=1, maxRadius=6.5):
	""" To generate two spirals dataset

	It generates two spirals dataset with x and y coordinates
	
	First spiral dataset will be generated with the function
	`generate_spiral_dataset`.

	Then, the coordinates will be flipped to get second
	spiral dataset

	Args:
		density (int)  : Density of the points
		maxRadius (float) : Maximum radius of the spiral

	Returns:
		array: Return spirals data and its class
	"""

	# Spirals data
	spirals_data = []
	
	# Spirals class
	spirals_class = []

	# First spirals data and class
	first_spiral_data, first_spiral_class = generate_spiral_dataset(density, maxRadius)

	# Construct complete two spirals dataset
	for fsd in first_spiral_data:
		# First spirals coordinate
		spirals_data.append(fsd)

		# Second spirals coordinate
		spirals_data.append([-fsd[0], -fsd[1]])

	# Construct complete two spirals classes
	for fsc in first_spiral_class:
		# First spirals class
		spirals_class.append(fsc)

		# Second spirals class
		spirals_class.append([1])

	return spirals_data, spirals_class

def generate_three_spirals_dataset(density=1, maxRadius=6.5):
	""" To generate three spirals dataset
	
	It generates three spirals dataset with x and y coordinates
	
	Two spirals dataset will be generated with the function
	`generate_two_spirals_dataset`.

	Then, new maximum radius will be calculated to generate
	third spirals dataset.

	New maximum radius is calculated with:
		newMaxRadius = (maxRadius + second_spirals[+max_x, 0]) / 2

	New maximum radius is used in `generate_spiral_dataset()` to
	get new third spiral dataset

	Args:
		density (int)  : Density of the points
		maxRadius (float) : Maximum radius of the spiral

	Returns:
		array: Return spirals data and its class
	"""
	
	# Spirals data
	spirals_data = []
	
	# Spirals class
	spirals_class = []
	
	# Two spirals data and class
	two_spirals_data, two_spirals_class = generate_two_spirals_dataset(density, maxRadius)

	# New maximum radius
	newMaxRadius = maxRadius;

	# Find new maximum radius (Find maximum x coordinate of second spiral)
	for tsd in two_spirals_data:
		if (tsd[0] > 0 and tsd[1] == 0 and tsd[0] < maxRadius):
			newMaxRadius = (maxRadius + tsd[0]) / 2
			break

	# Generate third spiral data
	third_spiral_data, third_spiral_class = generate_spiral_dataset(density, newMaxRadius, 2)
	
	# Construct spirals data
	for i in range(0, max(len(two_spirals_data), len(third_spiral_data)), 2):
		# Storing spirals data		
		if (i < len(two_spirals_data)):
			spirals_data.append(two_spirals_data[i])
			spirals_class.append(two_spirals_class[i])
		
		if ((i + 1) < len(two_spirals_data)):
			spirals_data.append(two_spirals_data[i + 1])
			spirals_class.append(two_spirals_class[i + 1])

		if (round(i / 2) < len(third_spiral_data)):
			spirals_data.append(third_spiral_data[round(i / 2)])
			spirals_class.append(third_spiral_class[round(i / 2)])

	return spirals_data, spirals_class
	
def generate_four_spirals_dataset(density=1, maxRadius=6.5):
	""" To generate four spirals dataset
	
	It generates four spirals dataset with x and y coordinates
	
	Three spirals dataset will be generated with the function
	`generate_three_spirals_dataset()`.

	Then, third spiral dataset will be flipped to generate the fourth
	spiral.

	Args:
		density (int)  : Density of the points
		maxRadius (float) : Maximum radius of the spiral

	Returns:
		array: Return spirals data and its class
	"""

	# Spirals data
	spirals_data = []
	
	# Spirals class
	spirals_class = []

	# Three spirals
	three_spirals_data, three_spirals_class = generate_three_spirals_dataset(density, maxRadius)

	# Construct spirals data
	i = 0 # Iterator
	for tsd in three_spirals_data:
		spirals_data.append(tsd)
		
		i += 1

		if (i == 3):
			spirals_data.append([-tsd[0], -tsd[1]])
			i = 0

	# Construct spirals class
	i = 0
	for tsc in three_spirals_class:
		spirals_class.append(tsc)

		i += 1

		if (i == 3):
			spirals_class.append([3])
			i = 0

	return spirals_data, spirals_class
		
