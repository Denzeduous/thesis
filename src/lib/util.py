import numpy as np

def rgba_to_hex(rgb):
	return '#%02x%02x%02x%02x' % rgb

def prediction_to_color(from_squares, to_squares):
	colors = {i: (0, 0) for i in range(len(from_squares))}

	from_squares = np.array(from_squares)
	to_squares   = np.array(to_squares)

	for i in range(len(from_squares)):
		color = colors[i]
		colors[i] = (255 * from_squares[i], color[1])

	for i in range(len(to_squares)):
		color = colors[i]
		colors[i] = (color[0], 255 * to_squares[i])

	largest_from = np.amax(from_squares)
	largest_to = np.amax(to_squares)

	return {
		i: rgba_to_hex((
			int(colors[i][0] / largest_from * 255),
			128,
			int(colors[i][1] / largest_to * 255),
			255))
		for i in range(len(from_squares))
	}