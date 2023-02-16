import numpy as np

def rgba_to_hex(rgb):
	return '#%02x%02x%02x%02x' % rgb

def prediction_to_color(from_squares, to_squares):
	from_squares = np.array(from_squares)
	to_squares   = np.array(to_squares)

	from_squares /= np.amax(from_squares)
	to_squares   /= np.amax(to_squares)

	from_squares *= 255
	to_squares   *= 255

	return {
		i: rgba_to_hex((
			int(from_squares[i]),
			128,
			int(to_squares[i]),
			255))
		for i in range(len(from_squares))
	}