def rgb_to_hex(rgb):
	return '#%02x%02x%02x' % rgb

def prediction_to_color(from_squares, to_squares):
	colors = {i: (0, 0, 0) for i in range(len(from_squares))}

	largest = 0

	for i in range(len(from_squares)):
		if from_squares[i] > largest:
			largest = from_squares[i]

		color = colors[i]
		colors[i] = (int(255 * from_squares[i]), color[1], color[2])

	for i in range(len(to_squares)):
		if to_squares[i] > largest:
			largest = to_squares[i]

		color = colors[i]
		colors[i] = (color[0], color[1], int(255 * to_squares[i]))

	return {
		i: rgb_to_hex((
			int(colors[i][0] / largest * 255),
			0,
			int(colors[i][2] / largest * 255)))
		for i in range(len(from_squares))
	}