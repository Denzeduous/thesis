def rgb_to_hex(rgb):
	return '#%02x%02x%02x' % rgb

def prediction_to_color(from_squares, to_squares):
	colors = {i: (0, 0, 0) for i in range(len(from_squares))}

	largest_from = 0

	for i in range(len(from_squares)):
		if from_squares[i] > largest_from:
			largest_from = from_squares[i]

		color = colors[i]
		# print(from_squares[i])
		# print(255 * from_squares[i])
		colors[i] = (int(255 * from_squares[i]), color[1], color[2])

	largest_to = 0

	for i in range(len(to_squares)):
		if to_squares[i] > largest_to:
			largest_to = to_squares[i]

		color = colors[i]
		# print(to_squares[i])
		# print(255 * to_squares[i])
		colors[i] = (color[0], color[1], int(255 * to_squares[i]))

	return {
		i: rgb_to_hex((
			int(colors[i][0] / largest_from * 255),
			0,
			int(colors[i][2] / largest_to * 255)))
		for i in range(len(from_squares))
	}