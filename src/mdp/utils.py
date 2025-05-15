#! /usr/bin/python
# utils.py

def print_maze_with_padding(maze):
    """
    Prints the maze with a border of 1s (walls) on all sides.
    """
    if not maze or not maze[0]:
        raise ValueError("Maze must be a non-empty 2D list")

    rows = len(maze)
    cols = len(maze[0])

    # Print top wall
    print("1 " * (cols + 2))

    for row in maze:
        print("1", end=" ")  # Left wall
        print(" ".join(str(cell) for cell in row), end=" ")
        print("1")  # Right wall

    # Print bottom wall
    print("1 " * (cols + 2))