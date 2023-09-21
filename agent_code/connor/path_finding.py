import heapq
import numpy as np

# Define the 4 possible movement directions (no diagonal)
DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
ACTIONS = ["UP", "DOWN", "RIGHT", "LEFT"]


def heuristic(a, b):
    # Manhattan distance heuristic
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(start, goal, field):
    x, y = np.meshgrid(np.arange(field.shape[0]), np.arange(field.shape[1]))
    grid = np.vstack([x.ravel(), y.ravel()]).T

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {tuple(point): float("inf") for point in grid}
    g_score[start] = 0
    f_score = {tuple(point): float("inf") for point in grid}
    f_score[start] = heuristic(start, goal)

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for dx, dy in DIRECTIONS:
            neighbor = current[0] + dx, current[1] + dy

            if (
                neighbor[0] < 0
                or neighbor[0] >= field.shape[0]
                or neighbor[1] < 0
                or neighbor[1] >= field.shape[1]
            ):
                continue

            if (
                field[neighbor] == -1 or field[neighbor] == 1
            ):  # Obstacle or unwalkable cell
                continue

            tentative_g_score = g_score[current] + 1  # Assuming each step cost is 1

            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # No path found
