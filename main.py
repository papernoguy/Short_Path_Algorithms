import networkx as nx
import heapq
import numpy as np
from queue import PriorityQueue


# Helper function to display full path from each node to its root
def full_path(start, parents):
    def find_path(start, end, parents):
        if end == start:
            return [start]
        else:
            path = [end]
            node = end
            while parents[node] != start:
                path.append(parents[node])
                node = parents[node]
            path.append(start)
            return path

    paths = {}
    for node in parents:
        if node != start:
            paths[node] = find_path(start, node, parents)
    return paths

# Helper function to feasible_solution function
def add_super_node(adj_matrix):
    n = len(adj_matrix)
    new_matrix = np.full((n + 1, n + 1), float('inf'))
    new_matrix[:n, :n] = adj_matrix
    for i in range(n + 1):
        new_matrix[i][n] = 0
    return new_matrix


# For positive edges only
def dijkstra(graph, start):
    costs = {node: float('infinity') for node in graph.nodes()}
    costs[start] = 0
    pi = {node: None for node in graph.nodes()}
    unvisited = [(0, start)]

    while unvisited:
        current_cost, current = heapq.heappop(unvisited)
        for neighbor in graph.neighbors(current):
            new_cost = costs[current] + graph[current][neighbor]['weight']
            if new_cost < costs[neighbor]:
                costs[neighbor] = new_cost
                pi[neighbor] = current
                heapq.heappush(unvisited, (new_cost, neighbor))
    print("dijkstra:::\n", costs, full_path(start, pi))
    return costs, full_path(start, pi)


# For positive and negative edges, but not negative circle
def bellman_ford(graph, source):
    # Initialize distances from the source to all other vertices as infinity
    # except the source itself, which is 0
    dist = [float('inf')] * len(graph)
    dist[source] = 0

    # Repeat the relaxation process |V| - 1 times
    for _ in range(len(graph) - 1):
        for u in range(len(graph)):
            for v in range(len(graph)):
                if graph[u][v] != float('inf') and dist[u] != float('inf') and dist[u] + graph[u][v] < dist[v]:
                    dist[v] = dist[u] + graph[u][v]

    # Check for negative-weight cycles
    for u in range(len(graph)):
        for v in range(len(graph)):
            if graph[u][v] != float('inf') and dist[u] != float('inf') and dist[u] + graph[u][v] < dist[v]:
                print("Graph contains a negative-weight cycle")
                return None

    return dist


# For positive and negative edges, but not circle
def shortest_path_on_DAG(graph, s):
    topological_order = list(nx.topological_sort(graph))
    shortest_dist = {node: float('inf') for node in graph}
    shortest_dist[s] = 0
    previous = {node: None for node in graph}
    for node in topological_order:
        for neighbor in graph[node]:
            if shortest_dist[neighbor] > shortest_dist[node] + graph[node][neighbor]['weight']:
                shortest_dist[neighbor] = shortest_dist[node] + graph[node][neighbor]['weight']
                previous[neighbor] = node

    print("shortest_path_on_DAG:::\n", shortest_dist, full_path(s, previous))
    return shortest_dist, full_path(s, previous)


# Find solution for constraints matrix
def feasible_solution(adj_matrix):
    new_adj_matrix = add_super_node(adj_matrix)
    source = 0
    solution = bellman_ford(new_adj_matrix, source)[:-1]
    if solution is None:
        print("No feasible solution")
        return None
    else:
        print("feasible_solution:::\n", solution)
        return solution


# For positive and negative edges, it solves the All Pairs Shortest Path problem
def floyd_warshall(graph):
    dist = np.array(graph)
    n = len(dist)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    # Check for negative weight cycles
    for i in range(n):
        if dist[i][i] < 0:
            print("Graph contains a negative-weight cycle")
            return None
    print("floyd_warshall:::\n", dist)
    return dist


# Heuristic function for a*
def euclidean_distance(a, b):
    # Calculate the norm of a vector or a matrix
    return np.linalg.norm(np.array(a) - np.array(b))


def a_star(graph, start, goal):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()
        if current == goal:
            break
        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph[current][next]['weight']
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + euclidean_distance(graph.nodes[goal]['pos'], graph.nodes[next]['pos'])
                frontier.put(next, priority)
                came_from[next] = current
    print("A star:::\n", came_from, cost_so_far)
    return came_from, cost_so_far


if __name__ == '__main__':
    g = nx.Graph()
    g.add_edge('A', 'B', weight=5)
    g.add_edge('A', 'C', weight=3)
    g.add_edge('B', 'D', weight=2)
    g.add_edge('C', 'D', weight=1)
    g.add_edge('D', 'E', weight=10)
    dij_output = dijkstra(g, 'A')

    dag = nx.DiGraph()
    # Add edges and edge weights to the graph
    dag.add_edge('a', 'b', weight=3)
    dag.add_edge('a', 'c', weight=5)
    dag.add_edge('b', 'd', weight=2)
    dag.add_edge('c', 'd', weight=4)
    dag.add_edge('c', 'e', weight=6)
    dag.add_edge('d', 'f', weight=1)
    dag.add_edge('e', 'f', weight=2)
    sp_dag_output = shortest_path_on_DAG(dag, 'a')

    n = 4  # number of vertices
    adj_matrix = np.full((n, n), float('inf'))
    adj_matrix[1][0] = 5  # A <- B with weight 5
    adj_matrix[2][0] = 6  # A <- C with weight 6
    adj_matrix[3][1] = -1  # B <- D with weight -1
    adj_matrix[3][2] = -2  # C <- D with weight -2
    adj_matrix[0][3] = -3  # D <- A with weight -3
    np.fill_diagonal(adj_matrix, 0)

    feasible_solution(adj_matrix)

    fs_graph = [[0, 3, float('inf'), 1],
             [float('inf'), 0, 3, 2],
             [float('inf'), float('inf'), 0, float('inf')],
             [float('inf'), float('inf'), -3, 0]]
    floyd_warshall(fs_graph)

    G = nx.Graph()
    G.add_node(1, pos=(1, 2))
    G.add_node(2, pos=(2, 3))
    G.add_node(3, pos=(3, 4))
    G.add_node(4, pos=(4, 5))
    G.add_edge(1, 2, weight=1)
    G.add_edge(1, 3, weight=2)
    G.add_edge(2, 4, weight=3)
    G.add_edge(3, 4, weight=4)
    path = a_star(G, 1, 4)

