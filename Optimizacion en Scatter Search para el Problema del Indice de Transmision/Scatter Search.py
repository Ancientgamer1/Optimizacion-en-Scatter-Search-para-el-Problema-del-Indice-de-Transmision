import random
import heapq

def generate_initial_paths(graph, start, end, k):
    """
    Creates an initial population of k diverse paths.
    """
    candidates = []
    for _ in range(k):
        p, c = random_walk(graph, start, end)
        if p:
            candidates.append((c, p))
    return candidates

def random_walk(graph, start, end):
    """
    Generates a random feasible path from start to end.
    This provides diversification in Scatter Search.
    """
    path = [start]
    cost = 0
    visited = {start}
    current = start

    while current != end:
        # Choose among unvisited neighbors
        neighbors = [(n, w) for n, w in graph[current] if n not in visited]
        if not neighbors:
            return None, float('inf')

        nxt, w = random.choice(neighbors)
        path.append(nxt)
        cost += w
        visited.add(nxt)
        current = nxt

    return path, cost

def path_cost(graph, path):
    """
    Computes the total weight of a path.
    """
    total = 0
    for i in range(len(path) - 1):
        for v, w in graph[path[i]]:
            if v == path[i + 1]:
                total += w
                break
    return total

def improve_path(graph, path):
    """
    Local improvement step.
    Currently re-evaluates cost but can be extended.
    """
    return path, path_cost(graph, path)

def build_reference_set(paths, size):
    """
    Selects the best solutions to form the reference set.
    """
    paths.sort(key=lambda x: x[0])
    return paths[:size]

def combine_paths(graph, p1, p2):
    """
    Combines two elite paths to create a new solution.
    """
    cut = min(len(p1), len(p2)) // 2
    new_path = p1[:cut] + p2[cut:]
    cost = path_cost(graph, new_path)
    return new_path, cost

def scatter_search(graph, start, end, num_paths, ref_size, iterations):
    """
    Scatter Search to compute a near-shortest path.
    """
    # Step 1: Diversification
    pool = generate_initial_paths(graph, start, end, num_paths)
    pool = [(path_cost(graph, p), p) for _, p in pool]

    # Step 2: Reference set initialization
    ref_set = build_reference_set(pool, ref_size)

    # Step 3: Solution combination loop
    for _ in range(iterations):
        offspring = []

        for i in range(len(ref_set) - 1):
            p1 = ref_set[i][1]
            p2 = ref_set[i + 1][1]

            new_p, new_c = combine_paths(graph, p1, p2)
            new_p, new_c = improve_path(graph, new_p)

            offspring.append((new_c, new_p))

        # Step 4: Reference set update
        ref_set.extend(offspring)
        ref_set = build_reference_set(ref_set, ref_size)

    return min(ref_set, key=lambda x: x[0])

def scatter_search_routing(graph, num_paths, ref_size, iterations):
    """
    Computes routing paths between all node pairs using Scatter Search.
    """
    routing = {}
    nodes = list(graph.keys())

    for s in nodes:
        routing[s] = {}
        for t in nodes:
            if s == t:
                routing[s][t] = [s]
            else:
                _, best_path = scatter_search(graph, s, t,num_paths, ref_size, iterations)
                routing[s][t] = best_path

    return routing

# -------------------------------
# Graph definition
# -------------------------------

graph = {
    'A': [('B', 4), ('C', 2), ('E', 3), ('F', 7)],
    'B': [('A', 4), ('C', 5), ('D', 10), ('E', 1), ('F', 6)],
    'C': [('A', 2), ('B', 5), ('D', 3), ('E', 4), ('F', 2)],
    'D': [('B', 10), ('C', 3), ('E', 6), ('G', 1)],
    'E': [('A', 3), ('B', 1), ('C', 4), ('D', 6), ('F', 3), ('G', 2)],
    'F': [('A', 7), ('B', 6), ('C', 2), ('D', 8), ('E', 3), ('G', 5)],
    'G': [('D', 1), ('E', 2), ('F', 5), ('H', 4)],
    'H': [('G', 4), ('I', 2)],
    'I': [('H', 2), ('J', 3)],
    'J': [('I', 3), ('K', 6)],
    'K': [('J', 6), ('L', 5)],
    'L': [('K', 5), ('M', 2)],
    'M': [('L', 2), ('N', 3)],
    'N': [('M', 3), ('O', 2)],
    'O': [('N', 2), ('P', 1)],
    'P': [('O', 1), ('Q', 4)],
    'Q': [('P', 4), ('R', 6)],
    'R': [('Q', 6)],
}


shortest_path = scatter_search(graph, 'A', 'H', num_paths=20, ref_size=8, iterations=15)
print("Shortest path found:", shortest_path)

# Extract the actual path from the (cost, path) tuple
_, path = shortest_path

# ---------------------------------------------------
# Build routing table based on this shortest path
# ---------------------------------------------------
def initialize_edge_load(graph):
    """
    Initializes edge load counters.
    """
    load = {}
    for u in graph:
        for v, _ in graph[u]:
            load[frozenset({u, v})] = 0
    return load


def transmission_index(graph, routing):
    """
    Computes the transmission (edge forwarding) index
    induced by Scatter Search routing.
    """
    edge_load = initialize_edge_load(graph)

    for s in routing:
        for t in routing[s]:
            if s == t:
                continue

            path = routing[s][t]
            for i in range(len(path) - 1):
                edge = frozenset({path[i], path[i + 1]})

                # Only count valid edges
                if edge in edge_load:
                    edge_load[edge] += 1

    return max(edge_load.values()) if edge_load else 0

def print_first_last_matrix(routing, nodes):
    col_width = 8

    # Header
    header = " " * col_width
    for v in nodes:
        header += f"{v:^{col_width}}"
    print(header)
    print("-" * len(header))

    # Rows
    for s in nodes:
        row = f"{s:^{col_width}}"
        for t in nodes:
            if s == t:
                cell = "--"
            else:
                path = routing[s][t]
                cell = f"({path[0]}, {path[-1]})"
            row += f"{cell:^{col_width}}"
        print(row)


routing = scatter_search_routing(
    graph,
    num_paths=20,
    ref_size=8,
    iterations=15
)

pi = transmission_index(graph, routing)

print("Indice de Transmision (Scatter Search):", pi)

nodes = list(graph.keys())
print_first_last_matrix(routing, nodes)
