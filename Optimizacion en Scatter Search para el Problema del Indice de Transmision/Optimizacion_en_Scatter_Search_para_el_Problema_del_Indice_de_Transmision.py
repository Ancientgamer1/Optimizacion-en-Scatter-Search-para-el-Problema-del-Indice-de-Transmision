import random
import itertools
from collections import defaultdict, deque
import copy

# =========================
# GRAFO
# =========================
class Graph:
    def __init__(self):
        self.adj = defaultdict(list)

    def add_edge(self, u, v):
        if v not in self.adj[u]:
            self.adj[u].append(v)
        if u not in self.adj[v]:
            self.adj[v].append(u)

    def nodes(self):
        return list(self.adj.keys())

    def neighbors(self, u):
        return self.adj[u]


# =========================
# LECTURA
# =========================
def read_graph(filename):
    g = Graph()
    with open(filename) as f:
        lines = f.readlines()

    pop_size, refset_size, iterations = map(int, lines[0].split())

    for line in lines[1:]:
        parts = line.split()
        u = parts[0]
        for v in parts[1:]:
            g.add_edge(u, v)

    return g, pop_size, refset_size, iterations


# =========================
# BIDIRECTIONAL BFS (faster for many pairs)
# =========================
def _reconstruct_bidir(parents1, parents2, meeting, start, end):
    path1 = []
    cur = meeting
    while cur is not None:
        path1.append(cur)
        cur = parents1.get(cur)
    path1.reverse()  # from start to meeting

    path2 = []
    cur = parents2.get(meeting)
    while cur is not None:
        path2.append(cur)
        cur = parents2.get(cur)
    # path2 is from node after meeting to end
    return path1 + path2


def bidir_bfs(graph, start, end):
    if start == end:
        return [start]

    # parents maps for reconstruction
    parents1 = {start: None}
    parents2 = {end: None}
    q1 = deque([start])
    q2 = deque([end])

    while q1 and q2:
        # expand smaller frontier first
        if len(q1) <= len(q2):
            for _ in range(len(q1)):
                cur = q1.popleft()
                for n in graph.neighbors(cur):
                    if n not in parents1:
                        parents1[n] = cur
                        q1.append(n)
                        if n in parents2:
                            return _reconstruct_bidir(parents1, parents2, n, start, end)
        else:
            for _ in range(len(q2)):
                cur = q2.popleft()
                for n in graph.neighbors(cur):
                    if n not in parents2:
                        parents2[n] = cur
                        q2.append(n)
                        if n in parents1:
                            return _reconstruct_bidir(parents1, parents2, n, start, end)

    return None


# =========================
# PARES REDUCIDOS
# =========================
def sample_pairs(nodes, k=50):
    all_pairs = list(itertools.permutations(nodes, 2))
    return random.sample(all_pairs, min(k, len(all_pairs)))


# =========================
# SOLUTION WRAPPER: cache edge loads and cost to avoid recompute
# =========================
class Solution:
    def __init__(self, routing=None, edge_load=None, cost=None):
        self.routing = routing or {}
        self.edge_load = edge_load or defaultdict(int)
        self.cost = cost


def compute_cost_from_edge_load(edge_load, alpha=1.0, beta=0.3):
    if not edge_load:
        return 0
    max_load = max(edge_load.values())
    avg_load = sum(edge_load.values()) / len(edge_load)
    return alpha * max_load + beta * avg_load


# =========================
# GENERAR SOLUCIÓN (build edge_load incrementally)
# =========================
def generate_solution(graph, pairs, path_cache):
    routing = {}
    edge_load = defaultdict(int)

    for u, v in pairs:
        key = (u, v)
        path = path_cache.get(key)
        if path is None:
            path = bidir_bfs(graph, u, v)
            path_cache[key] = path

        if path is None:
            continue

        # pequeña perturbación (still using cache aggressively)
        if random.random() < 0.2:
            neighbors = graph.neighbors(u)
            if neighbors:
                alt = random.choice(neighbors)
                alt_key = (alt, v)
                alt_path = path_cache.get(alt_key)
                if alt_path is None:
                    alt_path = bidir_bfs(graph, alt, v)
                    path_cache[alt_key] = alt_path
                if alt_path:
                    path = [u] + alt_path

        routing[key] = path

        for i in range(len(path) - 1):
            edge = tuple(sorted((path[i], path[i + 1])))
            edge_load[edge] += 1

    cost = compute_cost_from_edge_load(edge_load)
    return Solution(routing, edge_load, cost)


# =========================
# COMBINACIÓN (incremental cost construction)
# =========================
def combine(s1: Solution, s2: Solution):
    new_routing = {}
    new_edge_load = defaultdict(int)

    # iterate over union of keys to be robust
    keys = set(s1.routing.keys()) | set(s2.routing.keys())
    for k in keys:
        # prefer lower-cost parent's path probabilistically
        pick_from_s1 = random.random() < 0.5
        path = None
        if pick_from_s1:
            path = s1.routing.get(k) or s2.routing.get(k)
        else:
            path = s2.routing.get(k) or s1.routing.get(k)

        if not path:
            continue

        new_routing[k] = path
        for i in range(len(path) - 1):
            edge = tuple(sorted((path[i], path[i + 1])))
            new_edge_load[edge] += 1

    new_cost = compute_cost_from_edge_load(new_edge_load)
    return Solution(new_routing, new_edge_load, new_cost)


# =========================
# MEJORA LOCAL (attempt to replace paths with shorter cached ones and update edge_load minimally)
# =========================
def improve(graph, solution: Solution, path_cache):
    # create copies of routing and edge_load to modify
    routing = dict(solution.routing)
    edge_load = defaultdict(int, solution.edge_load)

    changed = False

    for (u, v), path in list(routing.items()):
        key = (u, v)
        best = path_cache.get(key)
        if best is None:
            best = bidir_bfs(graph, u, v)
            path_cache[key] = best

        if best and len(best) < len(path):
            # decrement counts for old path
            for i in range(len(path) - 1):
                edge = tuple(sorted((path[i], path[i + 1])))
                edge_load[edge] -= 1
                if edge_load[edge] <= 0:
                    del edge_load[edge]

            # apply new path
            routing[key] = best
            for i in range(len(best) - 1):
                edge = tuple(sorted((best[i], best[i + 1])))
                edge_load[edge] += 1

            changed = True

    if not changed:
        return solution  # no change

    cost = compute_cost_from_edge_load(edge_load)
    return Solution(routing, edge_load, cost)


# =========================
# REFSET (use precomputed costs)
# =========================
def build_refset(population, size):
    scored = sorted(population, key=lambda s: s.cost)
    best = scored[: max(1, size // 2)]

    # diversify by picking dissimilar routings (simple approach: random unique)
    diverse = []
    attempts = 0
    while len(diverse) < max(1, size // 2) and attempts < len(population) * 3:
        candidate = random.choice(population)
        if candidate not in diverse:
            diverse.append(candidate)
        attempts += 1

    ref = best + diverse
    # trim to desired size
    return ref[:size]


# =========================
# SCATTER SEARCH (works with Solution objects)
# =========================
def scatter_search(graph, pop_size, refset_size, iterations):
    nodes = graph.nodes()
    pairs = sample_pairs(nodes, k=50)

    path_cache = {}

    population = [
        generate_solution(graph, pairs, path_cache)
        for _ in range(pop_size)
    ]

    best_sol = None
    best_cost = float('inf')

    for _ in range(iterations):
        refset = build_refset(population, refset_size)
        new_pop = []

        for r1, r2 in itertools.combinations(refset, 2):
            child = combine(r1, r2)
            child = improve(graph, child, path_cache)

            if child.cost < best_cost:
                best_cost = child.cost
                best_sol = child

            new_pop.append(child)

        population = new_pop if new_pop else population

    return best_sol, best_cost


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    graph, pop, refset, iters = read_graph("test1.txt")

    print("=============Parámetros=============")
    print(f"Nodos: {pop} Refset: {refset} Iteraciones: {iters}")

    sol, cost = scatter_search(graph, pop, refset, iters)

    print("Mejor costo (aprox índice transmisión):", cost)