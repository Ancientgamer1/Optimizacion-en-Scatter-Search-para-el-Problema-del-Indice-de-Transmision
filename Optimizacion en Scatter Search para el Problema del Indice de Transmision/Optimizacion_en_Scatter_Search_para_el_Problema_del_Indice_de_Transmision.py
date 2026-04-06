import pprint
import random
import itertools
import time
from collections import defaultdict, deque

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


# ========================
# BIDIRECTIONAL BFS
# =========================
def _reconstruct_bidir(p1, p2, meet):
    path1 = []
    cur = meet
    while cur is not None:
        path1.append(cur)
        cur = p1.get(cur)
    path1.reverse()

    path2 = []
    cur = p2.get(meet)
    while cur is not None:
        path2.append(cur)
        cur = p2.get(cur)

    return path1 + path2


def bidir_bfs(graph, start, end):
    if start == end:
        return [start]

    p1 = {start: None}
    p2 = {end: None}
    q1 = deque([start])
    q2 = deque([end])

    while q1 and q2:
        if len(q1) <= len(q2):
            for _ in range(len(q1)):
                cur = q1.popleft()
                for n in graph.neighbors(cur):
                    if n not in p1:
                        p1[n] = cur
                        q1.append(n)
                        if n in p2:
                            return _reconstruct_bidir(p1, p2, n)
        else:
            for _ in range(len(q2)):
                cur = q2.popleft()
                for n in graph.neighbors(cur):
                    if n not in p2:
                        p2[n] = cur
                        q2.append(n)
                        if n in p1:
                            return _reconstruct_bidir(p1, p2, n)

    return None


# =========================
# PARES REDUCIDOS
# =========================
def sample_pairs(nodes, k):
    all_pairs = list(itertools.permutations(nodes, 2))
    return random.sample(all_pairs, min(k, len(all_pairs)))


# =========================
# FUNCIÓN DE CONGESTIÓN (CORRECTA)
# =========================
def compute_max_load(edge_load):
    undirected = defaultdict(int)

    for (u, v), load in edge_load.items():
        key = tuple(sorted((u, v)))
        undirected[key] += load

    return max(undirected.values()) if undirected else 0

# =========================
# SOLUCIÓN
# =========================
class Solution:
    def __init__(self, routing=None, edge_load=None):
        self.routing = routing or {}
        self.edge_load = edge_load or defaultdict(int)

        self.max_load = compute_max_load(self.edge_load)
        self.avg_load = sum(self.edge_load.values()) / len(self.edge_load)

        self.cost = self.max_load + 0.3 * self.avg_load

# =========================
# DISTANCIA (OPTIMIZADA)
# =========================
def distance(s1, s2):
    edges = set(s1.edge_load.keys()) | set(s2.edge_load.keys())
    return sum(abs(s1.edge_load.get(e, 0) - s2.edge_load.get(e, 0)) for e in edges)

# =========================
# GENERAR SOLUCIÓN
# =========================
def generate_solution(graph, pairs, cache):
    routing = {}
    edge_load = defaultdict(int)

    for u, v in pairs:
        key = (u, v)
        path = cache.get(key)

        if path is None:
            path = bidir_bfs(graph, u, v)
            cache[key] = path

        if not path:
            continue

        routing[key] = path

        # 🔥 CAMBIO CLAVE: DIRECCIONAL
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])  # NO sorted
            edge_load[edge] += 1

    return Solution(routing, edge_load)


# =========================
# COMBINACIÓN
# =========================
def combine(s1, s2):
    routing = {}
    edge_load = defaultdict(int)

    keys = set(s1.routing) | set(s2.routing)

    for k in keys:
        path = s1.routing.get(k) if random.random() < 0.5 else s2.routing.get(k)
        if not path:
            path = s1.routing.get(k) or s2.routing.get(k)

        if not path:
            continue

        routing[k] = path

        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])  # DIRECCIONAL
            edge_load[edge] += 1

    return Solution(routing, edge_load)


# =========================
# MEJORA LOCAL
# =========================
def improve(graph, sol, cache):
    routing = dict(sol.routing)
    edge_load = defaultdict(int, sol.edge_load)

    for (u, v), path in list(routing.items()):
        best = cache.get((u, v))
        if best is None:
            best = bidir_bfs(graph, u, v)
            cache[(u, v)] = best

        if best and len(best) < len(path):
            for i in range(len(path) - 1):
                edge = tuple(sorted((path[i], path[i + 1])))
                edge_load[edge] -= 1
                if edge_load[edge] <= 0:
                    del edge_load[edge]

            routing[(u, v)] = best

            for i in range(len(best) - 1):
                edge = tuple(sorted((best[i], best[i + 1])))
                edge_load[edge] += 1

    return Solution(routing, edge_load)


# =========================
# REFSET (DIVERSIDAD CONTROLADA)
# =========================
def build_refset(population, size):
    sorted_pop = sorted(population, key=lambda s: s.cost)
    best = sorted_pop[: max(1, size // 2)]

    diverse = []

    # LIMITAMOS candidatos → evita explosión
    candidates = random.sample(population, min(30, len(population)))

    while len(diverse) < max(1, size // 2):
        best_candidate = None
        best_dist = -1

        for c in candidates:
            d = min(distance(c, s) for s in best + diverse) if (best or diverse) else 0

            if d > best_dist:
                best_dist = d
                best_candidate = c

        if best_candidate:
            diverse.append(best_candidate)

    return (best + diverse)[:size]


# =========================
# SCATTER SEARCH
# =========================
def scatter_search(graph, pop_size, refset_size, iterations, sample_size):
    nodes = graph.nodes()
    pairs = sample_pairs(nodes, sample_size)

    cache = {}

    population = [
        generate_solution(graph, pairs, cache)
        for _ in range(pop_size)
    ]

    best_sol = None
    best_val = float("inf")

    for _ in range(iterations):
        refset = build_refset(population, refset_size)
        new_pop = []

        for s1, s2 in itertools.combinations(refset, 2):
            child = combine(s1, s2)
            child = improve(graph, child, cache)

            if child.max_load < best_val:
                best_val = child.max_load
                best_sol = child

            new_pop.append(child)

        if new_pop:
            population = new_pop

    return best_sol, best_val


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    SAMPLE_SIZE = 300

    graph, pop, refset, iters = read_graph("test1.txt")

    print("========================Parámetros========================")
    print(f"Nodos: {len(graph.nodes())} | Población: {pop} | RefSet: {refset} | Iteraciónes: {iters}")

    inicio = time.perf_counter()

    sol, best_index = scatter_search(graph, pop, refset, iters, SAMPLE_SIZE)

    fin = time.perf_counter()

    print("========================Resultados========================")

    print("Índice de Transmisión aproximado:", best_index)
    print(f"Tiempo de ejecución: {fin - inicio:.4f} segundos")