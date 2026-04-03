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


# =========================
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
def sample_pairs(nodes, k=50):
    all_pairs = list(itertools.permutations(nodes, 2))
    return random.sample(all_pairs, min(k, len(all_pairs)))


# =========================
# MÉTRICAS
# =========================
def compute_metrics(edge_load, alpha=1.0, beta=0.3):
    if not edge_load:
        return 0, 0, 0

    max_load = max(edge_load.values())
    avg_load = sum(edge_load.values()) / len(edge_load)

    cost = alpha * max_load + beta * avg_load  # heurística

    return max_load, avg_load, cost


# =========================
# SOLUCIÓN
# =========================
class Solution:
    def __init__(self, routing=None, edge_load=None):
        self.routing = routing or {}
        self.edge_load = edge_load or defaultdict(int)

        self.max_load, self.avg_load, self.cost = compute_metrics(self.edge_load)


# =========================
# GENERAR SOLUCIÓN
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

        # perturbación
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

    return Solution(routing, edge_load)


# =========================
# COMBINACIÓN
# =========================
def combine(s1, s2):
    new_routing = {}
    new_edge_load = defaultdict(int)

    keys = set(s1.routing.keys()) | set(s2.routing.keys())

    for k in keys:
        path = s1.routing.get(k) if random.random() < 0.5 else s2.routing.get(k)
        if not path:
            continue

        new_routing[k] = path

        for i in range(len(path) - 1):
            edge = tuple(sorted((path[i], path[i + 1])))
            new_edge_load[edge] += 1

    return Solution(new_routing, new_edge_load)


# =========================
# MEJORA LOCAL
# =========================
def improve(graph, solution, path_cache):
    routing = dict(solution.routing)
    edge_load = defaultdict(int, solution.edge_load)

    for (u, v), path in list(routing.items()):
        key = (u, v)

        best = path_cache.get(key)
        if best is None:
            best = bidir_bfs(graph, u, v)
            path_cache[key] = best

        if best and len(best) < len(path):

            # quitar viejo
            for i in range(len(path) - 1):
                edge = tuple(sorted((path[i], path[i + 1])))
                edge_load[edge] -= 1
                if edge_load[edge] <= 0:
                    del edge_load[edge]

            # agregar nuevo
            routing[key] = best
            for i in range(len(best) - 1):
                edge = tuple(sorted((best[i], best[i + 1])))
                edge_load[edge] += 1

    return Solution(routing, edge_load)


# =========================
# REFSET
# =========================
def build_refset(population, size):
    sorted_pop = sorted(population, key=lambda s: s.cost)

    best = sorted_pop[: max(1, size // 2)]

    diverse = []
    while len(diverse) < max(1, size // 2):
        candidate = random.choice(population)
        if candidate not in diverse:
            diverse.append(candidate)

    return (best + diverse)[:size]


# =========================
# SCATTER SEARCH
# =========================
def scatter_search(graph, pop_size, refset_size, iterations):
    nodes = graph.nodes()
    pairs = sample_pairs(nodes, k=300)

    path_cache = {}

    population = [
        generate_solution(graph, pairs, path_cache)
        for _ in range(pop_size)
    ]

    best_sol = None
    best_max = float('inf')  # ← índice real

    for _ in range(iterations):
        refset = build_refset(population, refset_size)
        new_pop = []

        for s1, s2 in itertools.combinations(refset, 2):
            child = combine(s1, s2)
            child = improve(graph, child, path_cache)

            # ✔ CRITERIO CORRECTO
            if child.max_load < best_max:
                best_max = child.max_load
                best_sol = child

            new_pop.append(child)

        if new_pop:
            population = new_pop

    return best_sol, best_max


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    graph, pop, refset, iters = read_graph("test1.txt")

    print("========================Parámetros========================")
    print(f"Nodos: {len(graph.nodes())} | Población: {pop} | RefSet: {refset} | Iteraciónes: {iters}")

    #print("=================Grafo=================")
    #for node in graph.nodes():
        #print(f"{node}: {graph.neighbors(node)}")

    inicio = time.perf_counter()

    sol, best_index = scatter_search(graph, pop, refset, iters)

    fin = time.perf_counter()

    print("=================Resultado==================")

    print("Índice de Transmisión aproximado:", best_index)
    print(f"Tiempo de ejecución: {fin - inicio:.4f} segundos")