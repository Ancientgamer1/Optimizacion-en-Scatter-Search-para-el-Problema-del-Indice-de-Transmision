import random
from collections import defaultdict, deque

# =========================
# GRAFO
# =========================
class Graph:
    def __init__(self):
        self.adj = defaultdict(list) # Lista de adyacencia para representar el grafo

    def add_edge(self, u, v):       # Agrega una arista entre los nodos u y v (grafo no dirigido)
        if v not in self.adj[u]:
            self.adj[u].append(v)
        if u not in self.adj[v]:
            self.adj[v].append(u)

    def nodes(self):
        return list(self.adj.keys())    # Devuelve una lista de los nodos del grafo

    def neighbors(self, u):
        return self.adj[u]      # Devuelve una lista de los vecinos del nodo u


# =========================
# SOLUCIÓN
# =========================
class Solution:
    def __init__(self, routing, edge_load, max_load):
        self.routing = routing              # routing: diccionario con rutas
        self.edge_load = edge_load          # edge_load: carga de cada arista
        self.max_load = max_load            # max_load: carga máxima en cualquier arista


# =========================
# LECTURA DEL GRAFO
# =========================
def read_graph(filename):
    g = Graph()

    with open(filename) as f:       # Abre el archivo para lectura
        lines = f.readlines()

    pop_size, refset_size, iterations = map(int, lines[0].split())      # Lee la primera línea para obtener el tamaño de la población, el tamaño del refset y el número de iteraciones

    for line in lines[1:]:
        parts = line.split()        # Lee cada línea del archivo para construir el grafo. La primera parte es el nodo u y las partes siguientes son los vecinos v de u
        u = parts[0]
        for v in parts[1:]:
            g.add_edge(u, v)        # Agrega una arista entre u y cada uno de sus vecinos v

    return g, pop_size


# =========================
# BFS CON VARIACIÓN
# =========================
def bfs_random(graph, start, end):
    queue = deque([start])      # Cola para BFS
    visited = {start: None}     # Diccionario para rastrear nodos visitados y sus predecesores

    while queue:
        current = queue.popleft()       # Extrae el nodo actual de la cola

        if current == end:
            break

        neighbors = graph.neighbors(current)[:]  # Obtiene los vecinos del nodo actual
        random.shuffle(neighbors)  # Mezcla los vecinos para introducir aleatoriedad en la búsqueda

        for neighbor in neighbors:      # Itera sobre los vecinos
            if neighbor not in visited:
                visited[neighbor] = current     # Marca el vecino como visitado y guarda su predecesor
                queue.append(neighbor)

    if end not in visited:
        return None

    path = []       # Lista para almacenar el camino (u, v))
    node = end      # Comienza desde el nodo de destino y retrocede hasta el nodo de inicio utilizando el diccionario visited para reconstruir el camino

    while node is not None:
        path.append(node)
        node = visited[node]

    path.reverse()
    
    return path


# =========================
# GENERAR PARES (Muestreo)
# =========================
def generate_pairs(nodes, sample_size=50):
    all_pairs = [(u, v) for u in nodes for v in nodes if u != v]    # Genera todos los pares posibles de nodos (u, v) donde u y v son diferentes
    return random.sample(all_pairs, min(sample_size, len(all_pairs)))   # Devuelve una muestra aleatoria de pares, limitando el tamaño a sample_size o al número total de pares disponibles


# =========================
# EDGE LOAD
# =========================
def compute_edge_load(routing):
    edge_load = defaultdict(int)    # Diccionario para contar la carga de cada arista

    for path in routing.values():   # Itera sobre todas las rutas en el diccionario de routing
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            edge_load[edge] += 1

    return edge_load


# =========================
# MAX LOAD
# =========================
def compute_max_load(edge_load):
    if not edge_load:
        return 0
    return max(edge_load.values())  # Devuelve la carga máxima entre todas las aristas, o 0 si no hay aristas


# =========================
# GENERAR UNA SOLUCIÓN
# =========================
def generate_solution(graph, sample_size=50):
    nodes = graph.nodes()                       # Obtiene la lista de nodos del grafo
    pairs = generate_pairs(nodes, sample_size)  # Genera una muestra de pares de nodos para los cuales se buscarán rutas

    routing = {}    # Diccionario para almacenar las rutas entre los pares de nodos

    for (u, v) in pairs:
        path = bfs_random(graph, u, v)  # Busca una ruta aleatoria entre los nodos u y v
        if path:
            routing[(u, v)] = path      # Al encontrar una ruta, se almacena en el routing con la clave (u, v)

    edge_load = compute_edge_load(routing)
    max_load = compute_max_load(edge_load)

    return Solution(routing, edge_load, max_load) # Devuelve una instancia de la clase Solution


# =========================
# GENERAR POBLACIÓN
# =========================
def generate_population(graph, pop_size, sample_size=50):
    population = []     # Lista para almacenar la población de soluciones

    for _ in range(pop_size):
        sol = generate_solution(graph, sample_size)     # Genera una solución aleatoria y la agrega a la población
        population.append(sol)

    return population


# =========================
# IMPRESIÓN
# =========================
def print_population(population):
    for i, sol in enumerate(population):
        print(f"\n=== SOLUCIÓN {i+1} ===")
        print(f"Max Load: {sol.max_load}")

        print("Rutas (primeros 5):")
        for j, ((u, v), path) in enumerate(sorted(sol.routing.items())):
            if j >= 5:
                break
            print(f"{u}->{v}: {path}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    graph, pop_size = read_graph("test2.txt")

    print("=== GRAFO ===")
    for node in sorted(graph.nodes()):
        print(f"{node}: {graph.neighbors(node)}")

    population = generate_population(graph, pop_size, sample_size=50)   # Sample size se puede ajustar para controlar cuántos pares de nodos se consideran al generar cada solución

    print_population(population)