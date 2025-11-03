import requests

# Prepare your graph batch (example)
graph_batch = [
    {"num_nodes": 5, "adjacency_list": [[1, 2],[0, 2],[0, 1, 3],[2, 4],[3]]},
    {"num_nodes": 4, "adjacency_list": [[1],[0,2],[1,3],[2]]},
    # ... more graphs ...
]

# Use a session
with requests.Session() as session:
    results = []
    for graph in graph_batch:
        response = session.post("http://localhost:8008/max_clique", json=graph)
        if response.ok:
            clique = response.json()["max_clique"]
            results.append(clique)
            print("Max clique:", clique)
        else:
            print("Error:", response.text)
