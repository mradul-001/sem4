#include <bits/stdc++.h>
using namespace std;

const int INF = 1e9;

struct Edge {
    int v, capacity, flow, rev;
};

class MaxFlow {
    int nodes;
    vector<vector<Edge>> adj;
    vector<int> excess;

public:
    MaxFlow(int n) : nodes(n), adj(n), excess(n, 0) {}

    void addEdge(int u, int v, int lower, int upper) {
        excess[u] -= lower;
        excess[v] += lower;
        Edge a = {v, upper - lower, 0, (int) adj[v].size()};
        Edge b = {u, 0, 0, (int) adj[u].size()};
        adj[u].push_back(a);
        adj[v].push_back(b);
    }

    bool feasibleFlow(int s, int t, int &maxFlowValue) {
        int total = 0;
        for (int i = 0; i < nodes; i++) {
            if (excess[i] > 0) {
                addEdge(s, i, 0, excess[i]);
                total += excess[i];
            } else if (excess[i] < 0) {
                addEdge(i, t, 0, -excess[i]);
            }
        }
        int flow = maxFlow(s, t);
        if (flow != total) return false;
        maxFlowValue = maxFlow(s, t);
        return true;
    }

    int bfs(int s, int t, vector<int>& parent) {
        fill(parent.begin(), parent.end(), -1);
        parent[s] = -2;
        queue<pair<int, int>> q;
        q.push({s, INF});
        while (!q.empty()) {
            int u = q.front().first, flow = q.front().second;
            q.pop();
            for (Edge &e : adj[u]) {
                if (parent[e.v] == -1 && e.flow < e.capacity) {
                    parent[e.v] = u;
                    int new_flow = min(flow, e.capacity - e.flow);
                    if (e.v == t) return new_flow;
                    q.push({e.v, new_flow});
                }
            }
        }
        return 0;
    }

    int maxFlow(int s, int t) {
        int flow = 0, new_flow;
        vector<int> parent(nodes);
        while ((new_flow = bfs(s, t, parent))) {
            int v = t;
            while (v != s) {
                int u = parent[v];
                for (Edge &e : adj[u]) {
                    if (e.v == v) {
                        e.flow += new_flow;
                        adj[v][e.rev].flow -= new_flow;
                        break;
                    }
                }
                v = u;
            }
            flow += new_flow;
        }
        return flow;
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <input_file>" << endl;
        return 1;
    }
    ifstream infile(argv[1]);
    int m, n;
    infile >> m >> n;
    vector<vector<int>> l(m, vector<int>(n)), u(m, vector<int>(n));
    vector<int> r(m), R(m), c(n), C(n);

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            infile >> l[i][j];
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            infile >> u[i][j];
    for (int i = 0; i < m; i++)
        infile >> r[i] >> R[i];
    for (int j = 0; j < n; j++)
        infile >> c[j] >> C[j];
    
    int source = m + n, sink = m + n + 1;
    MaxFlow graph(m + n + 2);
    for (int i = 0; i < m; i++) graph.addEdge(source, i, r[i], R[i]);
    for (int j = 0; j < n; j++) graph.addEdge(j + m, sink, c[j], C[j]);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            graph.addEdge(i, j + m, l[i][j], u[i][j]);
    
    int max_budget;
    if (!graph.feasibleFlow(source, sink, max_budget)) {
        cout << "0\nnot possible" << endl;
        return 0;
    }
    
    int min_budget = accumulate(r.begin(), r.end(), 0);
    cout << "1\n" << min_budget << "\n" << max_budget << endl;
    return 0;
}
