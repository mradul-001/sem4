#include <bits/stdc++.h>
using namespace std;

void addEdge(int from, int to, int cap, vector<vector<int>> &capMat, vector<vector<int>> &adj)
{
    capMat[from][to] += cap;
    adj[from].push_back(to);
    adj[to].push_back(from);
}

int edmondsKarp(int n, int s, int t, vector<vector<int>> &capMat, vector<vector<int>> &adj, vector<vector<int>> &flowMat)
{
    flowMat.assign(n, vector<int>(n, 0));
    int maxFlow = 0;
    while (true)
    {
        vector<int> parent(n, -1), flow(n, 0);
        queue<int> q;
        q.push(s);
        flow[s] = INT_MAX;
        while (!q.empty())
        {
            int u = q.front();
            q.pop();
            for (int v : adj[u])
            {
                if (parent[v] == -1 && capMat[u][v] > flowMat[u][v])
                {
                    parent[v] = u;
                    flow[v] = min(flow[u], capMat[u][v] - flowMat[u][v]);
                    if (v == t)
                        break;
                    q.push(v);
                }
            }
        }
        if (parent[t] == -1)
            break;
        maxFlow += flow[t];
        for (int v = t; v != s; v = parent[v])
        {
            int u = parent[v];
            flowMat[u][v] += flow[t];
            flowMat[v][u] -= flow[t];
        }
    }
    return maxFlow;
}

int main()
{

    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int m, n;
    cin >> m >> n;

    vector<vector<int>> cellL(m, vector<int>(n)), cellU(m, vector<int>(n));
    vector<int> rowL(m), rowU(m), colL(n), colU(n);

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            cin >> cellL[i][j] >> cellU[i][j];

    for (int i = 0; i < m; i++)
        cin >> rowL[i] >> rowU[i];

    for (int j = 0; j < n; j++)
        cin >> colL[j] >> colU[j];

    int total = m * n + 4;
    int S = m * n, T = m * n + 1, S_prime = m * n + 2, T_prime = m * n + 3;

    vector<vector<int>> cap(total, vector<int>(total, 0));
    vector<vector<int>> adj(total);
    int required = 0;

    // Handle cell lower bounds and connect to S' and T'
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            int node = i * n + j;
            int lb = cellL[i][j];
            addEdge(S_prime, node, lb, cap, adj);
            addEdge(node, T_prime, lb, cap, adj);
            required += lb;
            addEdge(S, node, cellU[i][j] - lb, cap, adj);
        }
    }

    // Handle row constraints
    for (int i = 0; i < m; i++)
    {
        int sumL = 0;
        for (int j = 0; j < n; j++)
            sumL += cellL[i][j];
        int rowNode = S;
        addEdge(S_prime, rowNode, rowL[i] + sumL, cap, adj);
        addEdge(rowNode, T_prime, rowL[i] + sumL, cap, adj);
        required += rowL[i];
        addEdge(S, rowNode, rowU[i] - rowL[i], cap, adj);
    }

    // Handle column constraints
    for (int j = 0; j < n; j++)
    {
        int sumL = 0;
        for (int i = 0; i < m; i++)
            sumL += cellL[i][j];
        int colNode = T;
        addEdge(S_prime, colNode, colL[j] + sumL, cap, adj);
        addEdge(colNode, T_prime, colL[j] + sumL, cap, adj);
        required += colL[j];
        addEdge(colNode, T, colU[j] - colL[j], cap, adj);
    }

    vector<vector<int>> flow;
    int phase1 = edmondsKarp(total, S_prime, T_prime, cap, adj, flow);

    if (phase1 != required)
    {
        cout << "0\n";
        return 0;
    }

    int preSatisfiedFlow = 0;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            int rowOutVertex = 2 * m * n + 2 * i + 1;
            int cellInVertex = 2 * i * n + 2 * j;
            preSatisfiedFlow += flow[rowOutVertex][cellInVertex];
        }
    }

    vector<vector<int>> capacityMatrixPhase2(total, vector<int>(total, 0));
    vector<vector<int>> adjacencyListPhase2(total);

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            int cellInVertex = 2 * i * n + 2 * j;
            int cellOutVertex = cellInVertex + 1;
            int usedFlow = flow[cellInVertex][cellOutVertex];
            int remainingCapacity = cellU[i][j] - (cellL[i][j] + usedFlow);
            addEdge(cellInVertex, cellOutVertex, remainingCapacity, capacityMatrixPhase2, adjacencyListPhase2);
        }
    }
    int rowInVertex;
    int rowOutVertex;
    for (int i = 0; i < m; i++)
    {
        rowInVertex = 2 * m * n + 2 * i;
        rowOutVertex = rowInVertex + 1;
        int usedFlow = flow[rowInVertex][rowOutVertex];
        addEdge(rowInVertex, rowOutVertex, rowU[i] - (rowL[i] + usedFlow), capacityMatrixPhase2, adjacencyListPhase2);
    }

    for (int j = 0; j < n; j++)
    {
        int colInVertex = 2 * m * n + 2 * m + 2 * j;
        int colOutVertex = colInVertex + 1;
        int usedFlow = flow[colInVertex][colOutVertex];
        addEdge(colInVertex, colOutVertex, colU[j] - (colL[j] + usedFlow), capacityMatrixPhase2, adjacencyListPhase2);
    }

    for (int i = 0; i < m; i++)
    {
        int rowInVertex = 2 * m * n + 2 * i;
        addEdge(rowInVertex, rowInVertex, INT_MAX, capacityMatrixPhase2, adjacencyListPhase2);
    }

    for (int j = 0; j < n; j++)
    {
        int colOutVertex = 2 * m * n + 2 * m + 2 * j + 1;
        addEdge(colOutVertex, rowOutVertex, INT_MAX, capacityMatrixPhase2, adjacencyListPhase2);
    }

    for (int i = 0; i < m; i++)
    {
        int rowOutVertex = 2 * m * n + 2 * i + 1;
        for (int j = 0; j < n; j++)
        {
            int cellInVertex = 2 * i * n + 2 * j;
            addEdge(rowOutVertex, cellInVertex, INT_MAX, capacityMatrixPhase2, adjacencyListPhase2);
        }
    }

    for (int j = 0; j < n; j++)
    {
        int colInVertex = 2 * m * n + 2 * m + 2 * j;
        for (int i = 0; i < m; i++)
        {
            int cellOutVertex = 2 * i * n + 2 * j + 1;
            addEdge(cellOutVertex, colInVertex, INT_MAX, capacityMatrixPhase2, adjacencyListPhase2);
        }
    }

    addEdge(rowOutVertex, rowInVertex, INT_MAX, capacityMatrixPhase2, adjacencyListPhase2);

    vector<vector<int>> flowMatrixPhase2;
    int maxFlowPhase2 = edmondsKarp(total, rowInVertex, rowOutVertex, capacityMatrixPhase2, adjacencyListPhase2, flowMatrixPhase2);

    cout << 1 << "\n";
    cout << (maxFlowPhase2 + preSatisfiedFlow) << "\n";
    cout << preSatisfiedFlow << "\n";

    return 0;
}