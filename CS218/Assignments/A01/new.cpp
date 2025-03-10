#include <bits/stdc++.h>
using namespace std;

// Structure for an edge: u-v endpoints, weight w, color c (0 = blue, 1 = red)
struct Edge
{
    int u, v, w, c;
};

// Graph structure holding number of vertices, number of edges, and the list of edges
struct Graph
{
    int V, E;
    vector<Edge> edges;
    Graph(int V, int E) : V(V), E(E) {}
    void addEdge(const Edge &e)
    {
        edges.push_back(e);
    }
};

// Disjoint Set Union (DSU) structure to support Kruskal’s MST construction
struct DisjointSet
{
    vector<int> parent, rank;
    DisjointSet(int n)
    {
        parent.resize(n);
        rank.resize(n, 1);
        for (int i = 0; i < n; i++)
            parent[i] = i;
    }
    int findRoot(int x)
    {
        return parent[x] == x ? x : parent[x] = findRoot(parent[x]);
    }
    void doUnion(int x, int y)
    {
        int rx = findRoot(x), ry = findRoot(y);
        if (rx != ry)
        {
            if (rank[rx] < rank[ry])
                swap(rx, ry);
            parent[ry] = rx;
            if (rank[rx] == rank[ry])
                rank[rx]++;
        }
    }
};

// Structure to store MST information: total cost, red edge count, and the vector of MST edges.
struct MSTInfo
{
    int cost;
    int redCount;
    vector<Edge> mstEdges;
};

bool comparator1(const Edge &a, const Edge &b)
{
    if (a.w == b.w)
        return a.c < b.c;
    return a.w < b.w;
}

vector<Edge> formMSTEdges(int n, vector<Edge> &edges)
{
    vector<Edge> mstEdges;
    sort(edges.begin(), edges.end(), comparator1);

    DisjointSet ds(n);
    for (auto &e : edges)
    {
        if (ds.findRoot(e.u) != ds.findRoot(e.v))
        {
            ds.doUnion(e.u, e.v);
            mstEdges.push_back(e);
        }
    }
    return mstEdges;
}

// Compute the MST cost and red edge count given an MST edge list.
pair<int, int> computeMSTStats(const vector<Edge> &mstEdges)
{
    int cost = 0, redCount = 0;
    for (const auto &e : mstEdges)
    {
        cost += e.w;
        if (e.c == 1)
            redCount++;
    }
    return {cost, redCount};
}

// DFS function to initialize par[0] and maxRed[0].
void dfs(int u, int p, const vector<vector<pair<int, pair<int, int>>>> &adj,
         vector<vector<int>> &par, vector<vector<pair<int, int>>> &maxRed, vector<int> &depth)
{
    par[0][u] = p;
    for (auto &nv : adj[u])
    {
        int v = nv.first, val = nv.second.first, idx = nv.second.second;
        if (v == p)
            continue;
        depth[v] = depth[u] + 1;
        maxRed[0][v] = {val, idx};
        dfs(v, u, adj, par, maxRed, depth);
    }
}

// Function to query the maximum red edge along the path from u to v.
pair<int, int> queryMaxRed(int u, int v, const vector<vector<int>> &par,
                           const vector<vector<pair<int, int>>> &maxRed, const vector<int> &depth, int LOG)
{
    pair<int, int> res = {0, -1};
    if (depth[u] < depth[v])
        swap(u, v);
    int d = depth[u] - depth[v];
    for (int k = 0; k < LOG; k++)
    {
        if (d & (1 << k))
        {
            if (maxRed[k][u].first > res.first)
                res = maxRed[k][u];
            u = par[k][u];
        }
    }
    if (u == v)
        return res;
    for (int k = LOG - 1; k >= 0; k--)
    {
        if (par[k][u] != par[k][v])
        {
            if (maxRed[k][u].first > res.first)
                res = maxRed[k][u];
            if (maxRed[k][v].first > res.first)
                res = maxRed[k][v];
            u = par[k][u];
            v = par[k][v];
        }
    }
    if (maxRed[0][u].first > res.first)
        res = maxRed[0][u];
    if (maxRed[0][v].first > res.first)
        res = maxRed[0][v];
    return res;
}

// Function to check if an edge is already in the MST.
bool edgeInMST(const Edge &e, const vector<Edge> &mstEdges)
{
    for (const auto &mstE : mstEdges)
    {
        if ((mstE.u == e.u && mstE.v == e.v && mstE.w == e.w && mstE.c == e.c) ||
            (mstE.u == e.v && mstE.v == e.u && mstE.w == e.w && mstE.c == e.c))
            return true;
    }
    return false;
}

vector<pair<pair<Edge, Edge>, int>> findCandidateBlueSwaps(int V, const vector<Edge> &mstEdges,
                                                           const vector<Edge> &allEdges,
                                                           int currentCost, int T)
{
    vector<vector<pair<int, pair<int, int>>>> adj(V);
    for (int i = 0; i < mstEdges.size(); i++)
    {
        const Edge &e = mstEdges[i];
        int val = (e.c == 1) ? e.w : 0;
        adj[e.u].push_back({e.v, {val, i}});
        adj[e.v].push_back({e.u, {val, i}});
    }

    int LOG = ceil(log2(V));
    vector<vector<int>> par(LOG, vector<int>(V, -1));
    vector<vector<pair<int, int>>> maxRed(LOG, vector<pair<int, int>>(V, {0, -1}));
    vector<int> depth(V, 0);

    dfs(0, -1, adj, par, maxRed, depth);

    for (int k = 1; k < LOG; k++)
    {
        for (int u = 0; u < V; u++)
        {
            if (par[k - 1][u] != -1)
            {
                par[k][u] = par[k - 1][par[k - 1][u]];
                pair<int, int> cand1 = maxRed[k - 1][u];
                pair<int, int> cand2 = maxRed[k - 1][par[k - 1][u]];
                maxRed[k][u] = (cand1.first >= cand2.first) ? cand1 : cand2;
            }
        }
    }

    vector<pair<pair<Edge, Edge>, int>> candidates;
    for (const auto &e : allEdges)
    {
        if (e.c != 0 || edgeInMST(e, mstEdges))
            continue;
        pair<int, int> queryRes = queryMaxRed(e.u, e.v, par, maxRed, depth, LOG);
        if (queryRes.first > 0)
        {
            int redEdgeWeight = queryRes.first;
            int redEdgeIdx = queryRes.second;
            int costDiff = e.w - redEdgeWeight;
            if (currentCost + costDiff <= T)
            {
                candidates.push_back({{e, mstEdges[redEdgeIdx]}, costDiff});
            }
        }
    }
    return candidates;
}

bool comparator2(const Edge &e, const Edge &redToRemove)
{
    return ((e.u == redToRemove.u && e.v == redToRemove.v && e.w == redToRemove.w && e.c == redToRemove.c) ||
            (e.u == redToRemove.v && e.v == redToRemove.u && e.w == redToRemove.w && e.c == redToRemove.c));
}

void performSwapCandidate(vector<Edge> &mstEdges, Edge candidateBlue, Edge redToRemove, int &totalCost, int &redCount)
{
    auto it = find_if(mstEdges.begin(), mstEdges.end(), bind(comparator2, std::placeholders::_1, redToRemove));
    if (it != mstEdges.end())
        mstEdges.erase(it);

    mstEdges.push_back(candidateBlue);

    totalCost = totalCost - redToRemove.w + candidateBlue.w;
    redCount--;
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m, T;
    cin >> n >> m >> T;

    vector<Edge> allEdges(m);
    for (int i = 0; i < m; i++)
        cin >> allEdges[i].u >> allEdges[i].v >> allEdges[i].w >> allEdges[i].c;

    vector<Edge> mstEdges = formMSTEdges(n, allEdges);
    auto [totalCost, redCount] = computeMSTStats(mstEdges);

    if (totalCost <= T)
    {
        while (true)
        {
            auto candidates = findCandidateBlueSwaps(n, mstEdges, allEdges, totalCost, T);
            if (candidates.empty())
                break;

            int bestDiff = INT_MAX;
            pair<Edge, Edge> bestCandidate;
            for (auto &cand : candidates)
            {
                if (cand.second < bestDiff)
                {
                    bestDiff = cand.second;
                    bestCandidate = cand.first;
                }
            }
            if (bestDiff == INT_MAX)
                break;

            performSwapCandidate(mstEdges, bestCandidate.first, bestCandidate.second, totalCost, redCount);
        }
        cout << redCount << "\n"
             << totalCost << "\n";
    }
    else
    {
        cout << "No valid MST under threshold\n";
    }
    return 0;
}
