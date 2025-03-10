#include <bits/stdc++.h>
using namespace std;

#define vi vector<int>
#define vE vector<Edge>
#define pii pair<int, int>

// structure for an edge
struct Edge
{
    int u, v, w, c;
    int id;
};

// structure for a graph
struct Graph
{
    int V, E;
    vE edges;
    Graph(int V, int E)
    {
        this->V = V;
        this->E = E;
    }
    void addEdge(const Edge &e)
    {
        edges.push_back(e);
    }
};

// disjoint set data structure
struct DisjointSet
{
    vi parent, rank;

    DisjointSet(int n)
    {
        parent.resize(n);
        rank.resize(n, 1);
        for (int i = 0; i < n; i++)
            parent[i] = i;
    }

    int findRoot(int x)
    {
        if (parent[x] == x)
        {
            return x;
        }
        else
        {
            return parent[x] = findRoot(parent[x]);
        }
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

// --------------------------------------------------------------------------
//  Here we are considering all the edges according to weights, color will
//  break the tie if weights are equal (blue edges will be preferred)
// --------------------------------------------------------------------------

bool comparator1(const Edge &a, const Edge &b)
{
    if (a.w == b.w)
        return a.c < b.c;
    return a.w < b.w;
}

vE formMSTEdges(int n, vE &edges)
{
    vE mstEdges;
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

pii getResFromMST(const vE &mstEdges)
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

// --------------------------------------------------------------------------
// This section of code finds the blue edge that is not present in the MST formed.
// Since addition of blue edge in MST will form of a cycle, we are finding maximum
// weight red edge in that cycle. We will swap this red edge from out blue edge
// --------------------------------------------------------------------------

void dfs(int node, int parent, const vector<vector<pair<int, pii>>> &adjList,
         vector<vi> &ancestor, vector<vector<pii>> &maxEdge, vi &depth)
{
    ancestor[0][node] = parent;
    for (auto &neighbor : adjList[node])
    {
        int nextNode = neighbor.first;
        int edgeWeight = neighbor.second.first;
        int edgeIndex = neighbor.second.second;

        if (nextNode == parent)
            continue;

        depth[nextNode] = depth[node] + 1;
        maxEdge[0][nextNode] = {edgeWeight, edgeIndex};
        dfs(nextNode, node, adjList, ancestor, maxEdge, depth);
    }
}

// function to find the heaviest red edge in the path from  nodeA to nodeB
pii getMaxRedEdge(int nodeA, int nodeB, const vector<vi> &ancestor,
                  const vector<vector<pii>> &maxRedEdge, const vi &depth, int maxLOG)
{
    pii maxEdge = {0, -1};

    // ensuring that the first node is deeper
    if (depth[nodeA] < depth[nodeB])
        swap(nodeA, nodeB);

    // lofting the first node to the same depth as second node
    int depthDiff = depth[nodeA] - depth[nodeB];
    for (int k = 0; k < maxLOG; k++)
    {
        if (depthDiff & (1 << k))
        {
            if (maxRedEdge[k][nodeA].first > maxEdge.first)
                maxEdge = maxRedEdge[k][nodeA];
            nodeA = ancestor[k][nodeA];
        }
    }

    if (nodeA == nodeB)
        return maxEdge;

    // lifting bith the nodes together
    for (int k = maxLOG - 1; k >= 0; k--)
    {
        if (ancestor[k][nodeA] != ancestor[k][nodeB])
        {
            if (maxRedEdge[k][nodeA].first > maxEdge.first)
                maxEdge = maxRedEdge[k][nodeA];
            if (maxRedEdge[k][nodeB].first > maxEdge.first)
                maxEdge = maxRedEdge[k][nodeB];

            nodeA = ancestor[k][nodeA];
            nodeB = ancestor[k][nodeB];
        }
    }

    if (maxRedEdge[0][nodeA].first > maxEdge.first)
        maxEdge = maxRedEdge[0][nodeA];
    if (maxRedEdge[0][nodeB].first > maxEdge.first)
        maxEdge = maxRedEdge[0][nodeB];

    return maxEdge;
}

// function to find all the blue edge present for swapping
vector<pair<pair<Edge, Edge>, int>> findAllBlueSwaps(
    int V, const vE &mstEdges, const vE &allEdges,
    int currentMSTCost, int T, const vector<bool> &isInMST)
{
    vector<vector<pair<int, pii>>> mstAdjList(V);
    for (int i = 0; i < mstEdges.size(); i++)
    {
        const Edge &edge = mstEdges[i];
        int redEdgeWeight = (edge.c == 1) ? edge.w : 0;
        mstAdjList[edge.u].push_back({edge.v, {redEdgeWeight, i}});
        mstAdjList[edge.v].push_back({edge.u, {redEdgeWeight, i}});
    }

    int maxLog = ceil(log2(V));
    vector<vector<int>> parent(maxLog, vector<int>(V, -1));
    vector<vector<pii>> maxRedEdge(maxLog, vector<pii>(V, {0, -1}));
    vector<int> nodeDepth(V, 0);

    dfs(0, -1, mstAdjList, parent, maxRedEdge, nodeDepth);

    for (int k = 1; k < maxLog; k++)
    {
        for (int node = 0; node < V; node++)
        {
            if (parent[k - 1][node] != -1)
            {
                parent[k][node] = parent[k - 1][parent[k - 1][node]];
                pii candidate1 = maxRedEdge[k - 1][node];
                pii candidate2 = maxRedEdge[k - 1][parent[k - 1][node]];
                maxRedEdge[k][node] = (candidate1.first >= candidate2.first) ? candidate1 : candidate2;
            }
        }
    }

    vector<pair<pair<Edge, Edge>, int>> candidateSwaps;
    for (const auto &edge : allEdges)
    {
        if (edge.c != 0 || isInMST[edge.id])
            continue;
        pii maxRedOnPath = getMaxRedEdge(edge.u, edge.v, parent, maxRedEdge, nodeDepth, maxLog);
        if (maxRedOnPath.first > 0)
        {
            int redEdgeWeight = maxRedOnPath.first;
            int redEdgeIndex = maxRedOnPath.second;
            int costDifference = edge.w - redEdgeWeight;
            if (currentMSTCost + costDifference <= T)
            {
                candidateSwaps.push_back({{edge, mstEdges[redEdgeIndex]}, costDifference});
            }
        }
    }
    return candidateSwaps;
}

// --------------------------------------------------------------------------
//  In this section of the code, we are actually performing the swap
//  between a blue edge and a red  edge. This will reduce the weight
//  of the tree and help us making the weight less than threshold.
// --------------------------------------------------------------------------

bool comparator2(const Edge &e, const Edge &redToRemove)
{
    return ((e.u == redToRemove.u && e.v == redToRemove.v && e.w == redToRemove.w && e.c == redToRemove.c) ||
            (e.u == redToRemove.v && e.v == redToRemove.u && e.w == redToRemove.w && e.c == redToRemove.c));
}

void doSwap(vE &mstEdges, Edge candidateBlue, Edge redToRemove, int &totalCost, int &redCount)
{
    auto it = find_if(mstEdges.begin(), mstEdges.end(), bind(comparator2, std::placeholders::_1, redToRemove));
    if (it != mstEdges.end())
        mstEdges.erase(it);

    mstEdges.push_back(candidateBlue);

    totalCost = totalCost - redToRemove.w + candidateBlue.w;
    redCount--;
}

// --------------------------------------------------------------------------
//      Main function
// --------------------------------------------------------------------------

int main()
{

    // for faster IO
    ios::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);

    int n, m, T;
    cin >> n >> m >> T;

    vE allEdges(m);
    for (int i = 0; i < m; i++)
    {
        cin >> allEdges[i].u >> allEdges[i].v >> allEdges[i].w >> allEdges[i].c;
        allEdges[i].id = i;
    }

    vector<bool> isMST(m, false);

    vE mstEdges = formMSTEdges(n, allEdges);
    for (auto &e : mstEdges)
        isMST[e.id] = true;

    // mst information after first step
    auto [totalCost, redCount] = getResFromMST(mstEdges);

    // looking to swap the blue edges from red edges
    if (totalCost <= T)
    {
        while (true)
        {
            auto candidates = findAllBlueSwaps(n, mstEdges, allEdges, totalCost, T, isMST);
            if (candidates.empty())
                break;

            int bestDiff = INT_MAX;
            pair<Edge, Edge> bestCandidate;
            
            // finding the best edge among all possible edges
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

            // swap the edges
            doSwap(mstEdges, bestCandidate.first, bestCandidate.second, totalCost, redCount);
            
            isMST[bestCandidate.second.id] = false;
            isMST[bestCandidate.first.id] = true;
        }
        cout << redCount << "\n"
             << totalCost << "\n";
    }
    else
    {
        cout << "No MST found!\n";
    }

    return 0;
}