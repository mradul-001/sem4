#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

const ll INF = 1e18; // A large number for infinity.

// ---------- MAX FLOW (Edmonds-Karp) Implementation ----------
struct Edge
{
    int to, rev;
    ll cap;
};

struct MaxFlow
{
    int n;
    vector<vector<Edge>> graph;
    MaxFlow(int n) : n(n), graph(n) {}

    void addEdge(int s, int t, ll cap)
    {
        Edge a = {t, (int)graph[t].size(), cap};
        Edge b = {s, (int)graph[s].size(), 0};
        graph[s].push_back(a);
        graph[t].push_back(b);
    }

    ll bfs(int s, int t, vector<int> &parent, vector<int> &parentEdge)
    {
        fill(parent.begin(), parent.end(), -1);
        parent[s] = s;
        queue<pair<int, ll>> q;
        q.push({s, INF});
        while (!q.empty())
        {
            int cur = q.front().first;
            ll flow = q.front().second;
            q.pop();
            for (int i = 0; i < graph[cur].size(); i++)
            {
                Edge &e = graph[cur][i];
                if (parent[e.to] == -1 && e.cap > 0)
                {
                    parent[e.to] = cur;
                    parentEdge[e.to] = i;
                    ll new_flow = min(flow, e.cap);
                    if (e.to == t)
                        return new_flow;
                    q.push({e.to, new_flow});
                }
            }
        }
        return 0;
    }

    ll maxFlow(int s, int t)
    {
        ll flow = 0;
        vector<int> parent(n), parentEdge(n);
        while (ll pushed = bfs(s, t, parent, parentEdge))
        {
            flow += pushed;
            int cur = t;
            while (cur != s)
            {
                int pe = parentEdge[cur];
                int prev = parent[cur];
                graph[prev][pe].cap -= pushed;
                graph[cur][graph[prev][pe].rev].cap += pushed;
                cur = prev;
            }
        }
        return flow;
    }
};

// ---------- Main helper.cpp begins here ----------
int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int m, n;
    cin >> m >> n;

    // Read the lower bound matrix L (denoted as `i,j in the assignment)
    vector<vector<ll>> L(m, vector<ll>(n));
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cin >> L[i][j];
        }
    }

    // Read the upper bound matrix U (denoted as u(i,j) in the assignment)
    vector<vector<ll>> U(m, vector<ll>(n));
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cin >> U[i][j];
        }
    }

    // Read row constraints: for each row i, r_i and R_i.
    vector<ll> r(m), R(m);
    for (int i = 0; i < m; i++)
    {
        cin >> r[i] >> R[i];
    }

    // Read column constraints: for each column j, c_j and C_j.
    vector<ll> c(n), C(n);
    for (int j = 0; j < n; j++)
    {
        cin >> c[j] >> C[j];
    }

    // Compute base sum = ∑ L[i][j] (not used directly here, but part of overall problem)
    ll baseSum = 0;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            baseSum += L[i][j];

    // Transform the row and column constraints by subtracting the base lower bounds.
    // For each row i, define:
    //   rowLower[i] = r[i] - (sum over j of L[i][j])
    //   rowUpper[i] = R[i] - (sum over j of L[i][j])
    vector<ll> rowLower(m), rowUpper(m);
    for (int i = 0; i < m; i++)
    {
        ll rowSum = 0;
        for (int j = 0; j < n; j++)
        {
            rowSum += L[i][j];
        }
        rowLower[i] = r[i] - rowSum;
        rowUpper[i] = R[i] - rowSum;
        if (rowLower[i] < 0 || rowUpper[i] < rowLower[i])
        {
            cout << 0 << "\n";
            return 0;
        }
    }

    // For each column j, define:
    //   colLower[j] = c[j] - (sum over i of L[i][j])
    //   colUpper[j] = C[j] - (sum over i of L[i][j])
    vector<ll> colLower(n), colUpper(n);
    for (int j = 0; j < n; j++)
    {
        ll colSum = 0;
        for (int i = 0; i < m; i++)
        {
            colSum += L[i][j];
        }
        colLower[j] = c[j] - colSum;
        colUpper[j] = C[j] - colSum;
        if (colLower[j] < 0 || colUpper[j] < colLower[j])
        {
            cout << 0 << "\n";
            return 0;
        }
    }

    // In our “old” network with demands, we use:
    // - For each row i: edge (s, i) with capacity = rowLower[i] and demand = rowLower[i].
    // - For each column j: edge (m+j, t) with capacity = colLower[j] and demand = colLower[j].
    // - For each cell (i,j): edge (i, m+j) with capacity = U[i][j] - L[i][j] and demand = 0.
    // - And edge (t, s) with capacity = INF and demand = 0.
    int oldN = m + n + 2; // vertices: row nodes [0, m-1], column nodes [m, m+n-1], s = m+n, t = m+n+1.
    int s = m + n, t = m + n + 1;
    MaxFlow oldNet(oldN);

    // Add edge (s, i) for each row i.
    for (int i = 0; i < m; i++)
    {
        // Capacity is rowLower[i], and since demand equals rowLower[i], the transformed capacity becomes:
        // c' = c(s, i) - d(s, i) = rowLower[i] - rowLower[i] = 0.
        oldNet.addEdge(s, i, rowLower[i]);
    }

    // Add edge (i, m+j) for each cell (i,j).
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            ll cap = U[i][j] - L[i][j]; // demand is 0.
            oldNet.addEdge(i, m + j, cap);
        }
    }

    // Add edge (m+j, t) for each column j.
    for (int j = 0; j < n; j++)
    {
        // Similarly, capacity becomes: colLower[j] - colLower[j] = 0.
        oldNet.addEdge(m + j, t, colLower[j]);
    }

    // Add edge (t, s) with infinite capacity.
    oldNet.addEdge(t, s, INF);

    // --- TRANSFORM TO NEW NETWORK ---
    // New network will include all old nodes plus two new nodes s' and t'.
    // Let new indices:
    int newN = oldN + 2; // new nodes: s' = oldN, t' = oldN+1.
    int sprime = oldN, tprime = oldN + 1;
    MaxFlow newNet(newN);

    // For every edge in the old network, add it with transformed capacity:
    //   For each original edge (u, v) with capacity c and demand d,
    //   add an edge (u, v) in newNet with capacity c - d.
    for (int u = 0; u < oldN; u++)
    {
        for (auto &e : oldNet.graph[u])
        {
            // We assume that for our forced edges, the demand equals the entire capacity,
            // hence for (s, i) and (m+j, t) the new capacity will be 0.
            // For other edges, demand is 0 so capacity remains the same.
            ll cap_trans = e.cap; // since e.cap is already (c - d) in our oldNet construction.
            // We add the edge in newNet from u to e.to with the same capacity.
            newNet.addEdge(u, e.to, cap_trans);
        }
    }

    // Now, for every vertex v in the old network (v = 0 .. oldN-1), add:
    //   Edge (s', v) with capacity = Σ_{u in V} d((u, v)).
    //   Edge (v, t') with capacity = Σ_{w in V} d((v, w)).
    // In our case:
    // - For a row node i (0 ≤ i < m): the only incoming edge with demand is (s, i) with demand = rowLower[i].
    //   So add edge (s', i) with capacity = rowLower[i].
    // - For a column node j (m ≤ j < m+n): no incoming edges have demand → capacity = 0.
    // - For vertex s: incoming edges: none (capacity = 0). Outgoing: (s, i) edges have demand rowLower[i] → so add edge (s, t') with capacity = sum rowLower[i].
    // - For vertex t: incoming: from (m+j, t) edges with demand colLower[j] → add edge (s', t) with capacity = sum colLower[j].
    ll totalRowDemand = 0;
    for (int i = 0; i < m; i++)
    {
        totalRowDemand += rowLower[i];
        newNet.addEdge(sprime, i, rowLower[i]);
    }
    for (int j = m; j < m + n; j++)
    {
        // no incoming demand → capacity = 0
        newNet.addEdge(sprime, j, 0);
    }
    // For vertex s (old node index s)
    newNet.addEdge(sprime, s, 0); // s has no incoming demand.
    // For vertex t (old node index t)
    ll totalColDemand = 0;
    for (int j = 0; j < n; j++)
    {
        totalColDemand += colLower[j];
    }
    newNet.addEdge(sprime, t, totalColDemand); // t gets demand from column edges.

    // Now add edges from every old vertex v to t' with capacity = sum of d on outgoing edges.
    // For row nodes i: outgoing edge (i, m+j) all have demand 0.
    for (int i = 0; i < m; i++)
    {
        newNet.addEdge(i, tprime, 0);
    }
    // For column nodes j: outgoing edge (m+j, t) has demand = colLower[j].
    for (int j = m; j < m + n; j++)
    {
        // Find the vertex’s corresponding column index j-m.
        int colIndex = j - m;
        newNet.addEdge(j, tprime, colLower[colIndex]);
    }
    // For vertex s: outgoing: edge (s, i) with demand rowLower[i] → sum = totalRowDemand.
    newNet.addEdge(s, tprime, totalRowDemand);
    // For vertex t: outgoing: (t, s) has demand 0.
    newNet.addEdge(t, tprime, 0);

    // The total required flow from s' is the sum of capacities on edges leaving s'.
    ll requiredFlow = 0;
    // Sum for all v in old network:
    // For row nodes: sum_{i} rowLower[i] = totalRowDemand.
    // For column nodes: 0.
    // For s: 0.
    // For t: totalColDemand.
    requiredFlow = totalRowDemand + totalColDemand;
    // (Note: since totalRowDemand == totalColDemand, requiredFlow = 2 * totalRowDemand)

    // Compute maximum flow from s' to t' in newNet.
    ll flowAchieved = newNet.maxFlow(sprime, tprime);

    // Check feasibility: every edge outgoing from s' must be saturated.
    if (flowAchieved == requiredFlow)
    {
        // --- Start Additional Flow Phase ---
        // Remove/ignore the artificial edge (t, s) by not using its residual capacity.
        // We now add new parallel edges to allow additional flow.
        // We reuse the old network (or build on top of newNet) but now with the proper extra capacities.
        // We'll create a new network that starts with the lower-bound flow fixed.
        int totalNodes = m + n + 2; // nodes: row[0..m-1], col[m..m+n-1], s, t.
        int origS = m + n, origT = m + n + 1;
        MaxFlow combined(totalNodes);

        // Add extra edges from s -> row nodes with capacity R[i]-r[i].
        for (int i = 0; i < m; i++)
        {
            // Note: rowLower[i] = r[i] - rowSum, rowUpper[i] = R[i] - rowSum.
            // Extra capacity = rowUpper[i] - rowLower[i] = R[i] - r[i].
            combined.addEdge(origS, i, (R[i] - r[i]));
        }

        // Add the cell edges (row i -> column j) with capacity U[i][j]-L[i][j]
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                combined.addEdge(i, m + j, U[i][j] - L[i][j]);
            }
        }

        // Add extra edges from column nodes -> t with capacity C[j]-c[j].
        for (int j = 0; j < n; j++)
        {
            combined.addEdge(m + j, origT, (C[j] - c[j]));
        }

        // Run max-flow on the additional network.
        ll additionalFlow = combined.maxFlow(origS, origT);

        // The fixed flow from feasibility equals baseSum + sum(rowLower[i]) = baseSum + sum(r[i]-rowSum)
        ll fixedFlow = baseSum;
        for (int i = 0; i < m; i++)
        {
            ll rowSum = 0;
            for (int j = 0; j < n; j++)
            {
                rowSum += L[i][j];
            }
            fixedFlow += (r[i] - rowSum);
        }

        ll totalFlow = fixedFlow + additionalFlow;

        // ---------------------------
        // --- Minimal Flow Phase ---
        // ---------------------------
        // Instead of allowing unlimited extra flow (via INF on (t,s)),
        // we binary search for the smallest capacity X on (t,s) that
        // still yields a saturating flow (i.e. all demands are met).
        // This candidate X then gives the minimal additional flow.

        // Set an upper bound on extra flow (the maximum extra that could possibly be pushed)
        ll extraUpper = 0;
        for (int i = 0; i < m; i++)
        {
            extraUpper += (R[i] - r[i]);
        }

        // Binary search on candidate capacity for edge (t, s)
        ll lo = 0, hi = extraUpper, ans = hi;
        while (lo <= hi)
        {
            ll mid = lo + (hi - lo) / 2;

            // --- Rebuild the "old network" with candidate capacity 'mid' ---
            int oldN2 = m + n + 2; // nodes: rows, cols, s, t.
            int s2 = m + n, t2 = m + n + 1;
            MaxFlow oldNet2(oldN2);

            // Row edges: (s, i) with capacity = rowLower[i]
            for (int i = 0; i < m; i++)
            {
                oldNet2.addEdge(s2, i, rowLower[i]);
            }
            // Cell edges: (i, m+j) with capacity = U[i][j]-L[i][j]
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    oldNet2.addEdge(i, m + j, U[i][j] - L[i][j]);
                }
            }
            // Column edges: (m+j, t) with capacity = colLower[j]
            for (int j = 0; j < n; j++)
            {
                oldNet2.addEdge(m + j, t2, colLower[j]);
            }
            // The candidate edge: (t, s) with capacity = mid (instead of INF).
            oldNet2.addEdge(t2, s2, mid);

            // --- Transform to new network, identical to before ---
            int newN2 = oldN2 + 2; // new nodes: s' and t'
            int sprime2 = oldN2, tprime2 = oldN2 + 1;
            MaxFlow newNet2(newN2);
            for (int u = 0; u < oldN2; u++)
            {
                for (auto &e : oldNet2.graph[u])
                {
                    newNet2.addEdge(u, e.to, e.cap);
                }
            }
            ll totalRowDemand2 = 0;
            for (int i = 0; i < m; i++)
            {
                totalRowDemand2 += rowLower[i];
                newNet2.addEdge(sprime2, i, rowLower[i]);
            }
            for (int j = m; j < m + n; j++)
            {
                newNet2.addEdge(sprime2, j, 0);
            }
            newNet2.addEdge(sprime2, s2, 0);
            ll totalColDemand2 = 0;
            for (int j = 0; j < n; j++)
            {
                totalColDemand2 += colLower[j];
            }
            newNet2.addEdge(sprime2, t2, totalColDemand2);
            for (int i = 0; i < m; i++)
            {
                newNet2.addEdge(i, tprime2, 0);
            }
            for (int j = m; j < m + n; j++)
            {
                int colIdx = j - m;
                newNet2.addEdge(j, tprime2, colLower[colIdx]);
            }
            newNet2.addEdge(s2, tprime2, totalRowDemand2);
            newNet2.addEdge(t2, tprime2, 0);

            ll requiredFlow2 = totalRowDemand2 + totalColDemand2; // note: these are equal.
            ll flow2 = newNet2.maxFlow(sprime2, tprime2);

            if (flow2 == requiredFlow2)
            {
                ans = mid;
                hi = mid - 1;
            }
            else
            {
                lo = mid + 1;
            }
        }

        ll minFlow = fixedFlow + ans;
        cout << 1 << endl;
        cout << "Maximum flow value: " << totalFlow << "\n";
        cout << "Minimum flow value: " << minFlow << "\n";
    }
    else
        cout << 0 << "\n";

    return 0;
}
