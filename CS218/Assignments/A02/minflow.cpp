#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
 
const ll INF = 1e18;  // A large number for infinity.
 
// ---------- MAX FLOW (Edmonds-Karp) Implementation ----------
struct Edge {
    int to, rev;
    ll cap;
};
 
struct MaxFlow {
    int n;
    vector<vector<Edge>> graph;
    MaxFlow(int n) : n(n), graph(n) {}
 
    void addEdge(int s, int t, ll cap) {
        Edge a = {t, (int)graph[t].size(), cap};
        Edge b = {s, (int)graph[s].size(), 0};
        graph[s].push_back(a);
        graph[t].push_back(b);
    }
 
    ll bfs(int s, int t, vector<int>& parent, vector<int>& parentEdge) {
        fill(parent.begin(), parent.end(), -1);
        parent[s] = s;
        queue<pair<int, ll>> q;
        q.push({s, INF});
        while (!q.empty()) {
            int cur = q.front().first;
            ll flow = q.front().second;
            q.pop();
            for (int i = 0; i < graph[cur].size(); i++) {
                Edge &e = graph[cur][i];
                if (parent[e.to] == -1 && e.cap > 0) {
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
 
    ll maxFlow(int s, int t) {
        ll flow = 0;
        vector<int> parent(n), parentEdge(n);
        while (ll pushed = bfs(s, t, parent, parentEdge)) {
            flow += pushed;
            int cur = t;
            while (cur != s) {
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
 
// ---------- Main Function ----------
int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    // Input reading.
    int m, n;
    cin >> m >> n;
    
    // Lower and upper bound matrices.
    vector<vector<ll>> L(m, vector<ll>(n)), U(m, vector<ll>(n));
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            cin >> L[i][j];
        }
    }
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            cin >> U[i][j];
        }
    }
    
    // Row constraints.
    vector<ll> r(m), R(m);
    for (int i = 0; i < m; i++){
        cin >> r[i] >> R[i];
    }
    
    // Column constraints.
    vector<ll> c(n), C(n);
    for (int j = 0; j < n; j++){
        cin >> c[j] >> C[j];
    }
    
    // Compute base sum.
    ll baseSum = 0;
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            baseSum += L[i][j];
        }
    }
    
    // Compute transformed row and column constraints.
    vector<ll> rowLower(m), rowUpper(m), colLower(n), colUpper(n);
    for (int i = 0; i < m; i++){
        ll rowSum = 0;
        for (int j = 0; j < n; j++){
            rowSum += L[i][j];
        }
        rowLower[i] = r[i] - rowSum;
        rowUpper[i] = R[i] - rowSum;
        if (rowLower[i] < 0 || rowUpper[i] < rowLower[i]){
            cout << 0 << "\n";
            return 0;
        }
    }
    for (int j = 0; j < n; j++){
        ll colSum = 0;
        for (int i = 0; i < m; i++){
            colSum += L[i][j];
        }
        colLower[j] = c[j] - colSum;
        colUpper[j] = C[j] - colSum;
        if (colLower[j] < 0 || colUpper[j] < colLower[j]){
            cout << 0 << "\n";
            return 0;
        }
    }
    
    // fixedFlow equals baseSum + sum of lower demands on rows, which equals sum(r[i])
    ll fixedFlow = baseSum;
    ll totalRowDemand = 0;
    for (int i = 0; i < m; i++){
        totalRowDemand += rowLower[i];
        // (rowLower[i] = r[i] - (sum of L in row i))
    }
    // totalColDemand is the same as totalRowDemand.
    
    // ---------- Feasibility Check Using the "Old" Network ----------
    // Nodes: rows [0, m-1], columns [m, m+n-1], source s = m+n, sink t = m+n+1.
    int oldN = m + n + 2;
    int s_orig = m + n, t_orig = m + n + 1;
    MaxFlow oldNet(oldN);
    // s -> row nodes: capacity = rowLower[i]
    for (int i = 0; i < m; i++){
        oldNet.addEdge(s_orig, i, rowLower[i]);
    }
    // Row nodes -> column nodes for each cell: capacity = U[i][j] - L[i][j]
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            oldNet.addEdge(i, m + j, U[i][j] - L[i][j]);
        }
    }
    // Column nodes -> t: capacity = colLower[j]
    for (int j = 0; j < n; j++){
        oldNet.addEdge(m + j, t_orig, colLower[j]);
    }
    // Artificial edge (t, s) with infinite capacity.
    oldNet.addEdge(t_orig, s_orig, INF);
    
    // Create transformed network (new network) with two extra nodes: s' and t'.
    int newN = oldN + 2;           // new nodes: s' = oldN, t' = oldN+1.
    int sprime = oldN, tprime = oldN + 1;
    MaxFlow newNet(newN);
    // Copy over all old network edges.
    for (int u = 0; u < oldN; u++){
        for (auto &e : oldNet.graph[u]){
            newNet.addEdge(u, e.to, e.cap);
        }
    }
    // Add edges from s' to every vertex.
    // For row nodes.
    for (int i = 0; i < m; i++){
        newNet.addEdge(sprime, i, rowLower[i]);
    }
    // For column nodes (no incoming demand).
    for (int j = m; j < m + n; j++){
        newNet.addEdge(sprime, j, 0);
    }
    // For source s_orig.
    newNet.addEdge(sprime, s_orig, 0);
    // For sink t_orig: incoming demand equals total column demand (== totalRowDemand).
    ll totalColDemand = 0;
    for (int j = 0; j < n; j++){
        totalColDemand += colLower[j];
    }
    newNet.addEdge(sprime, t_orig, totalColDemand);
 
    // Now add edges from every vertex to t'.
    // For row nodes.
    for (int i = 0; i < m; i++){
        newNet.addEdge(i, tprime, 0);
    }
    // For column nodes: capacity = colLower[j]
    for (int j = m; j < m + n; j++){
        int colIndex = j - m;
        newNet.addEdge(j, tprime, colLower[colIndex]);
    }
    // For source s_orig: outgoing demand equals totalRowDemand.
    newNet.addEdge(s_orig, tprime, totalRowDemand);
    // For sink t_orig: no outgoing demand.
    newNet.addEdge(t_orig, tprime, 0);
    
    // The required flow in the transformed network.
    ll requiredFlow = 2 * totalRowDemand;
    ll flowAchieved = newNet.maxFlow(sprime, tprime);
    
    if (flowAchieved != requiredFlow) {
        // No feasible circulation exists.
        cout << 0 << "\n";
        return 0;
    }
    
    // ---------- Maximum Flow Computation ----------
    // Extra capacity network: add parallel edges for additional flow.
    // Extra edges: from s to rows: capacity = R[i] - r[i] (since rowLower = r[i] - (row sum))
    //              from columns to t: capacity = C[j] - c[j]
    int totalNodes = m + n + 2; // same indexing: s = m+n, t = m+n+1.
    int origS = s_orig, origT = t_orig;
    MaxFlow extraNet(totalNodes);
    // s -> row nodes extra.
    for (int i = 0; i < m; i++){
        extraNet.addEdge(origS, i, (R[i] - r[i]));
    }
    // Row -> column edges remain with capacity = U[i][j] - L[i][j].
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            extraNet.addEdge(i, m + j, U[i][j] - L[i][j]);
        }
    }
    // Column nodes -> t extra.
    for (int j = 0; j < n; j++){
        extraNet.addEdge(m + j, origT, (C[j] - c[j]));
    }
    ll additionalFlow = extraNet.maxFlow(origS, origT);
    ll maxFlowValue = fixedFlow + additionalFlow;
    
    // ---------- Minimum Flow Computation via Binary Search ----------
    // The idea: limit the artificial edge (t, s) to a candidate capacity X.
    // For each candidate X (via binary search), rebuild the old and transformed networks and check
    // whether a feasible circulation (flow = requiredFlow) is obtained.
    // The minimal overall flow will then be: minFlow = fixedFlow + X_min.
    ll loBS = 0, hiBS = INF, bestCandidate = hiBS;
    while (loBS <= hiBS) {
        ll mid = (loBS + hiBS) / 2;
        // Build old network with candidate capacity on (t, s) = mid.
        MaxFlow testOld(oldN);
        // s -> row nodes.
        for (int i = 0; i < m; i++){
            testOld.addEdge(s_orig, i, rowLower[i]);
        }
        // Row -> column edges.
        for (int i = 0; i < m; i++){
            for (int j = 0; j < n; j++){
                testOld.addEdge(i, m + j, U[i][j] - L[i][j]);
            }
        }
        // Column nodes -> t.
        for (int j = 0; j < n; j++){
            testOld.addEdge(m + j, t_orig, colLower[j]);
        }
        // Artificial edge (t, s) with candidate capacity.
        testOld.addEdge(t_orig, s_orig, mid);
 
        // Build transformed network for binary search test.
        int newN_bin = oldN + 2;
        int sprime_bin = oldN, tprime_bin = oldN + 1;
        MaxFlow testNew(newN_bin);
        for (int u = 0; u < oldN; u++){
            for (auto &e : testOld.graph[u]){
                testNew.addEdge(u, e.to, e.cap);
            }
        }
        // s' -> vertices.
        for (int i = 0; i < m; i++){
            testNew.addEdge(sprime_bin, i, rowLower[i]);
        }
        for (int j = m; j < m + n; j++){
            testNew.addEdge(sprime_bin, j, 0);
        }
        testNew.addEdge(sprime_bin, s_orig, 0);
        testNew.addEdge(sprime_bin, t_orig, totalColDemand);
        // Vertices -> t'.
        for (int i = 0; i < m; i++){
            testNew.addEdge(i, tprime_bin, 0);
        }
        for (int j = m; j < m + n; j++){
            int colIndex = j - m;
            testNew.addEdge(j, tprime_bin, colLower[colIndex]);
        }
        testNew.addEdge(s_orig, tprime_bin, totalRowDemand);
        testNew.addEdge(t_orig, tprime_bin, 0);
 
        ll flowTest = testNew.maxFlow(sprime_bin, tprime_bin);
        if(flowTest == requiredFlow) {
            bestCandidate = mid;
            hiBS = mid - 1;
        } else {
            loBS = mid + 1;
        }
    }
    ll minFlowValue = fixedFlow + bestCandidate;
    
    // ---------- Output Results ----------
    cout << "Minimum flow value: " << minFlowValue << "\n";
    cout << "Maximum flow value: " << maxFlowValue << "\n";
    
    return 0;
}
