#include <bits/stdc++.h>

using namespace std;
#include <iostream>
#include <vector>
#include <queue>
#include <cstring>
#include <algorithm>

using namespace std;
// edmond karp taken from chatgpt
class Graph
{
private:
	int n; // number of nodes
	vector<vector<int>> capacities;
	vector<vector<int>> flowPassed;
	vector<vector<int>> adj;
	vector<int> parentsList;
	vector<int> currentPathCapacity;
	int bfs(int startNode, int endNode)
	{
		fill(parentsList.begin(), parentsList.end(), -1);
		fill(currentPathCapacity.begin(), currentPathCapacity.end(), 0);
		queue<int> q;
		q.push(startNode);
		parentsList[startNode] = -2;
		currentPathCapacity[startNode] = INT_MAX;
		while (!q.empty())
		{
			int currentNode = q.front();
			q.pop();
			for (int to : adj[currentNode])
			{
				if (parentsList[to] == -1 &&
					capacities[currentNode][to] - flowPassed[currentNode][to] > 0)
				{
					parentsList[to] = currentNode;
					currentPathCapacity[to] = min(
						currentPathCapacity[currentNode],
						capacities[currentNode][to] - flowPassed[currentNode][to]);
					if (to == endNode)
					{
						return currentPathCapacity[endNode];
					}
					q.push(to);
				}
			}
		}
		return 0;
	}

public:
	Graph(int nodes) : n(nodes)
	{
		capacities.assign(n, vector<int>(n, 0));
		flowPassed.assign(n, vector<int>(n, 0));
		adj.resize(n);
		parentsList.resize(n);
		currentPathCapacity.resize(n);
	}
	void addEdge(int from, int to, int capacity)
	{
		capacities[from][to] = capacity;
		adj[from].push_back(to);
		adj[to].push_back(from); // add reverse edge for residual graph
	}
	int edmondsKarp(int startNode, int endNode, vector<vector<int>> &resultFlow)
	{
		int maxFlow = 0;
		while (true)
		{
			int flow = bfs(startNode, endNode);
			if (flow == 0)
				break;
			maxFlow += flow;
			int currentNode = endNode;
			while (currentNode != startNode)
			{
				int prevNode = parentsList[currentNode];
				flowPassed[prevNode][currentNode] += flow;
				flowPassed[currentNode][prevNode] -= flow;
				currentNode = prevNode;
			}
		}
		resultFlow = flowPassed; // Return the final flow graph
		return maxFlow;
	}
};

int main()
{
	int m, n;
	cin >> m;
	cin >> n;
	std::vector<std::vector<int>> lower;
	std::vector<std::vector<int>> upper;
	std::vector<int> rowL;
	std::vector<int> rowU;
	std::vector<int> colL;
	std::vector<int> colU;
	int temp;
	for (int i = 0; i < m; i++)
	{
		std::vector<int> tempVector;
		for (int j = 0; j < n; j++)
		{
			cin >> temp;
			tempVector.push_back(temp);
		}
		lower.push_back(tempVector);
	}
	for (int i = 0; i < m; i++)
	{
		std::vector<int> tempVector;
		for (int j = 0; j < n; j++)
		{
			cin >> temp;
			tempVector.push_back(temp);
		}
		upper.push_back(tempVector);
	}
	for (int i = 0; i < m; i++)
	{
		cin >> temp;
		rowL.push_back(temp);
		cin >> temp;
		rowU.push_back(temp);
	}
	for (int j = 0; j < n; j++)
	{
		cin >> temp;
		colL.push_back(temp);
		cin >> temp;
		colU.push_back(temp);
	}
	int number_of_vertices = 2 * m + 2 * m * n + 2 * n + 2 + 2;
	int total = 0;
	Graph g(number_of_vertices);
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			g.addEdge(2 * i * n + 2 * j, 2 * i * n + 2 * j + 1, upper[i][j] - lower[i][j]);
			total = total + lower[i][j];
			g.addEdge(2 * i * n + 2 * j, 2 * m + 2 * m * n + 2 * n + 3, lower[i][j]);
			g.addEdge(2 * m + 2 * m * n + 2 * n + 2, 2 * i * n + 2 * j + 1, lower[i][j]);
		}
	}
	for (int i = 0; i < 2 * m; i = i + 2)
	{
		if (rowU[i / 2] - rowL[i / 2] < 0)
		{
			return 0;
		}
		total = total + rowL[i / 2];
		g.addEdge(2 * m * n + i, 2 * m * n + i + 1, rowU[i / 2] - rowL[i / 2]);
		g.addEdge(2 * m * n + i, 2 * m + 2 * m * n + 2 * n + 3, rowL[i / 2]);
		g.addEdge(2 * m + 2 * m * n + 2 * n + 2, 2 * m * n + i + 1, rowL[i / 2]);
	}
	for (int i = 0; i < 2 * n; i = i + 2)
	{
		if (colU[i / 2] - colL[i / 2] < 0)
		{
			return 0;
		}
		total = total + colL[i / 2];
		g.addEdge(2 * m * n + 2 * m + i, 2 * m * n + 2 * m + i + 1, colU[i / 2] - colL[i / 2]);
		g.addEdge(2 * m * n + 2 * m + i, 2 * m + 2 * m * n + 2 * n + 3, colL[i / 2]);
		g.addEdge(2 * m + 2 * m * n + 2 * n + 2, 2 * m * n + 2 * m + i + 1, colL[i / 2]);
	}
	for (int i = 0; i < m; i++)
	{
		g.addEdge(2 * m * n + 2 * m + 2 * n, 2 * m * n + 2 * i, INT_MAX);
	}
	for (int i = 0; i < n; i++)
	{
		g.addEdge(2 * m * n + 2 * m + 2 * i + 1, 2 * m * n + 2 * m + 2 * n + 1, INT_MAX);
	}
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			g.addEdge(2 * m * n + 2 * i + 1, 2 * i * n + 2 * j, INT_MAX);
		}
	}
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			g.addEdge(2 * j * n + 2 * i + 1, 2 * m * n + 2 * m + 2 * i, INT_MAX);
		}
	}
	g.addEdge(2 * m * n + 2 * m + 2 * n + 1, 2 * m * n + 2 * m + 2 * n, INT_MAX);
	vector<vector<int>> flow;
	int max_flow = g.edmondsKarp(2 * m * n + 2 * n + 2 * m + 2, 2 * m * n + 2 * n + 2 * m + 3, flow);
	if (total != max_flow)
	{
		cout << 0 << endl;
		return 0;
	}
	int flower = 0;
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			flower = flower + flow[2 * m * n + 2 * i + 1][2 * i * n + 2 * j];
		}
	}
	number_of_vertices = 2 * m * n + 2 * n + 2 * m + 2;
	Graph g2(number_of_vertices);
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			g2.addEdge(2 * i * n + 2 * j, 2 * i * n + 2 * j + 1, upper[i][j] - (lower[i][j] + flow[2 * i * n + 2 * j][2 * i * n + 2 * j + 1]));
		}
	}
	for (int i = 0; i < 2 * m; i = i + 2)
	{
		g2.addEdge(2 * m * n + i, 2 * m * n + i + 1, rowU[i / 2] - (rowL[i / 2] + flow[2 * m * n + i][2 * m * n + i + 1]));
	}
	for (int i = 0; i < 2 * n; i = i + 2)
	{
		g2.addEdge(2 * m * n + 2 * m + i, 2 * m * n + 2 * m + i + 1, colU[i / 2] - (colL[i / 2] + flow[2 * m * n + 2 * m + i][2 * m * n + 2 * m + i + 1]));
	}
	for (int i = 0; i < m; i++)
	{
		g2.addEdge(2 * m * n + 2 * m + 2 * n, 2 * m * n + 2 * i, INT_MAX);
	}
	for (int i = 0; i < n; i++)
	{
		g2.addEdge(2 * m * n + 2 * m + 2 * i + 1, 2 * m * n + 2 * m + 2 * n + 1, INT_MAX);
	}
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			g2.addEdge(2 * m * n + 2 * i + 1, 2 * i * n + 2 * j, INT_MAX);
		}
	}
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			g2.addEdge(2 * j * n + 2 * i + 1, 2 * m * n + 2 * m + 2 * i, INT_MAX);
		}
	}
	vector<vector<int>> flow2;
	int max_out = g2.edmondsKarp(2 * m * n + 2 * n + 2 * m, 2 * m * n + 2 * n + 2 * m + 1, flow2);
	cout << 1 << endl;
	cout << max_out + flower << endl;
	cout << flower << endl;
}