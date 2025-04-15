#include <bits/stdc++.h>
using namespace std;

// function to add an edge in adj list
void addEdge(int fromVertex, int toVertex, int edgeCapacity, vector<vector<int>> &capacityMatrix, vector<vector<int>> &adjacencyList)
{
    capacityMatrix[fromVertex][toVertex] += edgeCapacity;
    adjacencyList[fromVertex].push_back(toVertex);
    adjacencyList[toVertex].push_back(fromVertex);
}

// function to run edmondskarp algo
int edmondsKarp(int totalVertices, int sourceVertex, int sinkVertex, vector<vector<int>> &capacityMatrix,
                vector<vector<int>> &adjacencyList, vector<vector<int>> &flowMatrix)
{
    flowMatrix.assign(totalVertices, vector<int>(totalVertices, 0));
    int totalMaximumFlow = 0;

    while (true)
    {
        vector<int> parentVertex(totalVertices, -1);
        vector<int> edgeFlowFromSource(totalVertices, 0);
        queue<int> bfsQueue;

        parentVertex[sourceVertex] = sourceVertex;
        edgeFlowFromSource[sourceVertex] = INT_MAX;
        bfsQueue.push(sourceVertex);

        while (!bfsQueue.empty())
        {
            int currentVertex = bfsQueue.front();
            bfsQueue.pop();

            for (int adjacentVertex : adjacencyList[currentVertex])
            {
                if (parentVertex[adjacentVertex] == -1 && capacityMatrix[currentVertex][adjacentVertex] - flowMatrix[currentVertex][adjacentVertex] > 0)
                {
                    parentVertex[adjacentVertex] = currentVertex;
                    edgeFlowFromSource[adjacentVertex] = min(edgeFlowFromSource[currentVertex], capacityMatrix[currentVertex][adjacentVertex] - flowMatrix[currentVertex][adjacentVertex]);
                    if (adjacentVertex == sinkVertex)
                        break;
                    bfsQueue.push(adjacentVertex);
                }
            }
        }

        if (parentVertex[sinkVertex] == -1)
            break;

        int pathFlow = edgeFlowFromSource[sinkVertex];
        totalMaximumFlow += pathFlow;

        int currentVertex = sinkVertex;
        while (currentVertex != sourceVertex)
        {
            int previousVertex = parentVertex[currentVertex];
            flowMatrix[previousVertex][currentVertex] += pathFlow;
            flowMatrix[currentVertex][previousVertex] -= pathFlow;
            currentVertex = previousVertex;
        }
    }

    return totalMaximumFlow;
}

int main()
{
    // for faster IO
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int numRows, numCols;
    cin >> numRows >> numCols;

    vector<vector<int>> cellLowerBounds(numRows, vector<int>(numCols)), cellUpperBounds(numRows, vector<int>(numCols));
    vector<int> rowLowerBounds(numRows), rowUpperBounds(numRows), colLowerBounds(numCols), colUpperBounds(numCols);

    // take inputs
    for (int i = 0; i < numRows; i++)
        for (int j = 0; j < numCols; j++)
            cin >> cellLowerBounds[i][j];

    for (int i = 0; i < numRows; i++)
        for (int j = 0; j < numCols; j++)
            cin >> cellUpperBounds[i][j];

    for (int i = 0; i < numRows; i++)
        cin >> rowLowerBounds[i] >> rowUpperBounds[i];

    for (int j = 0; j < numCols; j++)
        cin >> colLowerBounds[j] >> colUpperBounds[j];

    int totalVertices = 2 * numRows * numCols + numRows + numCols + 4;
    int phase1Source = totalVertices - 2;
    int phase1Sink = totalVertices - 1;
    int rowSuperSourceVertex = 2 * numRows * numCols + numRows + numCols;
    int columnSuperSinkVertex = rowSuperSourceVertex + 1;
    
    vector<vector<int>> capacityMatrixPhase1(totalVertices, vector<int>(totalVertices, 0));
    vector<vector<int>> adjacencyListPhase1(totalVertices);
    int totalRequiredMinimumFlow = 0;

    for (int i = 0; i < numRows; i++)
    {
        for (int j = 0; j < numCols; j++)
        {
            int cellInVertex = 2 * i * numCols + 2 * j;
            int cellOutVertex = cellInVertex + 1;

            totalRequiredMinimumFlow += cellLowerBounds[i][j];
            addEdge(cellInVertex, cellOutVertex, cellUpperBounds[i][j] - cellLowerBounds[i][j], capacityMatrixPhase1, adjacencyListPhase1);
            addEdge(cellInVertex, phase1Sink, cellLowerBounds[i][j], capacityMatrixPhase1, adjacencyListPhase1);
            addEdge(phase1Source, cellOutVertex, cellLowerBounds[i][j], capacityMatrixPhase1, adjacencyListPhase1);
        }
    }
    for (int i = 0; i < numRows; i++)
    {
        int rowInVertex = rowSuperSourceVertex;
        int rowOutVertex = 2 * numRows * numCols + i;
        totalRequiredMinimumFlow += rowLowerBounds[i];
        addEdge(rowInVertex, rowOutVertex, rowUpperBounds[i] - rowLowerBounds[i], capacityMatrixPhase1, adjacencyListPhase1);
        addEdge(rowInVertex, phase1Sink, rowLowerBounds[i], capacityMatrixPhase1, adjacencyListPhase1);
        addEdge(phase1Source, rowOutVertex, rowLowerBounds[i], capacityMatrixPhase1, adjacencyListPhase1);
    }

    for (int j = 0; j < numCols; j++)
    {
        int colInVertex = 2 * numRows * numCols + numRows + j;
        int colOutVertex = columnSuperSinkVertex;
        totalRequiredMinimumFlow += colLowerBounds[j];
        addEdge(colInVertex, colOutVertex, colUpperBounds[j] - colLowerBounds[j], capacityMatrixPhase1, adjacencyListPhase1);
        addEdge(colInVertex, phase1Sink, colLowerBounds[j], capacityMatrixPhase1, adjacencyListPhase1);
        addEdge(phase1Source, colOutVertex, colLowerBounds[j], capacityMatrixPhase1, adjacencyListPhase1);
    }
    for (int i = 0; i < numRows; i++)
    {
        int rowOutVertex = 2 * numRows * numCols + i;
        for (int j = 0; j < numCols; j++)
        {
            int cellInVertex = 2 * i * numCols + 2 * j;
            addEdge(rowOutVertex, cellInVertex, INT_MAX, capacityMatrixPhase1, adjacencyListPhase1);
        }
    }

    for (int j = 0; j < numCols; j++)
    {
        int colInVertex = 2 * numRows * numCols + numRows + j;
        for (int i = 0; i < numRows; i++)
        {
            int cellOutVertex = 2 * i * numCols + 2 * j + 1;
            addEdge(cellOutVertex, colInVertex, INT_MAX, capacityMatrixPhase1, adjacencyListPhase1);
        }
    }

    addEdge(columnSuperSinkVertex, rowSuperSourceVertex, INT_MAX, capacityMatrixPhase1, adjacencyListPhase1);

    vector<vector<int>> flowMatrixPhase1;
    int maxFlowPhase1 = edmondsKarp(totalVertices, phase1Source, phase1Sink, capacityMatrixPhase1, adjacencyListPhase1, flowMatrixPhase1);
    if (totalRequiredMinimumFlow != maxFlowPhase1)
    {
        cout << 0 << "\n";
        return 0;
    }
    int preSatisfiedFlow = 0;
    for (int i = 0; i < numRows; i++)
    {
        int rowOutVertex = 2 * numRows * numCols + i;
        preSatisfiedFlow += flowMatrixPhase1[rowSuperSourceVertex][rowOutVertex] + rowLowerBounds[i];
    }
    vector<vector<int>> capacityMatrixPhase2(totalVertices, vector<int>(totalVertices, 0));
    vector<vector<int>> adjacencyListPhase2(totalVertices);

    for (int i = 0; i < numRows; i++)
    {
        for (int j = 0; j < numCols; j++)
        {
            int cellInVertex = 2 * i * numCols + 2 * j;
            int cellOutVertex = cellInVertex + 1;
            int usedFlow = flowMatrixPhase1[cellInVertex][cellOutVertex];
            int remainingCapacity = usedFlow;
            addEdge(cellInVertex, cellOutVertex, remainingCapacity, capacityMatrixPhase2, adjacencyListPhase2);
        }
    }

    for (int i = 0; i < numRows; i++)
    {
        int rowInVertex = rowSuperSourceVertex;
        int rowOutVertex = 2 * numRows * numCols + i;
        int usedFlow = flowMatrixPhase1[rowInVertex][rowOutVertex];
        addEdge(rowInVertex, rowOutVertex, usedFlow, capacityMatrixPhase2, adjacencyListPhase2);
    }

    for (int j = 0; j < numCols; j++)
    {
        int colInVertex = 2 * numRows * numCols + numRows + j;
        int colOutVertex = columnSuperSinkVertex;
        int usedFlow = flowMatrixPhase1[colInVertex][colOutVertex];
        addEdge(colInVertex, colOutVertex, usedFlow, capacityMatrixPhase2, adjacencyListPhase2);
    }

    for (int i = 0; i < numRows; i++)
    {
        int rowOutVertex = 2 * numRows * numCols + i;
        for (int j = 0; j < numCols; j++)
        {
            int cellInVertex = 2 * i * numCols + 2 * j;
            addEdge(rowOutVertex, cellInVertex, INT_MAX, capacityMatrixPhase2, adjacencyListPhase2);
        }
    }

    for (int j = 0; j < numCols; j++)
    {
        int colInVertex = 2 * numRows * numCols + numRows + j;
        for (int i = 0; i < numRows; i++)
        {
            int cellOutVertex = 2 * i * numCols + 2 * j + 1;
            addEdge(cellOutVertex, colInVertex, INT_MAX, capacityMatrixPhase2, adjacencyListPhase2);
        }
    }
    addEdge(columnSuperSinkVertex, rowSuperSourceVertex, INT_MAX, capacityMatrixPhase2, adjacencyListPhase2);

    vector<vector<int>> flowMatrixPhase2;
    int maxFlowPhase2 = edmondsKarp(totalVertices, rowSuperSourceVertex, columnSuperSinkVertex, capacityMatrixPhase2, adjacencyListPhase2, flowMatrixPhase2);
    vector<vector<int>> capacityMatrixphase3(totalVertices, vector<int>(totalVertices, 0));
    vector<vector<int>> adjacencyListphase3(totalVertices);

    for (int i = 0; i < numRows; i++)
    {
        for (int j = 0; j < numCols; j++)
        {
            int cellInVertex = 2 * i * numCols + 2 * j;
            int cellOutVertex = cellInVertex + 1;
            int usedFlow = flowMatrixPhase1[cellInVertex][cellOutVertex] - flowMatrixPhase2[cellInVertex][cellOutVertex];
            int remainingCapacity = cellUpperBounds[i][j] - (cellLowerBounds[i][j] + usedFlow);
            addEdge(cellInVertex, cellOutVertex, remainingCapacity, capacityMatrixphase3, adjacencyListphase3);
        }
    }

    for (int i = 0; i < numRows; i++)
    {
        int rowInVertex = rowSuperSourceVertex;
        int rowOutVertex = 2 * numRows * numCols + i;
        int usedFlow = flowMatrixPhase1[rowInVertex][rowOutVertex] - flowMatrixPhase2[rowInVertex][rowOutVertex];
        addEdge(rowInVertex, rowOutVertex, rowUpperBounds[i] - (rowLowerBounds[i] + usedFlow), capacityMatrixphase3, adjacencyListphase3);
    }

    for (int j = 0; j < numCols; j++)
    {
        int colInVertex = 2 * numRows * numCols + numRows + j;
        int colOutVertex = columnSuperSinkVertex;
        int usedFlow = flowMatrixPhase1[colInVertex][colOutVertex] - flowMatrixPhase2[colInVertex][colOutVertex];
        addEdge(colInVertex, colOutVertex, colUpperBounds[j] - (colLowerBounds[j] + usedFlow), capacityMatrixphase3, adjacencyListphase3);
    }

    for (int i = 0; i < numRows; i++)
    {
        int rowOutVertex = 2 * numRows * numCols + i;
        for (int j = 0; j < numCols; j++)
        {
            int cellInVertex = 2 * i * numCols + 2 * j;
            addEdge(rowOutVertex, cellInVertex, INT_MAX, capacityMatrixphase3, adjacencyListphase3);
        }
    }

    for (int j = 0; j < numCols; j++)
    {
        int colInVertex = 2 * numRows * numCols + numRows + j;
        for (int i = 0; i < numRows; i++)
        {
            int cellOutVertex = 2 * i * numCols + 2 * j + 1;
            addEdge(cellOutVertex, colInVertex, INT_MAX, capacityMatrixphase3, adjacencyListphase3);
        }
    }
    addEdge(columnSuperSinkVertex, rowSuperSourceVertex, INT_MAX, capacityMatrixphase3, adjacencyListphase3);
    vector<vector<int>> flowMatrixphase3;
    int maxFlowphase3 = edmondsKarp(totalVertices, rowSuperSourceVertex, columnSuperSinkVertex, capacityMatrixphase3, adjacencyListphase3, flowMatrixphase3);
    cout << 1 << "\n";
    cout << -maxFlowPhase2 + preSatisfiedFlow + maxFlowphase3 << endl;
    cout << -maxFlowPhase2 + preSatisfiedFlow << "\n";

    return 0;
}