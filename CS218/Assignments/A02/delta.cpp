vector<vector<int>> capacityMatrixPhase3(totalVertices, vector<int>(totalVertices, 0));
vector<vector<int>> adjacencyListPhase3(totalVertices);

for (int i = 0; i < numRows; i++)
{
    for (int j = 0; j < numCols; j++)
    {
        int cellInVertex = 2 * i * numCols + 2 * j;
        int cellOutVertex = cellInVertex + 1;
        int usedFlow = flowMatrixPhase1[cellInVertex][cellOutVertex];
        int remainingCapacity = cellUpperBounds[i][j] - (cellLowerBounds[i][j] + usedFlow);
        addEdge(cellInVertex, cellOutVertex, remainingCapacity, capacityMatrixphase3, adjacencyListphase3);
    }
}

for (int i = 0; i < numRows; i++)
{
    int rowInVertex = rowSuperSourceVertex;
    int rowOutVertex =  2 * numRows * numCols +  i;
    int usedFlow = flowMatrixPhase1[rowInVertex][rowOutVertex];
    addEdge(rowInVertex, rowOutVertex, rowUpperBounds[i] - (rowLowerBounds[i] + usedFlow), capacityMatrixphase3, adjacencyListphase3);
}

for (int j = 0; j < numCols; j++)
{
    int colInVertex = 2 * numRows * numCols + numRows + j;
    int colOutVertex = columnSuperSinkVertex;
    int usedFlow = flowMatrixPhase1[colInVertex][colOutVertex];
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
