#include "ShortPathCalculator.h"

#include <GTL/GTL.h>
#include "mygraph.h"
#include "mwbmatching.h"
#include "tokenise.h"

#include "LAPJV_algorithm/lap.h"

///
/// \brief SPBipart::Solve
/// \param costMatrix
/// \param N
/// \param M
/// \param assignment
/// \param maxCost
///
void SPBipart::Solve(const distMatrix_t& costMatrix, size_t colsTracks, size_t rowsRegions, assignments_t& assignmentT2R, track_t maxCost)
{
    //std::cout << "SPBipart::Solve: colsTracks = " << colsTracks << ", rowsRegions = " << rowsRegions << std::endl;

    MyGraph G;
    G.make_directed();

    std::vector<GTL::node> nodes(colsTracks + rowsRegions);

    for (size_t i = 0; i < nodes.size(); ++i)
    {
        nodes[i] = G.new_node();
    }

	GTL::edge_map<int> weights(G, 100);
    for (size_t i = 0; i < colsTracks; ++i)
    {
        bool hasZeroEdge = false;

        for (size_t j = 0; j < rowsRegions; ++j)
        {
            track_t currCost = costMatrix[i + j * colsTracks];

			GTL::edge e = G.new_edge(nodes[i], nodes[colsTracks + j]);

            if (currCost < m_settings.m_distThres)
            {
                int weight = static_cast<int>(maxCost - currCost + 1);
                G.set_edge_weight(e, weight);
                weights[e] = weight;
            }
            else
            {
                if (!hasZeroEdge)
                {
                    G.set_edge_weight(e, 0);
                    weights[e] = 0;
                }
                hasZeroEdge = true;
            }
        }
    }

	GTL::edges_t L = MAX_WEIGHT_BIPARTITE_MATCHING(G, weights);
    for (GTL::edges_t::iterator it = L.begin(); it != L.end(); ++it)
    {
        GTL::node a = it->source();
        GTL::node b = it->target();
        assignmentT2R[b.id()] = static_cast<assignments_t::value_type>(a.id() - colsTracks);
    }
}

///
void SPLAPJV::Solve(const distMatrix_t& costMatrix, size_t colsTracks, size_t rowsRegions, assignments_t& assignmentT2R, track_t /*maxCost*/)
{
    //std::cout << "SPLAPJV::Solve: colsTracks = " << colsTracks << ", rowsRegions = " << rowsRegions << std::endl;

    if (!colsTracks || !rowsRegions)
        return;

    bool swithReg2Track = (rowsRegions > colsTracks); // For this algorithm rows <= cols

    size_t dimRows = swithReg2Track ? colsTracks : rowsRegions; // Set the dimension of matrix to 10, dim is the problem size
    size_t dimCols = swithReg2Track ? rowsRegions : colsTracks;
    std::vector<std::vector<cost>> costMat; // A matrix to store all the costs from vertex i to vertex j
    std::vector<col> rowsol(dimRows, -1);   // An array to store column indexes assigned to row in solution
    std::vector<row> colsol(dimCols, -1);   // An array to store row indexes assigned to column in solution
    std::vector<cost> u(dimRows);           // u - dual variables, row reduction numbers
    std::vector<cost> v(dimCols);           // v - dual variables, column reduction numbers

    costMat.resize(dimRows);
    for (size_t i = 0; i < dimRows; i++)
    {
        costMat[i].resize(dimCols);
        for (size_t j = 0; j < dimCols; ++j)
        {
            costMat[i][j] = swithReg2Track ? costMatrix[j * colsTracks + i] : costMatrix[i * colsTracks + j];

            //std::cout << std::fixed << std::setw(2) << std::setprecision(2) << costMat[i][j] << " ";
        }
        //std::cout << std::endl;
    }
    //std::cout << "Cost matrix created" << std::endl;
    cost totalCost = lap(costMat, rowsol, colsol, u, v);  // Use lap algorithm to calculate the minimum total cost
    //std::cout << "totalCost = " << totalCost << std::endl;

    //for (size_t i = 0; i < rowsol.size(); ++i)
    //{
    //    std::cout << "row[" << i << "]: " << rowsol[i] << ", u = " << u[i] << std::endl;
    //}
    //for (size_t i = 0; i < colsol.size(); ++i)
    //{
    //    std::cout << "col[" << i << "]: " << colsol[i] << ", u = " << v[i] << std::endl;
    //}


    if (swithReg2Track)
    {
        for (size_t i = 0; i < colsol.size(); ++i)
        {
            if (colsol[i] >= 0)
                assignmentT2R[colsol[i]] = static_cast<int>(i);
        }
    }
    else
    {
        for (size_t i = 0; i < colsol.size(); ++i)
        {
            if (colsol[i] >= 0)
                assignmentT2R[i] = colsol[i];
        }
    }
}
