#include "ShortPathCalculator.h"

#include "LAPJV_algorithm/lap.h"

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
