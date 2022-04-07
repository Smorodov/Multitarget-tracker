#include "HungarianAlg.h"
#include <limits>

// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
track_t AssignmentProblemSolver::Solve(const distMatrix_t& distMatrixIn,
                                       size_t nOfRows,
                                       size_t nOfColumns,
                                       std::vector<int>& assignment,
                                       TMethod Method)
{
    assignment.resize(nOfRows, -1);

    track_t cost = 0;

    switch (Method)
    {
    case optimal:
        assignmentoptimal(assignment, cost, distMatrixIn, nOfRows, nOfColumns);
        break;

    case many_forbidden_assignments:
        assignmentsuboptimal1(assignment, cost, distMatrixIn, nOfRows, nOfColumns);
        break;

    case without_forbidden_assignments:
        assignmentsuboptimal2(assignment, cost, distMatrixIn, nOfRows, nOfColumns);
        break;
    }

    return cost;
}
// --------------------------------------------------------------------------
// Computes the optimal assignment (minimum overall costs) using Munkres algorithm.
// --------------------------------------------------------------------------
void AssignmentProblemSolver::assignmentoptimal(assignments_t& assignment, track_t& cost, const distMatrix_t& distMatrixIn, size_t nOfRows, size_t nOfColumns)
{
    if constexpr (HUNGARIAN_LOGS)
            std::cout << "assignmentoptimal: Generate distance cv::Matrix and check cv::Matrix elements positiveness, assignment = " << assignment.size() << ", cost = " << cost << ", distMatrixIn = " << distMatrixIn.size() << ", nOfRows = " << nOfRows << ", nOfColumns = " << nOfColumns << std::endl;

    if constexpr (HUNGARIAN_LOGS)
            std::cout << "assignmentoptimal: Total elements number" << std::endl;
    const size_t nOfElements = nOfRows * nOfColumns;
    if constexpr (HUNGARIAN_LOGS)
            std::cout << "assignmentoptimal: Memory allocation" << std::endl;
    m_distMatrix.assign(std::begin(distMatrixIn), std::end(distMatrixIn));
    const track_t* distMatrixEnd = m_distMatrix.data() + nOfElements;

    if constexpr (HUNGARIAN_LOGS)
            std::cout << "assignmentoptimal: Memory allocation" << std::endl;
    bool* coveredColumns = (bool*)calloc(nOfColumns, sizeof(bool));
    bool* coveredRows = (bool*)calloc(nOfRows, sizeof(bool));
    bool* starMatrix = (bool*)calloc(nOfElements, sizeof(bool));
    bool* primeMatrix = (bool*)calloc(nOfElements, sizeof(bool));
    bool* newStarMatrix = (bool*)calloc(nOfElements, sizeof(bool)); // used in step4

    if constexpr (HUNGARIAN_LOGS)
            std::cout << "assignmentoptimal: preliminary steps" << std::endl;
    if (nOfRows <= nOfColumns)
    {
        for (size_t row = 0; row < nOfRows; ++row)
        {
            if constexpr (HUNGARIAN_LOGS)
                    std::cout << "assignmentoptimal: find the smallest element in the row" << std::endl;
            track_t* distMatrixTemp = m_distMatrix.data() + row;
            track_t  minValue = *distMatrixTemp;
            distMatrixTemp += nOfRows;
            while (distMatrixTemp < distMatrixEnd)
            {
                track_t value = *distMatrixTemp;
                if (value < minValue)
                    minValue = value;

                distMatrixTemp += nOfRows;
            }
            if constexpr (HUNGARIAN_LOGS)
                    std::cout << "assignmentoptimal: subtract the smallest element from each element of the row" << std::endl;
            distMatrixTemp = m_distMatrix.data() + row;
            while (distMatrixTemp < distMatrixEnd)
            {
                *distMatrixTemp -= minValue;
                distMatrixTemp += nOfRows;
            }
        }
        if constexpr (HUNGARIAN_LOGS)
                std::cout << "assignmentoptimal: Steps 1 and 2a" << std::endl;
        for (size_t row = 0; row < nOfRows; ++row)
        {
            for (size_t col = 0; col < nOfColumns; ++col)
            {
                if (m_distMatrix[row + nOfRows*col] == 0)
                {
                    if (!coveredColumns[col])
                    {
                        starMatrix[row + nOfRows * col] = true;
                        coveredColumns[col] = true;
                        break;
                    }
                }
            }
        }
    }
    else // if(nOfRows > nOfColumns)
    {
        for (size_t col = 0; col < nOfColumns; ++col)
        {
            if constexpr (HUNGARIAN_LOGS)
                    std::cout << "assignmentoptimal: find the smallest element in the column" << std::endl;
            track_t* distMatrixTemp = m_distMatrix.data() + nOfRows*col;
            track_t* columnEnd = distMatrixTemp + nOfRows;
            track_t  minValue = *distMatrixTemp++;
            while (distMatrixTemp < columnEnd)
            {
                track_t value = *distMatrixTemp++;
                if (value < minValue)
                    minValue = value;
            }
            if constexpr (HUNGARIAN_LOGS)
                    std::cout << "assignmentoptimal: subtract the smallest element from each element of the column" << std::endl;
            distMatrixTemp = m_distMatrix.data() + nOfRows*col;
            while (distMatrixTemp < columnEnd)
            {
                *distMatrixTemp++ -= minValue;
            }
        }
        if constexpr (HUNGARIAN_LOGS)
                std::cout << "assignmentoptimal: Steps 1 and 2a" << std::endl;
        for (size_t col = 0; col < nOfColumns; ++col)
        {
            for (size_t row = 0; row < nOfRows; ++row)
            {
                if (m_distMatrix[row + nOfRows*col] == 0)
                {
                    if (!coveredRows[row])
                    {
                        starMatrix[row + nOfRows*col] = true;
                        coveredColumns[col] = true;
                        coveredRows[row] = true;
                        break;
                    }
                }
            }
        }

        for (size_t row = 0; row < nOfRows; ++row)
        {
            coveredRows[row] = false;
        }
    }
    if constexpr (HUNGARIAN_LOGS)
            std::cout << "assignmentoptimal: move to step 2b" << std::endl;
    step2b(assignment, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, (nOfRows <= nOfColumns) ? nOfRows : nOfColumns);
    if constexpr (HUNGARIAN_LOGS)
            std::cout << "assignmentoptimal: compute cost and remove invalid assignments" << std::endl;
    computeassignmentcost(assignment, cost, distMatrixIn, nOfRows);
    if constexpr (HUNGARIAN_LOGS)
            std::cout << "assignmentoptimal: free allocated memory" << std::endl;
    free(coveredColumns);
    free(coveredRows);
    free(starMatrix);
    free(primeMatrix);
    free(newStarMatrix);
    return;
}
// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::buildassignmentvector(assignments_t& assignment, bool *starMatrix, size_t nOfRows, size_t nOfColumns)
{
    for (size_t row = 0; row < nOfRows; ++row)
    {
        for (size_t col = 0; col < nOfColumns; ++col)
        {
            if (starMatrix[row + nOfRows * col])
            {
                assignment[row] = static_cast<int>(col);
                break;
            }
        }
    }
}
// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::computeassignmentcost(const assignments_t& assignment, track_t& cost, const distMatrix_t& distMatrixIn, size_t nOfRows)
{
    for (size_t row = 0; row < nOfRows; ++row)
    {
        const int col = assignment[row];
        if (col >= 0)
            cost += distMatrixIn[row + nOfRows * col];
    }
}

// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::step2a(assignments_t& assignment, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, size_t nOfRows, size_t nOfColumns, size_t minDim)
{
    bool *starMatrixTemp, *columnEnd;
    if constexpr (HUNGARIAN_LOGS)
            std::cout << "step2a: cover every column containing a starred zero" << std::endl;
    for (size_t col = 0; col < nOfColumns; ++col)
    {
        starMatrixTemp = starMatrix + nOfRows * col;
        columnEnd = starMatrixTemp + nOfRows;
        while (starMatrixTemp < columnEnd)
        {
            if (*starMatrixTemp++)
            {
                coveredColumns[col] = true;
                break;
            }
        }
    }
    if constexpr (HUNGARIAN_LOGS)
            std::cout << "step2a: move to step 3" << std::endl;
    step2b(assignment, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::step2b(assignments_t& assignment, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, size_t nOfRows, size_t nOfColumns, size_t minDim)
{
    if constexpr (HUNGARIAN_LOGS)
            std::cout << "step2b: count covered columns" << std::endl;
    size_t nOfCoveredColumns = 0;
    for (size_t col = 0; col < nOfColumns; ++col)
    {
        if (coveredColumns[col])
            nOfCoveredColumns++;
    }
    if (nOfCoveredColumns == minDim) // algorithm finished
        buildassignmentvector(assignment, starMatrix, nOfRows, nOfColumns);
    else                             // move to step 3
        step3_5(assignment, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::step3_5(assignments_t& assignment, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, size_t nOfRows, size_t nOfColumns, size_t minDim)
{
    for (;;)
    {
        if constexpr (HUNGARIAN_LOGS)
                std::cout << "step3_5: step 3" << std::endl;
        bool zerosFound = true;
        while (zerosFound)
        {
            zerosFound = false;
            for (size_t col = 0; col < nOfColumns; ++col)
            {
                if (!coveredColumns[col])
                {
                    for (size_t row = 0; row < nOfRows; ++row)
                    {
                        if ((!coveredRows[row]) && (m_distMatrix[row + nOfRows*col] == 0))
                        {
                            if constexpr (HUNGARIAN_LOGS)
                                    std::cout << "step3_5: prime zero" << std::endl;
                            primeMatrix[row + nOfRows*col] = true;
                            if constexpr (HUNGARIAN_LOGS)
                                    std::cout << "step3_5: find starred zero in current row" << std::endl;
                            size_t starCol = 0;
                            for (; starCol < nOfColumns; ++starCol)
                            {
                                if (starMatrix[row + nOfRows * starCol])
                                    break;
                            }
                            if constexpr (HUNGARIAN_LOGS)
                                    std::cout << "step3_5: starCol = " << starCol << ", nOfColumns = " << nOfColumns << std::endl;
                            if (starCol == nOfColumns) // no starred zero found
                            {
                                if constexpr (HUNGARIAN_LOGS)
                                        std::cout << "step3_5: move to step 4" << std::endl;
                                step4(assignment, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim, row, col);
                                return;
                            }
                            else
                            {
                                coveredRows[row] = true;
                                coveredColumns[starCol] = false;
                                zerosFound = true;
                                break;
                            }
                        }
                    }
                }
            }
        }
        if constexpr (HUNGARIAN_LOGS)
                std::cout << "step3_5: step 5" << std::endl;
        track_t h = std::numeric_limits<track_t>::max();
        for (size_t row = 0; row < nOfRows; ++row)
        {
            if (!coveredRows[row])
            {
                for (size_t col = 0; col < nOfColumns; ++col)
                {
                    if (!coveredColumns[col])
                    {
                        const track_t value = m_distMatrix[row + nOfRows*col];
                        if (value < h)
                            h = value;
                    }
                }
            }
        }
        if constexpr (HUNGARIAN_LOGS)
                std::cout << "step3_5: add h to each covered row, h = " << h << std::endl;
        for (size_t row = 0; row < nOfRows; ++row)
        {
            if (coveredRows[row])
            {
                for (size_t col = 0; col < nOfColumns; ++col)
                {
                    m_distMatrix[row + nOfRows*col] += h;
                }
            }
        }
        if constexpr (HUNGARIAN_LOGS)
                std::cout << "step3_5: subtract h from each uncovered column" << std::endl;
        for (size_t col = 0; col < nOfColumns; ++col)
        {
            if (!coveredColumns[col])
            {
                for (size_t row = 0; row < nOfRows; ++row)
                {
                    m_distMatrix[row + nOfRows*col] -= h;
                }
            }
        }
    }
}

// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::step4(assignments_t& assignment, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, size_t nOfRows, size_t nOfColumns, size_t minDim, size_t row, size_t col)
{
    const size_t nOfElements = nOfRows * nOfColumns;
    if constexpr (HUNGARIAN_LOGS)
            std::cout << "step4: generate temporary copy of starMatrix" << std::endl;
    for (size_t n = 0; n < nOfElements; ++n)
    {
        newStarMatrix[n] = starMatrix[n];
    }
    if constexpr (HUNGARIAN_LOGS)
            std::cout << "step4: star current zero" << std::endl;
    newStarMatrix[row + nOfRows*col] = true;
    if constexpr (HUNGARIAN_LOGS)
            std::cout << "step4: find starred zero in current column" << std::endl;
    size_t starCol = col;
    size_t starRow = 0;
    for (; starRow < nOfRows; ++starRow)
    {
        if (starMatrix[starRow + nOfRows * starCol])
            break;
    }
    while (starRow < nOfRows)
    {
        // unstar the starred zero
        newStarMatrix[starRow + nOfRows*starCol] = false;
        // find primed zero in current row
        size_t primeRow = starRow;
        size_t primeCol = 0;
        for (; primeCol < nOfColumns; ++primeCol)
        {
            if (primeMatrix[primeRow + nOfRows * primeCol])
                break;
        }
        // star the primed zero
        newStarMatrix[primeRow + nOfRows*primeCol] = true;
        // find starred zero in current column
        starCol = primeCol;
        for (starRow = 0; starRow < nOfRows; ++starRow)
        {
            if (starMatrix[starRow + nOfRows * starCol])
                break;
        }
    }
    // use temporary copy as new starMatrix
    // delete all primes, uncover all rows
    for (size_t n = 0; n < nOfElements; ++n)
    {
        primeMatrix[n] = false;
        starMatrix[n] = newStarMatrix[n];
    }
    for (size_t n = 0; n < nOfRows; ++n)
    {
        coveredRows[n] = false;
    }
    if constexpr (HUNGARIAN_LOGS)
            std::cout << "move to step 2a" << std::endl;
    step2a(assignment, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

// --------------------------------------------------------------------------
// Computes a suboptimal solution. Good for cases without forbidden assignments.
// --------------------------------------------------------------------------
void AssignmentProblemSolver::assignmentsuboptimal2(assignments_t& assignment, track_t& cost, const distMatrix_t& distMatrixIn, size_t nOfRows, size_t nOfColumns)
{
    if constexpr (HUNGARIAN_LOGS)
            std::cout << "make working copy of distance Matrix" << std::endl;
    m_distMatrix.assign(std::begin(distMatrixIn), std::end(distMatrixIn));

    if constexpr (HUNGARIAN_LOGS)
            std::cout << "recursively search for the minimum element and do the assignment" << std::endl;
    for (;;)
    {
        // find minimum distance observation-to-track pair
        track_t minValue = std::numeric_limits<track_t>::max();
        size_t tmpRow = 0;
        size_t tmpCol = 0;
        for (size_t row = 0; row < nOfRows; ++row)
        {
            for (size_t col = 0; col < nOfColumns; ++col)
            {
                const track_t value = m_distMatrix[row + nOfRows*col];
                if (value != std::numeric_limits<track_t>::max() && (value < minValue))
                {
                    minValue = value;
                    tmpRow = row;
                    tmpCol = col;
                }
            }
        }

        if (minValue != std::numeric_limits<track_t>::max())
        {
            assignment[tmpRow] = static_cast<int>(tmpCol);
            cost += minValue;
            for (size_t n = 0; n < nOfRows; ++n)
            {
                m_distMatrix[n + nOfRows*tmpCol] = std::numeric_limits<track_t>::max();
            }
            for (size_t n = 0; n < nOfColumns; ++n)
            {
                m_distMatrix[tmpRow + nOfRows*n] = std::numeric_limits<track_t>::max();
            }
        }
        else
        {
            break;
        }
    }
}
// --------------------------------------------------------------------------
// Computes a suboptimal solution. Good for cases with many forbidden assignments.
// --------------------------------------------------------------------------
void AssignmentProblemSolver::assignmentsuboptimal1(assignments_t& assignment, track_t& cost, const distMatrix_t& distMatrixIn, size_t nOfRows, size_t nOfColumns)
{
    if constexpr (HUNGARIAN_LOGS)
            std::cout << "assignmentsuboptimal1: make working copy of distance Matrix" << std::endl;
    m_distMatrix.assign(std::begin(distMatrixIn), std::end(distMatrixIn));

    if constexpr (HUNGARIAN_LOGS)
            std::cout << "assignmentsuboptimal1: allocate memory" << std::endl;
    int* nOfValidObservations = (int *)calloc(nOfRows, sizeof(int));
    int* nOfValidTracks = (int *)calloc(nOfColumns, sizeof(int));

    if constexpr (HUNGARIAN_LOGS)
            std::cout << "assignmentsuboptimal1: compute number of validations" << std::endl;
    bool infiniteValueFound = false;
    bool finiteValueFound = false;
    for (size_t row = 0; row < nOfRows; ++row)
    {
        for (size_t col = 0; col < nOfColumns; ++col)
        {
            if (m_distMatrix[row + nOfRows*col] != std::numeric_limits<track_t>::max())
            {
                nOfValidTracks[col] += 1;
                nOfValidObservations[row] += 1;
                finiteValueFound = true;
            }
            else
            {
                infiniteValueFound = true;
            }
        }
    }

    if (infiniteValueFound)
    {
        if (!finiteValueFound)
        {
            if constexpr (HUNGARIAN_LOGS)
                    std::cout << "assignmentsuboptimal1: free allocated memory" << std::endl;
            free(nOfValidObservations);
            free(nOfValidTracks);
            return;
        }
        bool repeatSteps = true;

        while (repeatSteps)
        {
            repeatSteps = false;

            if constexpr (HUNGARIAN_LOGS)
                    std::cout << "assignmentsuboptimal1: step 1: reject assignments of multiply validated tracks to singly validated observation" << std::endl;
            for (size_t col = 0; col < nOfColumns; ++col)
            {
                bool singleValidationFound = false;
                for (size_t row = 0; row < nOfRows; ++row)
                {
                    if (m_distMatrix[row + nOfRows * col] != std::numeric_limits<track_t>::max() && (nOfValidObservations[row] == 1))
                    {
                        singleValidationFound = true;
                        break;
                    }
                }
                if (singleValidationFound)
                {
                    for (size_t nestedRow = 0; nestedRow < nOfRows; ++nestedRow)
                        if ((nOfValidObservations[nestedRow] > 1) && m_distMatrix[nestedRow + nOfRows * col] != std::numeric_limits<track_t>::max())
                        {
                            m_distMatrix[nestedRow + nOfRows * col] = std::numeric_limits<track_t>::max();
                            nOfValidObservations[nestedRow] -= 1;
                            nOfValidTracks[col] -= 1;
                            repeatSteps = true;
                        }
                }
            }

            if constexpr (HUNGARIAN_LOGS)
                    std::cout << "assignmentsuboptimal1: step 2: reject assignments of multiply validated observations to singly validated tracks" << std::endl;
            if (nOfColumns > 1)
            {
                for (size_t row = 0; row < nOfRows; ++row)
                {
                    bool singleValidationFound = false;
                    for (size_t col = 0; col < nOfColumns; ++col)
                    {
                        if (m_distMatrix[row + nOfRows*col] != std::numeric_limits<track_t>::max() && (nOfValidTracks[col] == 1))
                        {
                            singleValidationFound = true;
                            break;
                        }
                    }

                    if (singleValidationFound)
                    {
                        for (size_t col = 0; col < nOfColumns; ++col)
                        {
                            if ((nOfValidTracks[col] > 1) && m_distMatrix[row + nOfRows*col] != std::numeric_limits<track_t>::max())
                            {
                                m_distMatrix[row + nOfRows*col] = std::numeric_limits<track_t>::max();
                                nOfValidObservations[row] -= 1;
                                nOfValidTracks[col] -= 1;
                                repeatSteps = true;
                            }
                        }
                    }
                }
            }
        } // while(repeatSteps)

        if constexpr (HUNGARIAN_LOGS)
                std::cout << "assignmentsuboptimal1: for each multiply validated track that validates only with singly validated observations, choose the observation with minimum distance" << std::endl;
        for (size_t row = 0; row < nOfRows; ++row)
        {
            if (nOfValidObservations[row] > 1)
            {
                bool allSinglyValidated = true;
                track_t minValue = std::numeric_limits<track_t>::max();
                size_t tmpCol = 0;
                for (size_t col = 0; col < nOfColumns; ++col)
                {
                    const track_t value = m_distMatrix[row + nOfRows*col];
                    if (value != std::numeric_limits<track_t>::max())
                    {
                        if (nOfValidTracks[col] > 1)
                        {
                            allSinglyValidated = false;
                            break;
                        }
                        else if ((nOfValidTracks[col] == 1) && (value < minValue))
                        {
                            tmpCol = col;
                            minValue = value;
                        }
                    }
                }

                if (allSinglyValidated)
                {
                    assignment[row] = static_cast<int>(tmpCol);
                    cost += minValue;
                    for (size_t n = 0; n < nOfRows; ++n)
                    {
                        m_distMatrix[n + nOfRows*tmpCol] = std::numeric_limits<track_t>::max();
                    }
                    for (size_t n = 0; n < nOfColumns; ++n)
                    {
                        m_distMatrix[row + nOfRows*n] = std::numeric_limits<track_t>::max();
                    }
                }
            }
        }

        if constexpr (HUNGARIAN_LOGS)
                std::cout << "assignmentsuboptimal1: for each multiply validated observation that validates only with singly validated  track, choose the track with minimum distance" << std::endl;
        for (size_t col = 0; col < nOfColumns; ++col)
        {
            if (nOfValidTracks[col] > 1)
            {
                bool allSinglyValidated = true;
                track_t minValue = std::numeric_limits<track_t>::max();
                size_t tmpRow = 0;
                for (size_t row = 0; row < nOfRows; ++row)
                {
                    const track_t value = m_distMatrix[row + nOfRows*col];
                    if (value != std::numeric_limits<track_t>::max())
                    {
                        if (nOfValidObservations[row] > 1)
                        {
                            allSinglyValidated = false;
                            break;
                        }
                        else if ((nOfValidObservations[row] == 1) && (value < minValue))
                        {
                            tmpRow = row;
                            minValue = value;
                        }
                    }
                }

                if (allSinglyValidated)
                {
                    assignment[tmpRow] = static_cast<int>(col);
                    cost += minValue;
                    for (size_t n = 0; n < nOfRows; ++n)
                    {
                        m_distMatrix[n + nOfRows*col] = std::numeric_limits<track_t>::max();
                    }
                    for (size_t n = 0; n < nOfColumns; ++n)
                    {
                        m_distMatrix[tmpRow + nOfRows*n] = std::numeric_limits<track_t>::max();
                    }
                }
            }
        }
    } // if(infiniteValueFound)


    if constexpr (HUNGARIAN_LOGS)
            std::cout << "assignmentsuboptimal1: now, recursively search for the minimum element and do the assignment" << std::endl;
    for (;;)
    {
        // find minimum distance observation-to-track pair
        track_t minValue = std::numeric_limits<track_t>::max();
        size_t tmpRow = 0;
        size_t tmpCol = 0;
        for (size_t row = 0; row < nOfRows; ++row)
        {
            for (size_t col = 0; col < nOfColumns; ++col)
            {
                const track_t value = m_distMatrix[row + nOfRows*col];
                if (value != std::numeric_limits<track_t>::max() && (value < minValue))
                {
                    minValue = value;
                    tmpRow = row;
                    tmpCol = col;
                }
            }
        }

        if (minValue != std::numeric_limits<track_t>::max())
        {
            assignment[tmpRow] = static_cast<int>(tmpCol);
            cost += minValue;
            for (size_t n = 0; n < nOfRows; ++n)
            {
                m_distMatrix[n + nOfRows*tmpCol] = std::numeric_limits<track_t>::max();
            }
            for (size_t n = 0; n < nOfColumns; ++n)
            {
                m_distMatrix[tmpRow + nOfRows*n] = std::numeric_limits<track_t>::max();
            }
        }
        else
        {
            break;
        }
    }

    if constexpr (HUNGARIAN_LOGS)
            std::cout << "assignmentsuboptimal1: free allocated memory" << std::endl;
    free(nOfValidObservations);
    free(nOfValidTracks);
}
