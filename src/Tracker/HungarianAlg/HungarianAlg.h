#include <vector>
#include <iostream>
#include <limits>
#include <time.h>
#include "defines.h"
// http://community.topcoder.com/tc?module=Static&d1=tutorials&d2=hungarianAlgorithm

///
/// \brief The AssignmentProblemSolver class
///
class AssignmentProblemSolver
{
public:
    enum TMethod
    {
        optimal,
        many_forbidden_assignments,
        without_forbidden_assignments
    };

    AssignmentProblemSolver() = default;
    ~AssignmentProblemSolver() = default;
    track_t Solve(const distMatrix_t& distMatrixIn, size_t nOfRows, size_t nOfColumns, assignments_t& assignment, TMethod Method = optimal);

private:
	// Computes the optimal assignment (minimum overall costs) using Munkres algorithm.
	void assignmentoptimal(assignments_t& assignment, track_t& cost, const distMatrix_t& distMatrixIn, size_t nOfRows, size_t nOfColumns);
	void buildassignmentvector(assignments_t& assignment, bool *starMatrix, size_t nOfRows, size_t nOfColumns);
	void computeassignmentcost(const assignments_t& assignment, track_t& cost, const distMatrix_t& distMatrixIn, size_t nOfRows);
    void step2a(assignments_t& assignment, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, size_t nOfRows, size_t nOfColumns, size_t minDim);
    void step2b(assignments_t& assignment, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, size_t nOfRows, size_t nOfColumns, size_t minDim);
    void step3_5(assignments_t& assignment, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, size_t nOfRows, size_t nOfColumns, size_t minDim);
    void step4(assignments_t& assignment, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, size_t nOfRows, size_t nOfColumns, size_t minDim, size_t row, size_t col);

	// Computes a suboptimal solution. Good for cases with many forbidden assignments.
	void assignmentsuboptimal1(assignments_t& assignment, track_t& cost, const distMatrix_t& distMatrixIn, size_t nOfRows, size_t nOfColumns);
	// Computes a suboptimal solution. Good for cases with many forbidden assignments.
	void assignmentsuboptimal2(assignments_t& assignment, track_t& cost, const distMatrix_t& distMatrixIn, size_t nOfRows, size_t nOfColumns);

    std::vector<track_t> m_distMatrix;

    static constexpr bool HUNGARIAN_LOGS = false;
};
