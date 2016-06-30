#include <vector>
#include <iostream>
#include <limits>
#include <time.h>
#include "defines.h"
// http://community.topcoder.com/tc?module=Static&d1=tutorials&d2=hungarianAlgorithm

typedef std::vector<int> assignments_t;

class AssignmentProblemSolver
{
private:
	// --------------------------------------------------------------------------
	// Computes the optimal assignment (minimum overall costs) using Munkres algorithm.
	// --------------------------------------------------------------------------
	void assignmentoptimal(assignments_t& assignment, track_t *cost, track_t *distMatrix, int nOfRows, int nOfColumns);
	void buildassignmentvector(assignments_t& assignment, bool *starMatrix, int nOfRows, int nOfColumns);
	void computeassignmentcost(const assignments_t& assignment, track_t *cost, track_t *distMatrix, int nOfRows);
	void step2a(assignments_t& assignment, track_t *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim);
	void step2b(assignments_t& assignment, track_t *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim);
	void step3(assignments_t& assignment, track_t *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim);
	void step4(assignments_t& assignment, track_t *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim, int row, int col);
	void step5(assignments_t& assignment, track_t *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim);
	// --------------------------------------------------------------------------
	// Computes a suboptimal solution. Good for cases with many forbidden assignments.
	// --------------------------------------------------------------------------
	void assignmentsuboptimal1(assignments_t& assignment, track_t *cost, track_t *distMatrixIn, int nOfRows, int nOfColumns);
	// --------------------------------------------------------------------------
	// Computes a suboptimal solution. Good for cases with many forbidden assignments.
	// --------------------------------------------------------------------------
	void assignmentsuboptimal2(assignments_t& assignment, track_t *cost, track_t *distMatrixIn, int nOfRows, int nOfColumns);

public:
	enum TMethod
	{
		optimal,
		many_forbidden_assignments,
		without_forbidden_assignments
	};

	AssignmentProblemSolver();
	~AssignmentProblemSolver();
	track_t Solve(std::vector<std::vector<track_t>>& distMatrix, assignments_t& assignment, TMethod Method = optimal);
};
