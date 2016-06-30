#include "HungarianAlg.h"
#include <limits>

AssignmentProblemSolver::AssignmentProblemSolver()
{
}

AssignmentProblemSolver::~AssignmentProblemSolver()
{
}

track_t AssignmentProblemSolver::Solve(std::vector<std::vector<track_t>>& distMatrix, std::vector<int>& assignment, TMethod Method)
{
	size_t N = distMatrix.size(); // number of columns (tracks)
	size_t M = distMatrix[0].size(); // number of rows (measurements)

	assignment.resize(N, -1);

	track_t* distIn = new track_t[N*M];

	track_t cost = 0;
	// Fill cv::Matrix with random numbers
	for (size_t i = 0; i < N; i++)
	{
		for (size_t j = 0; j < M; j++)
		{
			distIn[i + N * j] = distMatrix[i][j];
		}
	}
	switch (Method)
	{
	case optimal:
		assignmentoptimal(assignment, &cost, distIn, N, M);
		break;

	case many_forbidden_assignments:
		assignmentsuboptimal1(assignment, &cost, distIn, N, M);
		break;

	case without_forbidden_assignments:
		assignmentsuboptimal2(assignment, &cost, distIn, N, M);
		break;
	}

	delete[] distIn;
	return cost;
}
// --------------------------------------------------------------------------
// Computes the optimal assignment (minimum overall costs) using Munkres algorithm.
// --------------------------------------------------------------------------
void AssignmentProblemSolver::assignmentoptimal(assignments_t& assignment, track_t *cost, track_t *distMatrixIn, int nOfRows, int nOfColumns)
{
	track_t *distMatrix;
	track_t *distMatrixTemp;
	track_t *distMatrixEnd;
	track_t *columnEnd;
	track_t  value;
	track_t  minValue;

	bool *coveredColumns;
	bool *coveredRows;
	bool *starMatrix;
	bool *newStarMatrix;
	bool *primeMatrix;

	int nOfElements;
	int minDim;
	int row;
	int col;

	// Init
	*cost = 0;

	// Generate distance cv::Matrix 
	// and check cv::Matrix elements positiveness :)

	// Total elements number
	nOfElements = nOfRows * nOfColumns;
	// Memory allocation
	distMatrix = (track_t *)malloc(nOfElements * sizeof(track_t));
	// Pointer to last element
	distMatrixEnd = distMatrix + nOfElements;

	// 
	for (row = 0; row < nOfElements; row++)
	{
		value = distMatrixIn[row];
		assert(value >= 0);
		distMatrix[row] = value;
	}

	// Memory allocation
	coveredColumns = (bool *)calloc(nOfColumns, sizeof(bool));
	coveredRows = (bool *)calloc(nOfRows, sizeof(bool));
	starMatrix = (bool *)calloc(nOfElements, sizeof(bool));
	primeMatrix = (bool *)calloc(nOfElements, sizeof(bool));
	newStarMatrix = (bool *)calloc(nOfElements, sizeof(bool)); /* used in step4 */

	/* preliminary steps */
	if (nOfRows <= nOfColumns)
	{
		minDim = nOfRows;
		for (row = 0; row < nOfRows; row++)
		{
			/* find the smallest element in the row */
			distMatrixTemp = distMatrix + row;
			minValue = *distMatrixTemp;
			distMatrixTemp += nOfRows;
			while (distMatrixTemp < distMatrixEnd)
			{
				value = *distMatrixTemp;
				if (value < minValue)
				{
					minValue = value;
				}
				distMatrixTemp += nOfRows;
			}
			/* subtract the smallest element from each element of the row */
			distMatrixTemp = distMatrix + row;
			while (distMatrixTemp < distMatrixEnd)
			{
				*distMatrixTemp -= minValue;
				distMatrixTemp += nOfRows;
			}
		}
		/* Steps 1 and 2a */
		for (row = 0; row < nOfRows; row++)
		{
			for (col = 0; col < nOfColumns; col++)
			{
				if (distMatrix[row + nOfRows*col] == 0)
				{
					if (!coveredColumns[col])
					{
						starMatrix[row + nOfRows*col] = true;
						coveredColumns[col] = true;
						break;
					}
				}
			}
		}
	}
	else /* if(nOfRows > nOfColumns) */
	{
		minDim = nOfColumns;
		for (col = 0; col < nOfColumns; col++)
		{
			/* find the smallest element in the column */
			distMatrixTemp = distMatrix + nOfRows*col;
			columnEnd = distMatrixTemp + nOfRows;
			minValue = *distMatrixTemp++;
			while (distMatrixTemp < columnEnd)
			{
				value = *distMatrixTemp++;
				if (value < minValue)
				{
					minValue = value;
				}
			}
			/* subtract the smallest element from each element of the column */
			distMatrixTemp = distMatrix + nOfRows*col;
			while (distMatrixTemp < columnEnd)
			{
				*distMatrixTemp++ -= minValue;
			}
		}
		/* Steps 1 and 2a */
		for (col = 0; col < nOfColumns; col++)
		{
			for (row = 0; row < nOfRows; row++)
			{
				if (distMatrix[row + nOfRows*col] == 0)
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

		for (row = 0; row < nOfRows; row++)
		{
			coveredRows[row] = false;
		}
	}
	/* move to step 2b */
	step2b(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
	/* compute cost and remove invalid assignments */
	computeassignmentcost(assignment, cost, distMatrixIn, nOfRows);
	/* free allocated memory */
	free(distMatrix);
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
void AssignmentProblemSolver::buildassignmentvector(assignments_t& assignment, bool *starMatrix, int nOfRows, int nOfColumns)
{
	for (int row = 0; row < nOfRows; row++)
	{
		for (int col = 0; col < nOfColumns; col++)
		{
			if (starMatrix[row + nOfRows * col])
			{
				assignment[row] = col;
				break;
			}
		}
	}
}
// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::computeassignmentcost(const assignments_t& assignment, track_t *cost, track_t *distMatrix, int nOfRows)
{
	for (int row = 0; row < nOfRows; row++)
	{
		const int col = assignment[row];
		if (col >= 0)
		{
			*cost += distMatrix[row + nOfRows * col];
		}
	}
}

// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::step2a(assignments_t& assignment, track_t *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim)
{
	bool *starMatrixTemp, *columnEnd;
	/* cover every column containing a starred zero */
	for (int col = 0; col < nOfColumns; col++)
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
	/* move to step 3 */
	step2b(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::step2b(assignments_t& assignment, track_t *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim)
{
	/* count covered columns */
	int nOfCoveredColumns = 0;
	for (int col = 0; col < nOfColumns; col++)
	{
		if (coveredColumns[col])
		{
			nOfCoveredColumns++;
		}
	}
	if (nOfCoveredColumns == minDim)
	{
		/* algorithm finished */
		buildassignmentvector(assignment, starMatrix, nOfRows, nOfColumns);
	}
	else
	{
		/* move to step 3 */
		step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
	}
}

// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::step3(assignments_t& assignment, track_t *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim)
{
	bool zerosFound = true;
	while (zerosFound)
	{
		zerosFound = false;
		for (int col = 0; col < nOfColumns; col++)
		{
			if (!coveredColumns[col])
			{
				for (int row = 0; row < nOfRows; row++)
				{
					if ((!coveredRows[row]) && (distMatrix[row + nOfRows*col] == 0))
					{
						/* prime zero */
						primeMatrix[row + nOfRows*col] = true;
						/* find starred zero in current row */
						int starCol = 0;
						for (; starCol < nOfColumns; starCol++)
						{
							if (starMatrix[row + nOfRows * starCol])
							{
								break;
							}
						}
						if (starCol == nOfColumns) /* no starred zero found */
						{
							/* move to step 4 */
							step4(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim, row, col);
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
	/* move to step 5 */
	step5(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::step4(assignments_t& assignment, track_t *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim, int row, int col)
{
	const int nOfElements = nOfRows * nOfColumns;
	/* generate temporary copy of starMatrix */
	for (int n = 0; n < nOfElements; n++)
	{
		newStarMatrix[n] = starMatrix[n];
	}
	/* star current zero */
	newStarMatrix[row + nOfRows*col] = true;
	/* find starred zero in current column */
	int starCol = col;
	int starRow = 0;
	for (; starRow < nOfRows; starRow++)
	{
		if (starMatrix[starRow + nOfRows * starCol])
		{
			break;
		}
	}
	while (starRow < nOfRows)
	{
		/* unstar the starred zero */
		newStarMatrix[starRow + nOfRows*starCol] = false;
		/* find primed zero in current row */
		int primeRow = starRow;
		int primeCol = 0;
		for (; primeCol < nOfColumns; primeCol++)
		{
			if (primeMatrix[primeRow + nOfRows * primeCol])
			{
				break;
			}
		}
		/* star the primed zero */
		newStarMatrix[primeRow + nOfRows*primeCol] = true;
		/* find starred zero in current column */
		starCol = primeCol;
		for (starRow = 0; starRow < nOfRows; starRow++)
		{
			if (starMatrix[starRow + nOfRows * starCol])
			{
				break;
			}
		}
	}
	/* use temporary copy as new starMatrix */
	/* delete all primes, uncover all rows */
	for (int n = 0; n < nOfElements; n++)
	{
		primeMatrix[n] = false;
		starMatrix[n] = newStarMatrix[n];
	}
	for (int n = 0; n < nOfRows; n++)
	{
		coveredRows[n] = false;
	}
	/* move to step 2a */
	step2a(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::step5(assignments_t& assignment, track_t *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim)
{
	/* find smallest uncovered element h */
	float h = std::numeric_limits<track_t>::max();
	for (int row = 0; row < nOfRows; row++)
	{
		if (!coveredRows[row])
		{
			for (int col = 0; col < nOfColumns; col++)
			{
				if (!coveredColumns[col])
				{
					const float value = distMatrix[row + nOfRows*col];
					if (value < h)
					{
						h = value;
					}
				}
			}
		}
	}
	/* add h to each covered row */
	for (int row = 0; row < nOfRows; row++)
	{
		if (coveredRows[row])
		{
			for (int col = 0; col < nOfColumns; col++)
			{
				distMatrix[row + nOfRows*col] += h;
			}
		}
	}
	/* subtract h from each uncovered column */
	for (int col = 0; col < nOfColumns; col++)
	{
		if (!coveredColumns[col])
		{
			for (int row = 0; row < nOfRows; row++)
			{
				distMatrix[row + nOfRows*col] -= h;
			}
		}
	}
	/* move to step 3 */
	step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}


// --------------------------------------------------------------------------
// Computes a suboptimal solution. Good for cases without forbidden assignments.
// --------------------------------------------------------------------------
void AssignmentProblemSolver::assignmentsuboptimal2(assignments_t& assignment, track_t *cost, track_t *distMatrixIn, int nOfRows, int nOfColumns)
{
	/* make working copy of distance Matrix */
	const int nOfElements = nOfRows * nOfColumns;
	float* distMatrix = (float*)malloc(nOfElements * sizeof(float));
	for (int n = 0; n < nOfElements; n++)
	{
		distMatrix[n] = distMatrixIn[n];
	}

	/* initialization */
	*cost = 0;

	/* recursively search for the minimum element and do the assignment */
	for (;;)
	{
		/* find minimum distance observation-to-track pair */
		float minValue = std::numeric_limits<track_t>::max();
		int tmpRow = 0;
		int tmpCol = 0;
		for (int row = 0; row < nOfRows; row++)
		{
			for (int col = 0; col < nOfColumns; col++)
			{
				const float value = distMatrix[row + nOfRows*col];
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
			assignment[tmpRow] = tmpCol;
			*cost += minValue;
			for (int n = 0; n < nOfRows; n++)
			{
				distMatrix[n + nOfRows*tmpCol] = std::numeric_limits<track_t>::max();
			}
			for (int n = 0; n < nOfColumns; n++)
			{
				distMatrix[tmpRow + nOfRows*n] = std::numeric_limits<track_t>::max();
			}
		}
		else
		{
			break;
		}
	}

	free(distMatrix);
}
// --------------------------------------------------------------------------
// Computes a suboptimal solution. Good for cases with many forbidden assignments.
// --------------------------------------------------------------------------
void AssignmentProblemSolver::assignmentsuboptimal1(assignments_t& assignment, track_t *cost, track_t *distMatrixIn, int nOfRows, int nOfColumns)
{
	/* make working copy of distance Matrix */
	const int nOfElements = nOfRows * nOfColumns;
	float* distMatrix = (float *)malloc(nOfElements * sizeof(float));
	for (int n = 0; n < nOfElements; n++)
	{
		distMatrix[n] = distMatrixIn[n];
	}
	/* initialization */
	*cost = 0;

	/* allocate memory */
	int* nOfValidObservations = (int *)calloc(nOfRows, sizeof(int));
	int* nOfValidTracks = (int *)calloc(nOfColumns, sizeof(int));

	/* compute number of validations */
	bool infiniteValueFound = false;
	bool finiteValueFound = false;
	for (int row = 0; row < nOfRows; row++)
	{
		for (int col = 0; col < nOfColumns; col++)
		{
			if (distMatrix[row + nOfRows*col] != std::numeric_limits<track_t>::max())
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
			return;
		}
		bool repeatSteps = true;

		while (repeatSteps)
		{
			repeatSteps = false;

			/* step 1: reject assignments of multiply validated tracks to singly validated observations		 */
			for (int col = 0; col < nOfColumns; col++)
			{
				bool singleValidationFound = false;
				for (int row = 0; row < nOfRows; row++)
				{
					if (distMatrix[row + nOfRows*col] != std::numeric_limits<track_t>::max() && (nOfValidObservations[row] == 1))
					{
						singleValidationFound = true;
						break;
					}

					if (singleValidationFound)
					{
						for (int row = 0; row < nOfRows; row++)
							if ((nOfValidObservations[row] > 1) && distMatrix[row + nOfRows*col] != FLT_MAX)
							{
								distMatrix[row + nOfRows*col] = FLT_MAX;
								nOfValidObservations[row] -= 1;
								nOfValidTracks[col] -= 1;
								repeatSteps = true;
							}
					}
				}
			}

			/* step 2: reject assignments of multiply validated observations to singly validated tracks */
			if (nOfColumns > 1)
			{
				for (int row = 0; row < nOfRows; row++)
				{
					bool singleValidationFound = false;
					for (int col = 0; col < nOfColumns; col++)
					{
						if (distMatrix[row + nOfRows*col] != std::numeric_limits<track_t>::max() && (nOfValidTracks[col] == 1))
						{
							singleValidationFound = true;
							break;
						}
					}

					if (singleValidationFound)
					{
						for (int col = 0; col < nOfColumns; col++)
						{
							if ((nOfValidTracks[col] > 1) && distMatrix[row + nOfRows*col] != FLT_MAX)
							{
								distMatrix[row + nOfRows*col] = std::numeric_limits<track_t>::max();
								nOfValidObservations[row] -= 1;
								nOfValidTracks[col] -= 1;
								repeatSteps = true;
							}
						}
					}
				}
			}
		} /* while(repeatSteps) */

		/* for each multiply validated track that validates only with singly validated  */
		/* observations, choose the observation with minimum distance */
		for (int row = 0; row < nOfRows; row++)
		{
			if (nOfValidObservations[row] > 1)
			{
				bool allSinglyValidated = true;
				float minValue = std::numeric_limits<track_t>::max();
				int tmpCol = 0;
				for (int col = 0; col < nOfColumns; col++)
				{
					const float value = distMatrix[row + nOfRows*col];
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
					assignment[row] = tmpCol;
					*cost += minValue;
					for (int n = 0; n < nOfRows; n++)
					{
						distMatrix[n + nOfRows*tmpCol] = std::numeric_limits<track_t>::max();
					}
					for (int n = 0; n < nOfColumns; n++)
					{
						distMatrix[row + nOfRows*n] = std::numeric_limits<track_t>::max();
					}
				}
			}
		}

		// for each multiply validated observation that validates only with singly validated  track, choose the track with minimum distance
		for (int col = 0; col < nOfColumns; col++)
		{
			if (nOfValidTracks[col] > 1)
			{
				bool allSinglyValidated = true;
				float minValue = std::numeric_limits<track_t>::max();
				int tmpRow = 0;
				for (int row = 0; row < nOfRows; row++)
				{
					const float value = distMatrix[row + nOfRows*col];
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
					assignment[tmpRow] = col;
					*cost += minValue;
					for (int n = 0; n < nOfRows; n++)
					{
						distMatrix[n + nOfRows*col] = std::numeric_limits<track_t>::max();
					}
					for (int n = 0; n < nOfColumns; n++)
					{
						distMatrix[tmpRow + nOfRows*n] = std::numeric_limits<track_t>::max();
					}
				}
			}
		}
	} /* if(infiniteValueFound) */


	/* now, recursively search for the minimum element and do the assignment */
	for (;;)
	{
		/* find minimum distance observation-to-track pair */
		float minValue = std::numeric_limits<track_t>::max();
		int tmpRow = 0;
		int tmpCol = 0;
		for (int row = 0; row < nOfRows; row++)
		{
			for (int col = 0; col < nOfColumns; col++)
			{
				const float value = distMatrix[row + nOfRows*col];
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
			assignment[tmpRow] = tmpCol;
			*cost += minValue;
			for (int n = 0; n < nOfRows; n++)
			{
				distMatrix[n + nOfRows*tmpCol] = std::numeric_limits<track_t>::max();
			}
			for (int n = 0; n < nOfColumns; n++)
			{
				distMatrix[tmpRow + nOfRows*n] = std::numeric_limits<track_t>::max();
			}
		}
		else
		{
			break;
		}
	}

	/* free allocated memory */
	free(nOfValidObservations);
	free(nOfValidTracks);
}
