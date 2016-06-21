#include "HungarianAlg.h"
#include <limits>

AssignmentProblemSolver::AssignmentProblemSolver()
{
}

AssignmentProblemSolver::~AssignmentProblemSolver()
{
}

track_t AssignmentProblemSolver::Solve(std::vector<std::vector<track_t>>& distMatrix,std::vector<int>& Assignment,TMethod Method)
{
	size_t N=distMatrix.size(); // number of columns (tracks)
	size_t M = distMatrix[0].size(); // number of rows (measurements)

	int *assignment		=new int[N];
	track_t *distIn		=new track_t[N*M];

	track_t  cost;
	// Fill cv::Matrix with random numbers
	for (size_t i = 0; i<N; i++)
	{
		for (size_t j = 0; j<M; j++)
		{
			distIn[i+N*j] = distMatrix[i][j];
		}
	}
	switch(Method)
	{
	case optimal: assignmentoptimal(assignment, &cost, distIn, N, M); break;
	
	case many_forbidden_assignments: assignmentsuboptimal1(assignment, &cost, distIn, N, M); break;
	
	case without_forbidden_assignments: assignmentsuboptimal2(assignment, &cost, distIn, N, M); break;
	}

	// form result 
	Assignment.clear();
	for (size_t x = 0; x<N; x++)
	{
		Assignment.push_back(assignment[x]);
	}

	delete[] assignment;
	delete[] distIn;
	return cost;
}
// --------------------------------------------------------------------------
// Computes the optimal assignment (minimum overall costs) using Munkres algorithm.
// --------------------------------------------------------------------------
void AssignmentProblemSolver::assignmentoptimal(int *assignment, track_t *cost, track_t *distMatrixIn, int nOfRows, int nOfColumns)
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
	bool *newstarMatrix;
	bool *primeMatrix;

	int nOfElements;
	int minDim;
	int row;
	int col;

	// Init
	*cost = 0;
	for(row=0; row<nOfRows; row++)
	{
		assignment[row] = -1;
	}

	// Generate distance cv::Matrix 
	// and check cv::Matrix elements positiveness :)

	// Total elements number
	nOfElements   = nOfRows * nOfColumns; 
	// Memory allocation
	distMatrix    = (track_t *)malloc(nOfElements * sizeof(track_t));
	// cv::Pointer to last element
	distMatrixEnd = distMatrix + nOfElements;

	// 
	for(row=0; row<nOfElements; row++)
	{
		value = distMatrixIn[row];
		if(value < 0)
		{
			std::cout << "All cv::Matrix elements have to be non-negative." << std::endl;
		}
		distMatrix[row] = value;
	}

	// Memory allocation
	coveredColumns = (bool *)calloc(nOfColumns,  sizeof(bool));
	coveredRows    = (bool *)calloc(nOfRows,     sizeof(bool));
	starMatrix     = (bool *)calloc(nOfElements, sizeof(bool));
	primeMatrix    = (bool *)calloc(nOfElements, sizeof(bool));
	newstarMatrix  = (bool *)calloc(nOfElements, sizeof(bool)); /* used in step4 */

	/* preliminary steps */
	if(nOfRows <= nOfColumns)
	{
		minDim = nOfRows;
		for(row=0; row<nOfRows; row++)
		{
			/* find the smallest element in the row */
			distMatrixTemp = distMatrix + row;
			minValue = *distMatrixTemp;
			distMatrixTemp += nOfRows;
			while(distMatrixTemp < distMatrixEnd)
			{
				value = *distMatrixTemp;
				if(value < minValue)
				{
					minValue = value;
				}
				distMatrixTemp += nOfRows;
			}
			/* subtract the smallest element from each element of the row */
			distMatrixTemp = distMatrix + row;
			while(distMatrixTemp < distMatrixEnd)
			{
				*distMatrixTemp -= minValue;
				distMatrixTemp += nOfRows;
			}
		}
		/* Steps 1 and 2a */
		for(row=0; row<nOfRows; row++)
		{
			for(col=0; col<nOfColumns; col++)
			{
				if(distMatrix[row + nOfRows*col] == 0)
				{
					if(!coveredColumns[col])
					{
						starMatrix[row + nOfRows*col] = true;
						coveredColumns[col]           = true;
						break;
					}
				}
			}
		}
	}
	else /* if(nOfRows > nOfColumns) */
	{
		minDim = nOfColumns;
		for(col=0; col<nOfColumns; col++)
		{
			/* find the smallest element in the column */
			distMatrixTemp = distMatrix     + nOfRows*col;
			columnEnd      = distMatrixTemp + nOfRows;
			minValue = *distMatrixTemp++;
			while(distMatrixTemp < columnEnd)
			{
				value = *distMatrixTemp++;
				if(value < minValue)
				{
					minValue = value;
				}
			}
			/* subtract the smallest element from each element of the column */
			distMatrixTemp = distMatrix + nOfRows*col;
			while(distMatrixTemp < columnEnd)
			{
				*distMatrixTemp++ -= minValue;
			}
		}
		/* Steps 1 and 2a */
		for(col=0; col<nOfColumns; col++)
		{
			for(row=0; row<nOfRows; row++)
			{
				if(distMatrix[row + nOfRows*col] == 0)
				{
					if(!coveredRows[row])
					{
						starMatrix[row + nOfRows*col] = true;
						coveredColumns[col]           = true;
						coveredRows[row]              = true;
						break;
					}
				}
			}
		}

		for(row=0; row<nOfRows; row++)
		{
			coveredRows[row] = false;
		}
	}
	/* move to step 2b */
	step2b(assignment, distMatrix, starMatrix, newstarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
	/* compute cost and remove invalid assignments */
	computeassignmentcost(assignment, cost, distMatrixIn, nOfRows);
	/* free allocated memory */
	free(distMatrix);
	free(coveredColumns);
	free(coveredRows);
	free(starMatrix);
	free(primeMatrix);
	free(newstarMatrix);
	return;
}
// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::buildassignmentvector(int *assignment, bool *starMatrix, int nOfRows, int nOfColumns)
{
	int row, col;
	for(row=0; row<nOfRows; row++)
	{
		for(col=0; col<nOfColumns; col++)
		{
			if(starMatrix[row + nOfRows*col])
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
void AssignmentProblemSolver::computeassignmentcost(int *assignment, track_t *cost, track_t *distMatrix, int nOfRows)
{
	int row, col;
	for(row=0; row<nOfRows; row++)
	{
		col = assignment[row];
		if(col >= 0)
		{
			*cost += distMatrix[row + nOfRows*col];
		}
	}
}

// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::step2a(int *assignment, track_t *distMatrix, bool *starMatrix, bool *newstarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim)
{
	bool *starMatrixTemp, *columnEnd;
	int col;
	/* cover every column containing a starred zero */
	for(col=0; col<nOfColumns; col++)
	{
		starMatrixTemp = starMatrix     + nOfRows*col;
		columnEnd      = starMatrixTemp + nOfRows;
		while(starMatrixTemp < columnEnd)
		{
			if(*starMatrixTemp++)
			{
				coveredColumns[col] = true;
				break;
			}
		}
	}
	/* move to step 3 */
	step2b(assignment, distMatrix, starMatrix, newstarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::step2b(int *assignment, track_t *distMatrix, bool *starMatrix, bool *newstarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim)
{
	int col, nOfCoveredColumns;
	/* count covered columns */
	nOfCoveredColumns = 0;
	for(col=0; col<nOfColumns; col++)
	{
		if(coveredColumns[col])
		{
			nOfCoveredColumns++;
		}
	}
	if(nOfCoveredColumns == minDim)
	{
		/* algorithm finished */
		buildassignmentvector(assignment, starMatrix, nOfRows, nOfColumns);
	}
	else
	{
		/* move to step 3 */
		step3(assignment, distMatrix, starMatrix, newstarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
	}
}

// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::step3(int *assignment, track_t *distMatrix, bool *starMatrix, bool *newstarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim)
{
	bool zerosFound;
	int row, col, starCol;
	zerosFound = true;
	while(zerosFound)
	{
		zerosFound = false;
		for(col=0; col<nOfColumns; col++)
		{
			if(!coveredColumns[col])
			{
				for(row=0; row<nOfRows; row++)
				{
					if((!coveredRows[row]) && (distMatrix[row + nOfRows*col] == 0))
					{
						/* prime zero */
						primeMatrix[row + nOfRows*col] = true;
						/* find starred zero in current row */
						for(starCol=0; starCol<nOfColumns; starCol++)
							if(starMatrix[row + nOfRows*starCol])
							{
								break;
							}
							if(starCol == nOfColumns) /* no starred zero found */
							{
								/* move to step 4 */
								step4(assignment, distMatrix, starMatrix, newstarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim, row, col);
								return;
							}
							else
							{
								coveredRows[row]        = true;
								coveredColumns[starCol] = false;
								zerosFound              = true;
								break;
							}
					}
				}
			}
		}
	}
	/* move to step 5 */
	step5(assignment, distMatrix, starMatrix, newstarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::step4(int *assignment, track_t *distMatrix, bool *starMatrix, bool *newstarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim, int row, int col)
{
	int n, starRow, starCol, primeRow, primeCol;
	int nOfElements = nOfRows*nOfColumns;
	/* generate temporary copy of starMatrix */
	for(n=0; n<nOfElements; n++)
	{
		newstarMatrix[n] = starMatrix[n];
	}
	/* star current zero */
	newstarMatrix[row + nOfRows*col] = true;
	/* find starred zero in current column */
	starCol = col;
	for(starRow=0; starRow<nOfRows; starRow++)
	{
		if(starMatrix[starRow + nOfRows*starCol])
		{
			break;
		}
	}
	while(starRow<nOfRows)
	{
		/* unstar the starred zero */
		newstarMatrix[starRow + nOfRows*starCol] = false;
		/* find primed zero in current row */
		primeRow = starRow;
		for(primeCol=0; primeCol<nOfColumns; primeCol++)
		{
			if(primeMatrix[primeRow + nOfRows*primeCol])
			{
				break;
			}
		}
		/* star the primed zero */
		newstarMatrix[primeRow + nOfRows*primeCol] = true;
		/* find starred zero in current column */
		starCol = primeCol;
		for(starRow=0; starRow<nOfRows; starRow++)
		{
			if(starMatrix[starRow + nOfRows*starCol])
			{
				break;
			}
		}
	}
	/* use temporary copy as new starMatrix */
	/* delete all primes, uncover all rows */
	for(n=0; n<nOfElements; n++)
	{
		primeMatrix[n] = false;
		starMatrix[n]  = newstarMatrix[n];
	}
	for(n=0; n<nOfRows; n++)
	{
		coveredRows[n] = false;
	}
	/* move to step 2a */
	step2a(assignment, distMatrix, starMatrix, newstarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::step5(int *assignment, track_t *distMatrix, bool *starMatrix, bool *newstarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim)
{
	track_t h, value;
	int row, col;
	/* find smallest uncovered element h */
	h = std::numeric_limits<track_t>::max();
	for(row=0; row<nOfRows; row++)
	{
		if(!coveredRows[row])
		{
			for(col=0; col<nOfColumns; col++)
			{
				if(!coveredColumns[col])
				{
					value = distMatrix[row + nOfRows*col];
					if(value < h)
					{
						h = value;
					}
				}
			}
		}
	}
	/* add h to each covered row */
	for(row=0; row<nOfRows; row++)
	{
		if(coveredRows[row])
		{
			for(col=0; col<nOfColumns; col++)
			{
				distMatrix[row + nOfRows*col] += h;
			}
		}
	}
	/* subtract h from each uncovered column */
	for(col=0; col<nOfColumns; col++)
	{
		if(!coveredColumns[col])
		{
			for(row=0; row<nOfRows; row++)
			{
				distMatrix[row + nOfRows*col] -= h;
			}
		}
	}
	/* move to step 3 */
	step3(assignment, distMatrix, starMatrix, newstarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}


// --------------------------------------------------------------------------
// Computes a suboptimal solution. Good for cases without forbidden assignments.
// --------------------------------------------------------------------------
void AssignmentProblemSolver::assignmentsuboptimal2(int *assignment, track_t *cost, track_t *distMatrixIn, int nOfRows, int nOfColumns)
{
	int n, row, col, tmpRow, tmpCol, nOfElements;
	track_t value, minValue, *distMatrix;


	/* make working copy of distance cv::Matrix */
	nOfElements   = nOfRows * nOfColumns;
	distMatrix    = (track_t *)malloc(nOfElements * sizeof(track_t));
	for(n=0; n<nOfElements; n++)
	{
		distMatrix[n] = distMatrixIn[n];
	}

	/* initialization */
	*cost = 0;
	for(row=0; row<nOfRows; row++)
	{
		assignment[row] = -1;
	}

	/* recursively search for the minimum element and do the assignment */
	while(true)
	{
		/* find minimum distance observation-to-track pair */
		minValue = std::numeric_limits<track_t>::max();
		for(row=0; row<nOfRows; row++)
			for(col=0; col<nOfColumns; col++)
			{
				value = distMatrix[row + nOfRows*col];
				if(value!=std::numeric_limits<track_t>::max() && (value < minValue))
				{
					minValue = value;
					tmpRow   = row;
					tmpCol   = col;
				}
			}

			if(minValue!=std::numeric_limits<track_t>::max())
			{
				assignment[tmpRow] = tmpCol;
				*cost += minValue;
				for(n=0; n<nOfRows; n++)
				{
					distMatrix[n + nOfRows*tmpCol] = std::numeric_limits<track_t>::max();
				}
				for(n=0; n<nOfColumns; n++)
				{
					distMatrix[tmpRow + nOfRows*n] = std::numeric_limits<track_t>::max();
				}
			}
			else
				break;

	} /* while(true) */

	free(distMatrix);
}
// --------------------------------------------------------------------------
// Computes a suboptimal solution. Good for cases with many forbidden assignments.
// --------------------------------------------------------------------------
void AssignmentProblemSolver::assignmentsuboptimal1(int *assignment, track_t *cost, track_t *distMatrixIn, int nOfRows, int nOfColumns)
{
	bool infiniteValueFound, finiteValueFound, repeatSteps, allSinglyValidated, singleValidationFound;
	int n, row, col, tmpRow, tmpCol, nOfElements;
	int *nOfValidObservations, *nOfValidTracks;
	track_t value, minValue, *distMatrix;


	/* make working copy of distance cv::Matrix */
	nOfElements   = nOfRows * nOfColumns;
	distMatrix    = (track_t *)malloc(nOfElements * sizeof(track_t));
	for(n=0; n<nOfElements; n++)
	{
		distMatrix[n] = distMatrixIn[n];
	}
	/* initialization */
	*cost = 0;

	for(row=0; row<nOfRows; row++)
	{
		assignment[row] = -1;
	}

	/* allocate memory */
	nOfValidObservations  = (int *)calloc(nOfRows,    sizeof(int));
	nOfValidTracks        = (int *)calloc(nOfColumns, sizeof(int));

	/* compute number of validations */
	infiniteValueFound = false;
	finiteValueFound  = false;
	for(row=0; row<nOfRows; row++)
	{
		for(col=0; col<nOfColumns; col++)
		{
			if(distMatrix[row + nOfRows*col]!=std::numeric_limits<track_t>::max())
			{
				nOfValidTracks[col]       += 1;
				nOfValidObservations[row] += 1;
				finiteValueFound = true;
			}
			else
				infiniteValueFound = true;
		}
	}

	if(infiniteValueFound)
	{
		if(!finiteValueFound)
		{
			return;
		}	
		repeatSteps = true;

		while(repeatSteps)
		{
			repeatSteps = false;

			/* step 1: reject assignments of multiply validated tracks to singly validated observations		 */
			for(col=0; col<nOfColumns; col++)
			{
				singleValidationFound = false;
				for(row=0; row<nOfRows; row++)
					if(distMatrix[row + nOfRows*col]!=std::numeric_limits<track_t>::max() && (nOfValidObservations[row] == 1))
					{
						singleValidationFound = true;
						break;
					}

					if(singleValidationFound)
					{
						for(row=0; row<nOfRows; row++)
							if((nOfValidObservations[row] > 1) && distMatrix[row + nOfRows*col]!=std::numeric_limits<track_t>::max())
							{
								distMatrix[row + nOfRows*col] = std::numeric_limits<track_t>::max();
								nOfValidObservations[row] -= 1;							
								nOfValidTracks[col]       -= 1;	
								repeatSteps = true;				
							}
					}
			}

			/* step 2: reject assignments of multiply validated observations to singly validated tracks */
			if(nOfColumns > 1)			
			{	
				for(row=0; row<nOfRows; row++)
				{
					singleValidationFound = false;
					for(col=0; col<nOfColumns; col++)
					{
						if(distMatrix[row + nOfRows*col]!=std::numeric_limits<track_t>::max() && (nOfValidTracks[col] == 1))
						{
							singleValidationFound = true;
							break;
						}
					}

					if(singleValidationFound)
					{
						for(col=0; col<nOfColumns; col++)
						{
							if((nOfValidTracks[col] > 1) && distMatrix[row + nOfRows*col]!=std::numeric_limits<track_t>::max())
							{
								distMatrix[row + nOfRows*col] = std::numeric_limits<track_t>::max();
								nOfValidObservations[row] -= 1;
								nOfValidTracks[col]       -= 1;
								repeatSteps = true;								
							}
						}
					}
				}
			}
		} /* while(repeatSteps) */

		/* for each multiply validated track that validates only with singly validated  */
		/* observations, choose the observation with minimum distance */
		for(row=0; row<nOfRows; row++)
		{
			if(nOfValidObservations[row] > 1)
			{
				allSinglyValidated = true;
				minValue = std::numeric_limits<track_t>::max();
				for(col=0; col<nOfColumns; col++)
				{
					value = distMatrix[row + nOfRows*col];
					if(value!=std::numeric_limits<track_t>::max())
					{
						if(nOfValidTracks[col] > 1)
						{
							allSinglyValidated = false;
							break;
						}
						else if((nOfValidTracks[col] == 1) && (value < minValue))
						{
							tmpCol   = col;
							minValue = value;
						}
					}
				}

				if(allSinglyValidated)
				{
					assignment[row] = tmpCol;
					*cost += minValue;
					for(n=0; n<nOfRows; n++)
					{
						distMatrix[n + nOfRows*tmpCol] = std::numeric_limits<track_t>::max();
					}
					for(n=0; n<nOfColumns; n++)
					{
						distMatrix[row + nOfRows*n] = std::numeric_limits<track_t>::max();
					}
				}
			}
		}

		/* for each multiply validated observation that validates only with singly validated  */
		/* track, choose the track with minimum distance */
		for(col=0; col<nOfColumns; col++)
		{
			if(nOfValidTracks[col] > 1)
			{
				allSinglyValidated = true;
				minValue = std::numeric_limits<track_t>::max();
				for(row=0; row<nOfRows; row++)
				{
					value = distMatrix[row + nOfRows*col];
					if(value!=std::numeric_limits<track_t>::max())
					{
						if(nOfValidObservations[row] > 1)
						{
							allSinglyValidated = false;
							break;
						}
						else if((nOfValidObservations[row] == 1) && (value < minValue))
						{
							tmpRow   = row;
							minValue = value;
						}
					}
				}

				if(allSinglyValidated)
				{
					assignment[tmpRow] = col;
					*cost += minValue;
					for(n=0; n<nOfRows; n++)
						distMatrix[n + nOfRows*col] = std::numeric_limits<track_t>::max();
					for(n=0; n<nOfColumns; n++)
						distMatrix[tmpRow + nOfRows*n] = std::numeric_limits<track_t>::max();
				}
			}
		}	
	} /* if(infiniteValueFound) */


	/* now, recursively search for the minimum element and do the assignment */
	while(true)
	{
		/* find minimum distance observation-to-track pair */
		minValue = std::numeric_limits<track_t>::max();
		for(row=0; row<nOfRows; row++)
			for(col=0; col<nOfColumns; col++)
			{
				value = distMatrix[row + nOfRows*col];
				if(value!=std::numeric_limits<track_t>::max() && (value < minValue))
				{
					minValue = value;
					tmpRow   = row;
					tmpCol   = col;
				}
			}

			if(minValue!=std::numeric_limits<track_t>::max())
			{
				assignment[tmpRow] = tmpCol;
				*cost += minValue;
				for(n=0; n<nOfRows; n++)
					distMatrix[n + nOfRows*tmpCol] = std::numeric_limits<track_t>::max();
				for(n=0; n<nOfColumns; n++)
					distMatrix[tmpRow + nOfRows*n] = std::numeric_limits<track_t>::max();			
			}
			else
				break;

	} /* while(true) */

	/* free allocated memory */
	free(nOfValidObservations);
	free(nOfValidTracks);
}
