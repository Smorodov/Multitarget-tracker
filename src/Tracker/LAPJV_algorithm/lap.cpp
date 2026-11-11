/************************************************************************
*
*  lap.cpp
   version 1.0 - 4 September 1996
   author: Roy Jonker @ MagicLogic Optimization Inc.
   e-mail: roy_jonker@magiclogic.com

   Code for Linear Assignment Problem, according to

   "A Shortest Augmenting Path Algorithm for Dense and Sparse Linear
    Assignment Problems," Computing 38, 325-340, 1987

   by

   R. Jonker and A. Volgenant, University of Amsterdam.

*
   CHANGED 2016-05-13 by Yong Yang(yongyanglink@gmail.com) in column reduction part according to
   matlab version of LAPJV algorithm(Copyright (c) 2010, Yi Cao All rights reserved)--
   https://www.mathworks.com/matlabcentral/fileexchange/26836-lapjv-jonker-volgenant-algorithm-for-linear-assignment-problem-v3-0:
*
*************************************************************************/

#include <algorithm>
#include <cassert>
#include <limits>
#include "lap.h"


/*This function is the jv shortest augmenting path algorithm to solve the assignment problem.
  It expects a square cost matrix (dimRows == dimCols); the public lap() wrapper below
  reduces a rectangular problem to this form by adding zero-cost dummy rows/columns.*/
static cost lapSquare(const std::vector<std::vector<cost>>& assigncost,
                      std::vector<col>& rowsol,
                      std::vector<row>& colsol,
                      std::vector<cost>& u,
                      std::vector<cost>& v)

// input:
// assigncost - cost matrix

// output:
// rowsol     - column assigned to row in solution
// colsol     - row assigned to column in solution
// u          - dual variables, row reduction numbers
// v          - dual variables, column reduction numbers

{
    if (assigncost.empty() || assigncost[0].empty())
    {
        assert(false);
        return 0;
    }

    const int dimRows = static_cast<int>(assigncost.size());
    const int dimCols = static_cast<int>(assigncost[0].size());

    // lapSquare() implements the classical LAPJV, which requires a square
    // matrix; the lap() wrapper below guarantees this invariant.
    assert(dimRows == dimCols);
    if (dimRows != dimCols ||
        static_cast<int>(rowsol.size()) < dimRows ||
        static_cast<int>(colsol.size()) < dimCols ||
        static_cast<int>(u.size()) < dimRows ||
        static_cast<int>(v.size()) < dimCols)
    {
        return 0;
    }

    bool unassignedfound = false;
    row numfree = 0;
    col j2 = 0, endofpath = 0, last = 0;
    cost min = std::numeric_limits<cost>::max();
    
    std::vector<row> freeunassigned(dimRows);     // list of unassigned rows.
    std::vector<col> collist(dimCols);            // list of columns to be scanned in various ways.
    std::vector<col> matches(dimRows, 0);         // counts how many times a row could be assigned.
    std::vector<cost> d(dimCols);                 // 'cost-distance' in augmenting path calculation.
    std::vector<row> pred(dimCols);               // row-predecessor of column in augmenting/alternating path.

    // COLUMN REDUCTION
    for (col j = dimCols; j--;)  // reverse order gives better results.
    {
        // find minimum cost over rows.
        min = assigncost[0][j];
        row imin = 0;
        for (row i = 1; i < dimRows; i++)
            if (assigncost[i][j] < min) {
                min = assigncost[i][j];
                imin = i;
            }
        v[j] = min;
        if (++matches[imin] == 1) {
            // init assignment if minimum row assigned for first time.
            rowsol[imin] = j;
            colsol[j] = imin;
        } else if (v[j] < v[rowsol[imin]]) {
            int j1 = rowsol[imin];
            rowsol[imin] = j;
            colsol[j] = imin;
            colsol[j1] = -1;
        } else
            colsol[j] = -1;  // row already assigned, column not assigned.
    }

    // REDUCTION TRANSFER
    for (row i = 0; i < dimRows; i++)
        if (matches[i] == 0)  // fill list of unassigned 'free' rows.
            freeunassigned[numfree++] = i;
        else if (matches[i] == 1)  // transfer reduction from rows that are assigned once.
        {
            col j1 = rowsol[i];
            min = std::numeric_limits<cost>::max();
            for (col j = 0; j < dimCols; j++)
                if (j != j1)
                    if (assigncost[i][j] - v[j] < min) min = assigncost[i][j] - v[j];
            v[j1] = v[j1] - min;
        }

    //   AUGMENTING ROW REDUCTION
    int loopcnt = 0;  // do-loop to be done twice.
    do {
        loopcnt++;

        //     scan all free rows.
        //     in some cases, a free row may be replaced with another one to be scanned next.
        row k = 0;
        row prvnumfree = numfree;
        numfree = 0;  // start list of rows still free after augmenting row reduction.
        while (k < prvnumfree) {
            row i = freeunassigned[k];
            k++;

            //       find minimum and second minimum reduced cost over columns.
            cost umin = assigncost[i][0] - v[0];
            col j1 = 0;
            cost usubmin = std::numeric_limits<cost>::max();
            for (col j = 1; j < dimCols; j++) {
                cost h = assigncost[i][j] - v[j];
                if (h < usubmin)
                    if (h >= umin) {
                        usubmin = h;
                        j2 = j;
                    } else {
                        usubmin = umin;
                        umin = h;
                        j2 = j1;
                        j1 = j;
                    }
            }

            row i0 = colsol[j1];
            if (umin < usubmin)
                //         change the reduction of the minimum column to increase the minimum
                //         reduced cost in the row to the subminimum.
                v[j1] = v[j1] - (usubmin - umin);
            else              // minimum and subminimum equal.
                if (i0 > -1)  // minimum column j1 is assigned.
            {
                //           swap columns j1 and j2, as j2 may be unassigned.
                j1 = j2;
                i0 = colsol[j2];
            }

            //       (re-)assign i to j1, possibly de-assigning an i0.
            rowsol[i] = j1;
            colsol[j1] = i;

            if (i0 > -1)  // minimum column j1 assigned earlier.
                if (umin < usubmin)
                    //           put in current k, and go back to that k.
                    //           continue augmenting path i - j1 with i0.
                    freeunassigned[--k] = i0;
                else
                    //           no further augmenting reduction possible.
                    //           store i0 in list of free rows for next phase.
                    freeunassigned[numfree++] = i0;
        }
    } while (loopcnt < 2);  // repeat once.

    // AUGMENT SOLUTION for each free row.
    for (row f = 0; f < numfree; f++) {
        row freerow = freeunassigned[f];  // start row of augmenting path.

        // Dijkstra shortest path algorithm.
        // runs until unassigned column added to shortest path tree.
        for (col j = dimCols; j--;) {
            d[j] = assigncost[freerow][j] - v[j];
            pred[j] = freerow;
            collist[j] = j;  // init column list.
        }

        col low = 0;  // columns in 0..low-1 are ready, now none.
        col up = 0;   // columns in low..up-1 are to be scanned for current minimum, now none.
                  // columns in up..dim-1 are to be considered later to find new minimum,
                  // at this stage the list simply contains all columns
        unassignedfound = false;
        do {
            if (up == low)  // no more columns to be scanned for current minimum.
            {
                last = low - 1;

                // Defensive check: with a finite cost matrix an augmenting
                // path always exists, but a non-finite cost can break the
                // search and make `up` reach dimCols.
                if (up >= dimCols)
                    break;

                // scan columns for up..dim-1 to find all indices for which new minimum occurs.
                // store these indices between low..up-1 (increasing up).
                min = d[collist[up++]];
                for (row k = up; k < dimCols; k++) {
                    col j = collist[k];
                    cost h = d[j];
                    if (h <= min) {
                        if (h < min)  // new minimum.
                        {
                            up = low;  // restart list at index low.
                            min = h;
                        }
                        // new index with same minimum, put on undex up, and extend list.
                        collist[k] = collist[up];
                        collist[up++] = j;
                    }
                }
                // check if any of the minimum columns happens to be unassigned.
                // if so, we have an augmenting path right away.
                for (row k = low; k < up; k++)
                    if (colsol[collist[k]] < 0) {
                        endofpath = collist[k];
                        unassignedfound = true;
                        break;
                    }
            }

            if (!unassignedfound) {
                // update 'distances' between freerow and all unscanned columns, via next scanned
                // column.
                col j1 = collist[low];
                low++;
                row i = colsol[j1];
                cost h = assigncost[i][j1] - v[j1] - min;

                for (row k = up; k < dimCols; k++) {
                    col j = collist[k];
                    cost v2 = assigncost[i][j] - v[j] - h;
                    if (v2 < d[j]) {
                        pred[j] = i;
                        if (v2 == min)  // new column found at same minimum value
                            if (colsol[j] < 0) {
                                // if unassigned, shortest augmenting path is complete.
                                endofpath = j;
                                unassignedfound = true;
                                break;
                            }
                            // else add to list to be scanned right away.
                            else {
                                collist[k] = collist[up];
                                collist[up++] = j;
                            }
                        d[j] = v2;
                    }
                }
            }
        } while (!unassignedfound);

        if (!unassignedfound)
        {
            // No augmenting path was found (only possible with non-finite
            // costs). Return a consistent "no assignment" result instead of
            // reading past the end of collist or leaving stale assignments.
            std::fill(rowsol.begin(), rowsol.begin() + dimRows, -1);
            std::fill(colsol.begin(), colsol.begin() + dimCols, -1);
            return 0;
        }

        // update column prices.
        for (row k = last + 1; k--;) {
            col j1 = collist[k];
            v[j1] = v[j1] + d[j1] - min;
        }

        // reset row and column assignments along the alternating path.
        row i = 0; 
        do {
            i = pred[endofpath];
            colsol[endofpath] = i;
            col j1 = endofpath;
            endofpath = rowsol[i];
            rowsol[i] = j1;
        } while (i != freerow);
    }

    // calculate optimal cost.
    cost lapcost = 0;
    //  for (i = 0; i < dim; i++)
    for (row i = dimRows; i--;) {
        col j = rowsol[i];
        if (j < 0)
        {
            u[i] = 0;
            continue;
        }
        u[i] = assigncost[i][j] - v[j];
        lapcost = lapcost + assigncost[i][j];
    }

    return lapcost;
}

///
/// \brief lap
/// Solve the rectangular linear assignment problem (rows may be less, equal
/// or greater than columns).
///
/// lapSquare() implements the classical square LAPJV. A rectangular problem is
/// reduced to it by padding the cost matrix with zero-cost dummy rows and/or
/// columns: a dummy row/column never increases the total cost, so the optimum
/// restricted to the real rows/columns is the optimum of the original problem.
/// Real rows assigned to a dummy column are reported as unassigned (-1).
///
cost lap(const std::vector<std::vector<cost>>& assigncost,
         std::vector<col>& rowsol,
         std::vector<row>& colsol,
         std::vector<cost>& u,
         std::vector<cost>& v)
{
    if (assigncost.empty() || assigncost[0].empty())
    {
        assert(false);
        return 0;
    }

    const int rows = static_cast<int>(assigncost.size());
    const int cols = static_cast<int>(assigncost[0].size());

    if (static_cast<int>(rowsol.size()) < rows ||
        static_cast<int>(colsol.size()) < cols ||
        static_cast<int>(u.size()) < rows ||
        static_cast<int>(v.size()) < cols)
    {
        assert(false);
        return 0;
    }

    if (rows == cols)
        return lapSquare(assigncost, rowsol, colsol, u, v);

    const int dim = std::max(rows, cols);
    std::vector<std::vector<cost>> square(dim, std::vector<cost>(dim, 0.0));
    for (int i = 0; i < rows; ++i)
        std::copy(assigncost[i].begin(), assigncost[i].end(), square[i].begin());

    std::vector<col> rowsolDim(dim, -1);
    std::vector<row> colsolDim(dim, -1);
    std::vector<cost> uDim(dim, 0.0), vDim(dim, 0.0);

    const cost total = lapSquare(square, rowsolDim, colsolDim, uDim, vDim);

    // Copy the real part of the solution back; drop assignments to dummies.
    for (int i = 0; i < rows; ++i)
        rowsol[i] = (rowsolDim[i] >= 0 && rowsolDim[i] < cols) ? rowsolDim[i] : -1;
    for (int j = 0; j < cols; ++j)
        colsol[j] = (colsolDim[j] >= 0 && colsolDim[j] < rows) ? colsolDim[j] : -1;
    for (int i = 0; i < rows; ++i)
        u[i] = uDim[i];
    for (int j = 0; j < cols; ++j)
        v[j] = vDim[j];

    return total;
}
