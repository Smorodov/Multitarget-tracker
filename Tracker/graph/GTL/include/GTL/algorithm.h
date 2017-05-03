/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   algorithm.h 
//
//==========================================================================
// $Id: algorithm.h,v 1.14 2003/03/24 15:58:54 raitner Exp $

#ifndef GTL_ALGORITHM_H
#define GTL_ALGORITHM_H

#include <GTL/GTL.h>
#include <GTL/graph.h>

__GTL_BEGIN_NAMESPACE

/**
 * $Date: 2003/03/24 15:58:54 $
 * $Revision: 1.14 $
 *
 * @brief Abstract baseclass for all algoritm-classes.
 */
class GTL_EXTERN algorithm {
public:
    /** 
     * @var algorithm::GTL_OK 
     * Used as (positive) return value of algorithm::check and 
     * algorithm::run.
     */

    /** 
     * @var algorithm::GTL_ERROR 
     * Used as (negative) return value of algorithm::check and 
     * algorithm::run.
     */

    /**
     * @brief Return values for algorithm::check and algorithm::run
     */
    enum {
	GTL_OK = 1,
	GTL_ERROR = 0
    };

    /**
     * @brief Creates an algorithm object.
     */
    algorithm () { };
    
    /**
     * @brief Destroys the algorithm object.
     */
    virtual ~algorithm () { };    

    /**
     * @brief Applies %algorithm to %graph g. 
     * 
     * @param g %graph
     * @retval algorithm::GTL_OK on success
     * @retval algorithm::GTL_ERROR otherwise
     */
    virtual int run (graph& g) = 0;
    
    /**
     * @brief Checks whether all preconditions are satisfied.
     * 
     * @em Please @em note: It is
     * definitly required (and #run relies on it),
     * that this method was called in advance.
     * 
     * @param g %graph
     * @retval algorithm::GTL_OK if %algorithm can be applied
     * @retval algorithm::GTL_ERROR otherwise.
     */
    virtual int check (graph& g) = 0;
    
    /**
     * @brief Resets %algorithm 
     * 
     * Prepares the %algorithm to be applied to
     * another %graph. @em Please @em note: The options an
     * %algorithm may support do @em not get reset by
     * this. It is just to reset internally used datastructures.
     */
    virtual void reset () = 0;
};

__GTL_END_NAMESPACE

#endif // GTL_ALGORITHM_H

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
