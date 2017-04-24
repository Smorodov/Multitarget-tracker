/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   topsort.h 
//
//==========================================================================
// $Id: topsort.h,v 1.8 2000/09/11 07:36:43 raitner Exp $

#ifndef GTL_TOPSORT
#define GTL_TOPSORT

#include <GTL/GTL.h>
#include <GTL/dfs.h>

__GTL_BEGIN_NAMESPACE

/**
 * @short Topological sorting.
 *
 * Assigns to each node <code>n</code> a number <code>top_num</code> such
 * that for every edge <code>(u,v)</code> <code>top_num[u]</code> &lt;
 * <code>top_num[v]</code>, if possible, i.e. iff the directed graph is
 * acyclic.  
 * 
 * <p>
 * Similar to the testing of biconnectivity, which extends DFS to calculate 
 * low-numbers, the topsort-algorithm extends DFS to calculate the new
 * numbering (and thus to test whether such a numbering is possible).
 *
 * <p>
 * In order to traverse all the nodes in the order of its top-numbers, a 
 * new iterator, <code>topsort_iterator</code> is provided.
 */

class GTL_EXTERN topsort : public dfs 
{
public:
    /**
     * default constructor; enables scanning of the whole_graph.
     *
     * @see dfs#dfs
     */
    topsort () : dfs () {whole_graph = true; acyclic = true;}

    /**
     * Number in topological order.
     * 
     * @param <code>n</code> node.
     * @return number in topological order.
     */
    int top_num (const node& n) const 
	{ return top_numbers[n]; }

    /**
     * Tests if graph was acyclic.
     * 
     * @return true iff graph was acyclic.
     */
    bool is_acyclic () const
	{ return acyclic; }

    /**
     * @internal
     */
	typedef nodes_t::const_iterator topsort_iterator;

    /**
     * Iterate through nodes in topsort-order. 
     * 
     * @return start-iterator.
     */
    topsort_iterator top_order_begin() const
	{ return top_order.begin(); }

    /**
     * Iterate through nodes in topsort-order. 
     * 
     * @return end-iterator.
     */
    topsort_iterator top_order_end() const
	{ return top_order.end(); }

    /**
     * Preconditions:
     * <ul>
     * <li> <code>G</code> is directed.
     * <li> DFS may be applied 
     * </ul>
     *
     * @param <code>G</code> graph.
     * @return <code>algorithm::GTL_OK</code> if topsort may be applied to 
     * <code>G</code>. 
     * @see dfs#check
     */
    virtual int check (graph& G);

    /**
     * Reset
     * @see dfs#reset
     */
    virtual void reset ();

    /**
     * @internal
     */
    virtual void init_handler (graph& G);

    /**
     * @internal 
     */
    virtual void leave_handler (graph&, node&, node&);

    /**
     * @internal 
     */
    virtual void old_adj_node_handler (graph&, edge&, node&);

protected:
    /**
     * @internal 
     */
    int act_top_num;
    /**
     * @internal 
     */
    node_map<int> top_numbers;
    /**
     * @internal 
     */
	nodes_t top_order;
    /**
     * @internal 
     */
    bool acyclic;
};

__GTL_END_NAMESPACE

#endif // GTL_TOPSORT

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
