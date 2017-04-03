// $Id: mincut.h,v 1.1.1.1 2003/11/05 15:19:13 rdmp1c Exp $

#ifndef MINCUT_H
#define MINCUT_H


#include <GTL/algorithm.h>

/**
 * @typedef node_pair
 * A pair of nodes
 */

typedef std::pair<node, node> node_pair;

class GTL_EXTERN mincut : public algorithm
{
public:
	mincut ();
	virtual ~mincut();
	
    /**
     * Sets weight of every edge for mincut calculation.
     *
     * @param <code>edge_weight</code> weight of every edge.
     */
	void set_vars(const edge_map<int>& edge_weight);
	
    /**
     * Finds a mincut of G. 
     *
     * @param <code>G</code> graph.
     * @return <code>algorithm::GTL_OK</code> on success,
     * <code>algorithm::GTL_ERROR</code> otherwise.
     * @see algorithm#run
     */
    int run (graph& G);


    /**
     * Checks whether the preconditions for mincut are satisfied.
     *
     * @param <code>G</code> graph.
     * @return <code>algorithm::GTL_OK</code> on success,
     * <code>algorithm::GTL_ERROR</code> otherwise.
     * @see algorithm#check
     */
    virtual int check (graph& G);
	
    /**
     * Reset. 
     *
     * @see algorithm#reset
     */
    virtual void reset ();
	
	/**
	 * Returns the mincut for the graph G.
	 *
	 * @return mincut value
	 *
	 */
	int get_mincut() const { return min_cut; };
	
protected:
    /**
     * @internal
     */
	int min_cut;
	
    /**
     * @internal
     */
    bool set_vars_executed;
	
    /**
     * @internal
     */
    edge_map<int> edge_weight;
	
    /**
     * @internal
     */
	std::list<node_pair> st_list;




};


#endif
