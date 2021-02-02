// $Id: mwbmatching.h,v 1.3 2007/10/28 08:47:21 rdmp1c Exp $

#ifndef MWBMATCHING_H
#define MWBMATCHING_H

#include <GTL/algorithm.h>
#include <map>
#include "fheap.h"

class GTL_EXTERN mwbmatching final : public GTL::algorithm
{
public:
	mwbmatching ();
	virtual ~mwbmatching() = default;
	
    /**
     * Sets weight of every edge for maximum weight bipartite matching calculation.
     *
     * @param <code>edge_weight</code> weight of every edge.
     */
	void set_vars(const GTL::edge_map<int>& edge_weight);
	
    /**
     * Finds a maximum weight bipartite matching of G. 
     *
     * @param <code>G</code> graph.
     * @return <code>algorithm::GTL_OK</code> on success,
     * <code>algorithm::GTL_ERROR</code> otherwise.
     * @see algorithm#run
     */
    int run (GTL::graph& G);


    /**
     * Checks whether the preconditions for maximum weight bipartite matching are satisfied.
     *
     * @param <code>G</code> graph.
     * @return <code>algorithm::GTL_OK</code> on success,
     * <code>algorithm::GTL_ERROR</code> otherwise.
     * @see algorithm#check
     */
    virtual int check (GTL::graph& G);
	
    /**
     * Reset. 
     *
     * @see algorithm#reset
     */
    virtual void reset () {}
	
	/**
	 * Returns the value of the maximum weight bipartite matching for the graph G.
	 *
	 * @return maximum weight bipartite matching value
	 *
	 */
	int get_mwbm() const { return m_mwbm; }
	
	/**
	 * Returns the maximum weight bipartite matching for the graph G as a list of
	 * edges.
	 *
	 * @return list of edges in maximum weight bipartite matching 
	 *
	 */
	GTL::edges_t get_match() { return m_result; }
	
protected:
    /**
     * @internal
     */
	long m_mwbm = 0;
	
    /**
     * @internal
     */
    bool m_set_vars_executed = false;
	
    /**
     * @internal
     */
	GTL::edge_map<int> m_edge_weight;
    
	GTL::edges_t m_result;
    
	GTL::node_map<long> m_pot;
	GTL::node_map<bool> m_free;
	GTL::node_map<long> m_dist;
	GTL::node_map<long> m_pred;
	std::map <int, GTL::node, std::less<int> > m_node_from_id;
	std::map <int, GTL::edge, std::less<int> > m_edge_from_id;
	
    fheap_t* m_pq = nullptr;
	
	int augment(GTL::graph& G, GTL::node a);
	inline void augment_path_to (GTL::graph &G, GTL::node v);
};


/**
 * Wrapper around the maximum weight bipartite matching algorithm to simplify
 * it's use. Note that the algorithm expects a directed graph where nodes in one
 * partition are sources and nodes in the other are targets. It uses this to determine
 * how to partition the nodes.
 *
 */
GTL::edges_t MAX_WEIGHT_BIPARTITE_MATCHING(GTL::graph &G, GTL::edge_map<int> weights);




#endif

