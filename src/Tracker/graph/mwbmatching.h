// $Id: mwbmatching.h,v 1.3 2007/10/28 08:47:21 rdmp1c Exp $

#ifndef MWBMATCHING_H
#define MWBMATCHING_H

#include <GTL/algorithm.h>
#include <map>
#include "fheap.h"

class GTL_EXTERN mwbmatching : public algorithm
{
public:
	mwbmatching ();
	virtual ~mwbmatching();
	
    /**
     * Sets weight of every edge for maximum weight bipartite matching calculation.
     *
     * @param <code>edge_weight</code> weight of every edge.
     */
	void set_vars(const edge_map<int>& edge_weight);
	
    /**
     * Finds a maximum weight bipartite matching of G. 
     *
     * @param <code>G</code> graph.
     * @return <code>algorithm::GTL_OK</code> on success,
     * <code>algorithm::GTL_ERROR</code> otherwise.
     * @see algorithm#run
     */
    int run (graph& G);


    /**
     * Checks whether the preconditions for maximum weight bipartite matching are satisfied.
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
    virtual void reset () {};
	
	/**
	 * Returns the value of the maximum weight bipartite matching for the graph G.
	 *
	 * @return maximum weight bipartite matching value
	 *
	 */
	int get_mwbm() const { return mwbm; };
	
	/**
	 * Returns the maximum weight bipartite matching for the graph G as a list of
	 * edges.
	 *
	 * @return list of edges in maximum weight bipartite matching 
	 *
	 */
	edges_t get_match() { return result; };
	
protected:
    /**
     * @internal
     */
	long mwbm;
	
    /**
     * @internal
     */
    bool set_vars_executed;
	
    /**
     * @internal
     */
    edge_map<int> edge_weight;
    
	edges_t result;
    
	node_map<long> pot;
	node_map<bool> free;
	node_map<long> dist;
	node_map<long> pred;
	std::map <int, node, std::less<int> > node_from_id;
	std::map <int, edge, std::less<int> > edge_from_id;
	
    fheap_t *pq;
	
	int augment(graph& G, node a);
	inline void augment_path_to (graph &G, node v);



};


/**
 * Wrapper around the maximum weight bipartite matching algorithm to simplify
 * it's use. Note that the algorithm expects a directed graph where nodes in one
 * partition are sources and nodes in the other are targets. It uses this to determine
 * how to partition the nodes.
 *
 */
edges_t MAX_WEIGHT_BIPARTITE_MATCHING(graph &G, edge_map<int> weights);




#endif

