/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   maxflow_ff.h
//
//==========================================================================
// $Id: maxflow_ff.h,v 1.5 2003/01/31 08:15:05 chris Exp $

#ifndef GTL_MAXFLOW_FF_H
#define GTL_MAXFLOW_FF_H

#include <GTL/GTL.h>
#include <GTL/graph.h>
#include <GTL/node_map.h>
#include <GTL/edge_map.h>
#include <GTL/algorithm.h>

#include <queue>

__GTL_BEGIN_NAMESPACE

/**
 * @short Maximum flow algorithm (Edmonds-Karp).
 */
class GTL_EXTERN maxflow_ff : public algorithm
{
public:
    /**
     * Default constructor. Enables only the calculation of
     * maximum flow.
     * 
     * @see algorithm#algorithm
     */
    maxflow_ff();

    /**
     * Destructor.
     *
     * @see algorithm#~algorithm
     */
    virtual ~maxflow_ff();

    /**
     * Sets capacity of every edge for maximum flow calculation
     * where artificial start-node and end_node will be computed
     * automatically.
     *
     * @param edge_capacity capacity of every edge
     */
    void set_vars(const edge_map<double>& edge_capacity);

    /**
     * Sets capacity of every edge for maximum flow calculation
     *
     * @param edge_capacity capacity of every edge
     * @param net_source start-node
     * @param net_target end-node
     */
    void set_vars(
	const edge_map<double>& edge_capacity, 
	const node& net_source, 
	const node& net_target);

    /**
     * Checks whether following preconditions are satisfied:
     * <ul>
     * <li> @ref maxflow_ff#set_vars has been executed before.
     * <li> only edge_capacities >= 0 are applied.
     * <li> <code>G</code> is directed.
     * <li> <code>G</code> is connected.
     * <li> <code>G</code> has at least one edge and two nodes.
     * <li> if not applied, start-nodes and end-nodes exists.
     * <li> if applied, start-node is not the same node as end-node.
     * </ul>
     * 
     * @param G graph
     * @return <code>algorithm::GTL_OK</code> on success 
     * <code>algorithm::GTL_ERROR</code> otherwise
     * @see algorithm#check
     */
    virtual int check(graph& G);
	    
    /**
     * Computes maximum flow of graph <code>G</code>.
     * 
     * @param G graph
     * @return <code>algorithm::GTL_OK</code> on success 
     * <code>algorithm::GTL_ERROR</code> otherwise
     * @see algorithm#run
     */
    int run(graph& G);
		
    /**
     * Returns the maximum flow of an edge.
     *
     * @param e edge of a graph G
     * @return maximum flow value
     */
    double get_max_flow(const edge& e) const;

    /**
     * Returns the maximum flow of the whole graph G.
     *
     * @return maximum flow value
     */
    double get_max_flow() const;
	
    /**
     * Returns the remaining free capacity of an edge.
     *
     * @param e edge of a graph G
     * @return remaining capacity
     */
    double get_rem_cap(const edge& e) const;

    /**
     * Resets maximum flow algorithm, i.e. prepares the
     * algorithm to be applied to another graph. 
     *
     * @see algorithm#reset
     */
    virtual void reset();
protected:
    /**
     * @internal
     */
    enum {SP_FOUND = 2, NO_SP_FOUND = 3};

    /**
     * @internal
     */
    bool artif_source_target;

    /**
     * @internal
     */
    bool set_vars_executed;

    /**
     * @internal
     */
    double max_graph_flow;

    /**
     * @internal
     */
    node net_source;

    /**
     * @internal
     */
    node net_target;

    /**
     * @internal edges to remove from G after run
     */
	edges_t edges_not_org;

    /**
     * @internal original edge or inserted back edge
     */
    edge_map<bool> edge_org;

    /**
     * @internal
     */
    edge_map<bool> back_edge_exists;

    /**
     * @internal every edge knows its back edge
     */
    edge_map<edge> back_edge;

    /**
     * @internal
     */
    edge_map<double> edge_capacity;

    /**
     * @internal
     */
    edge_map<double> edge_max_flow;

    /**
     * @internal
     */
    void create_artif_source_target(graph& G);

    /**
     * @internal
     */
    void prepare_run(const graph& G);

    /**
     * @internal
     */
    void comp_single_flow(graph& G, node_map<edge>& last_edge);

    /**
     * @internal every node knows its predecessor then
     */
    int get_sp(const graph& G, node_map<edge>& last_edge);

    /**
     * @internal
     */
    int comp_sp(
	const graph& G, 
	std::queue<node>& next_nodes,
	node_map<bool>& visited, 
	node_map<edge>& last_edge);

    /**
     * @internal
     */
    double extra_charge(const node_map<edge>& last_edge) const;

    /**
     * @internal
     */
    void create_back_edge(graph& G, const edge& org_edge);

    /**
     * @internal
     */
    void comp_max_flow(const graph& G);

    /**
     * @internal
     */
    void restore_graph(graph& G);
};

__GTL_END_NAMESPACE

#endif // GTL_MAXFLOW_FF_H

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
