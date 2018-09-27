/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   maxflow_sap.h
//
//==========================================================================
// $Id: maxflow_sap.h,v 1.4 2003/01/31 08:15:05 chris Exp $

#ifndef GTL_MAXFLOW_SAP_H
#define GTL_MAXFLOW_SAP_H

#include <GTL/GTL.h>
#include <GTL/graph.h>
#include <GTL/node_map.h>
#include <GTL/edge_map.h>
#include <GTL/algorithm.h>

#include <queue>

__GTL_BEGIN_NAMESPACE

/**
 * @short Maximum flow algorithm with shortest augmenting paths
 *
 * This class implements a maximum flow algorithm with shortest augmenting
 * paths due to Ahuja and Orlin.
 *
 * <p> In the case V is the set of vertices and E is the set of edges of
 * the graph, the algorithm needs O(|V| * |V| * |E|) time to proceed.
 *
 * @see maxflow_ff
 * @see maxflow_pp
 */
class GTL_EXTERN maxflow_sap : public algorithm
{
public:
    /**
     * Default constructor. Enables only the calculation of
     * maximum flow.
     * 
     * @see algorithm#algorithm
     */
    maxflow_sap();

    /**
     * Destructor.
     *
     * @see algorithm#~algorithm
     */
    virtual ~maxflow_sap();

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
    void set_vars(const edge_map<double>& edge_capacity, 
				  const node& net_source, 
				  const node& net_target);

    /**
     * Checks whether following preconditions are satisfied:
     * <ul>
     * <li> @ref maxflow_sap#set_vars has been executed before.
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
     * @param e edge of a graph @c G
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
     * @param e edge of a graph @c G
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
    enum {AP_FOUND = 2, NO_AP_FOUND = 3};

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
     * @internal
     */
    node_map<int> dist_label;
	
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
	void comp_dist_labels(const graph& G, std::vector<int>& numb);

    /**
     * @internal
     */
	bool has_an_admissible_arc(const node cur_node);

    /**
     * @internal
     */
	void advance(node& cur_node, node_map<edge>& last_edge);

    /**
     * @internal
     */
	void augment(graph& G, const node_map<edge>& last_edge);

    /**
     * @internal
     */
	bool retreat(const int number_of_nodes,
				 node& cur_node,
				 const node_map<edge>& last_edge,
				 std::vector<int>& numb);

    /**
     * @internal
     */
	int min_neighbour_label(const int number_of_nodes,
							const node cur_node) const;

    /**
     * @internal
     */
    double free_capacity(const node_map<edge>& last_edge) const;

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

#endif // GTL_MAXFLOW_SAP_H

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
