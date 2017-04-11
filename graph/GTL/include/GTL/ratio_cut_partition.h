/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//	ratio_cut_partition.h
//
//==========================================================================
// $Id: ratio_cut_partition.h,v 1.8 2003/01/31 08:15:04 chris Exp $

#ifndef GTL_RATIO_CUT_PARTITION_H
#define GTL_RATIO_CUT_PARTITION_H

#include <GTL/GTL.h>
#include <GTL/graph.h>
#include <GTL/node_map.h>
#include <GTL/edge_map.h>
#include <GTL/algorithm.h>

__GTL_BEGIN_NAMESPACE


/**
 * @short Heuristic graph bi-partitioning algorithm (Wei-Cheng).
 *
 * This class implements a heuristic graph bi-partitioning algorithm using
 * the ratio cut method proposed by Y. C. Wei and C. K. Cheng in 1991.
 *
 * <p> In the case E is the set of edges of the graph, the algorithm needs
 * <code>O(|E|)</code> time to proceed.
 *
 * @see fm_partition
 */
class GTL_EXTERN ratio_cut_partition : public algorithm
{
public:
    /**
     * Return type of @ref ratio_cut_partition#get_side_of_node.
     *
     * @see ratio_cut_partition#A
     * @see ratio_cut_partition#B
     */
    typedef int side_type;

    /**
     * <code>A</code> means the node is on side A.
     *
     * @see ratio_cut_partition#side_type
     */
    const static side_type A;

    /**
     * <code>B</code> means the node is on side B.
     *
     * @see ratio_cut_partition#side_type
     */
    const static side_type B;

    /**
     * Fix type of each node (needed with
     * @ref ratio_cut_partition#set_vars).
     *
     * @see ratio_cut_partition#FIXA
     * @see ratio_cut_partition#FIXB
     * @see ratio_cut_partition#UNFIXED
     */
    typedef short int fix_type;

    /**
     * <code>FIXA</code> means fix node on side <code>A</code>.
     *
     * @see ratio_cut_partition#set_vars
     */
    const static fix_type FIXA;

    /**
     * <code>FIXB</code> means fix node on side <code>B</code>.
     *
     * @see ratio_cut_partition#fixe_type
     */
    const static fix_type FIXB;

    /**
     * <code>UNFIXED</code> means node is free.
     *
     * @see ratio_cut_partition#fixe_type
     */
    const static fix_type UNFIXED;

    /**
     * Default constructor.
     *
     * @see algorithm#algorithm
     */
    ratio_cut_partition();

    /**
     * Destructor.
     *
     * @see algorithm#~algorithm
     */
    virtual ~ratio_cut_partition();

    /**
     * Sets variables.
     * Must be executed before @ref ratio_cut_partition#check!
     * <code>source_node</code> and <code>target_node</code> will be
     * determined automatically.
     *
     * @param G undirected graph
     * @param node_weight weight of each node
     * @param edge_weight weight of each edge.
     * @see ratio_cut_partition#check
     */
    void set_vars(const graph& G, const node_map<int>& node_weight,
	const edge_map<int>& edge_weight);

    /**
     * Sets variables.
     * Must be executed before @ref ratio_cut_partition#check!
     * In order to get good results, you should take two graph
     * theoretically far away nodes as source and target.
     *
     * @param G undirected graph
     * @param node_weight weight of each node
     * @param edge_weight weight of each edge
     * @param source_node start-node, remains on side <code>A</code>
     * @param target_node end-node, remains on side <code>B</code>
     * @see ratio_cut_partition#check
     */
    void set_vars(const graph& G, const node_map<int>& node_weight,
	const edge_map<int>& edge_weight, const node source_node,
	const node target_node);

    /**
     * Sets variables.
     * Must be executed before @ref ratio_cut_partition#check!
     * In order to get good results, you should take two graph
     * theoretically far away nodes as source and target. Additionally
     * <code>init_side</code> should nearly be in balance.
     * <code>source_node</code> must be on side <code>A</code> in <code>
     * init_side</code> and <code>target_node</code> on side <code>B
     * </code> respectively.
     *
     * @param G undirected graph
     * @param node_weight weight of each node
     * @param edge_weight weight of each edge
     * @param source_node start-node, remains on side <code>A</code>
     * @param target_node end-node, remains on side <code>B</code>
     * @param init_side initial bi-partitioning
     * @see ratio_cut_partition#check
     */
    void set_vars(const graph& G, const node_map<int>& node_weight,
	const edge_map<int>& edge_weight, const node source_node,
	const node target_node, const node_map<side_type>& init_side);
			
    /**
     * Sets variables.
     * Must be executed before @ref ratio_cut_partition#check!
     * In order to get good results, you should take two graph
     * theoretically far away nodes as source and target.
     * <code>source_node</code> must not be fixed on side <code>B
     * </code>.
     * <code>target_node</code> must not be fixed on side <code>A
     * </code>.
     *
     * @param G undirected graph
     * @param node_weight weight of each node
     * @param edge_weight weight of each edge
     * @param source_node start-node, remains on side <code>A</code>
     * @param target_node end-node, remains on side <code>B</code>
     * @param fixed fixed nodes
     * @see ratio_cut_partition#check
     */
    void set_vars(const graph& G, const node_map<int>& node_weight,
	const edge_map<int>& edge_weight, const node source_node,
	const node target_node, const node_map<fix_type>& fixed);

    /**
     * Sets variables.
     * Must be executed before @ref ratio_cut_partition#check!
     * In order to get good results, you should take two graph
     * theoretically far away nodes as source and target. Additionally
     * <code>init_side</code> should nearly be in balance. Fixed nodes
     * are on their fix side, their initial side is overwritten then.
     * <code>source_node</code> must be on side A in <code>init_side
     * </code> and <code>target_node</code> on side B respectively.
     * <code>source_node</code> must not be fixed on side <code>B
     * </code>.
     * <code>target_node</code> must not be fixed on side <code>A
     * </code>.
     *
     * @param G undirected graph
     * @param node_weight weight of each node
     * @param edge_weight weight of each edge
     * @param source_node start-node, remains on side <code>A</code>
     * @param target_node end-node, remains on side <code>B</code>
     * @param init_side initial bi-partitioning
     * @param fixed fixed nodes
     * @see ratio_cut_partition#check
     */
    void set_vars(const graph& G, const node_map<int>& node_weight,
	const edge_map<int>& edge_weight, const node source_node,
	const node target_node, const node_map<side_type>& init_side,
	const node_map<fix_type>& fixed);

    /**
     * Enables the storing of cut-edges. If enabled the list of
     * cut-edges can be traversed using @ref
     * ratio_cut_partition#cut_edges_iterator.
     *
     * @param set if <code>true</code> cut_edges will be stored
     * @see ratio_cut_partition#cut_edges_begin
     * @see ratio_cut_partition#cut_edges_end
     */
    void store_cut_edges(const bool set);

    /**
     * Enables the storing of nodes on their side. If enabled the nodes
     * of each side can be traversed using
     * ratio_cut_partition#nodes_of_one_side_iterator.
     *
     * @param set if <code>true</code> nodes on their side will be stored
     * @see ratio_cut_partition#nodes_of_sideA_begin
     * @see ratio_cut_partition#nodes_of_sideA_end
     * @see ratio_cut_partition#nodes_of_sideB_begin
     * @see ratio_cut_partition#nodes_of_sideB_end
     */
    void store_nodesAB(const bool set);

    /**
     * Checks whether following preconditions are satisfied:
     * <ul>
     * <li> One of the @ref ratio_cut_partition#set_vars procedures has
     * been executed before.
     * <li> graph <code>G</code> is undirected.
     * <li> if applied, <code>source_node</code> and <code>target_node
     * </code> are 2 distinct nodes with node weights > 0.
     * <li> only node_weights >= 0 are applied.
     * <li> only edge_weights >= 0 are applied.
     * <li> if <code>G</code> has more than 2 nodes, then at least
     * two of them have a weight > 0.
     * <li> if applied fixed source node, <code>fixed[source_node]
     * </code> is <code>FIXA</code>.
     * <li> if applied fixed target node, <code>fixed[target_node]
     * </code> is <code>FIXB</code>.
     * </ul>
     * 
     * @param G graph
     * @return <code>algorithm::GTL_OK</code> on success,
     * <code>algorithm::GTL_ERROR</code> otherwise
     * @see ratio_cut_partition#set_vars
     * @see algorithm#check
     */
    virtual int check(graph& G);

    /**
     * Computes a partitioning of <code>G</code>, that means a division
     * of its vertices in two sides <code>ratio_cut_partition::A</code>
     * and <code>ratio_cut_partition::B</code>.
     * 
     * @param G graph
     * @return <code>algorithm::GTL_OK</code> on success,
     * <code>algorithm::GTL_ERROR</code> otherwise
     * @see algorithm#run
     */
    int run(graph& G);

    /**
     * Gets the size of the cut after bi-partitioning.
     *
     * @return cutsize
     */
    int get_cutsize();
		
    /**
     * Gets the ratio of the cut after bi-partitioning as defined in
     * [WeiChe91].
     *
     * @return cutratio
     */
    double get_cutratio();
		
    /**
     * Gets side of the node after bi-partitioning.
     * 
     * @param n node of graph G
     * @return <code>ratio_cut_partition::A</code> if <code>n</code>
     * lies on side <code>A</code>, <code>ratio_cut_partition::B</code>
     * otherwise
     */
    side_type get_side_of_node(const node& n) const;

    /**
     * Gets side of the node after bi-partitioning.
     * 
     * @param n node of graph G
     * @return <code>ratio_cut_partition::A</code> if <code>n</code>
     * lies on side <code>A</code>, <code>ratio_cut_partition::B</code>
     * otherwise
     * @see ratio_cut_partition#get_side_of_node
     */
    side_type operator [](const node& n) const;
				
    /**
     * Gets the sum of all node weights from nodes on side <code>A
     * </code>.
     *
     * @param G graph
     * @return <code>node_weight_on_sideA</code>
     */
    int get_weight_on_sideA(const graph& G) const;

    /**
     * Gets the sum of all node weights from nodes on side <code>B
     * </code>.
     *
     * @param G graph
     * @return <code>node_weight_on_sideB</code>
     */
    int get_weight_on_sideB(const graph& G) const;

    /**
     * Iterator type for edges which belong to the cut.
     */
	typedef edges_t::const_iterator cut_edges_iterator;

    /**
     * Iterate through all edges which belong to the cut, that means
     * all edges with end-nodes on different sides.
     * It is only valid if enabled with @ref
     * ratio_cut_partition#store_cut_edges before.
     *
     * @return start for iteration through all cut edges
     */
    cut_edges_iterator cut_edges_begin() const;

    /**
     * End-Iterator for iteration through all edges which belong to the
     * cut.
     * It is only valid if enabled with @ref
     * ratio_cut_partition#store_cut_edges before.
     *
     * @return end for iteration through all cut-edges
     */
    cut_edges_iterator cut_edges_end() const;
		
    /**
     * Iterator type for nodes of a side.
     */
	typedef nodes_t::const_iterator nodes_of_one_side_iterator;

    /**
     * Iterate through all nodes which belong to side <code>A</code>,
     * It is only valid if enabled with @ref
     * ratio_cut_partition#store_nodesAB before.
     *
     * @return start for iteration through all nodes on <code>A</code>
     */
    nodes_of_one_side_iterator nodes_of_sideA_begin() const;

    /**
     * End-Iterator for iteration through all nodes which belong to side
     * <code>A</code>,
     * It is only valid if enabled with @ref
     * ratio_cut_partition#store_nodesAB before.
     *
     * @return end for iteration through all nodes on <code>A</code>
     */
    nodes_of_one_side_iterator nodes_of_sideA_end() const;

    /**
     * Iterate through all nodes which belong to side <code>B</code>,
     * It is only valid if enabled with @ref
     * ratio_cut_partition#store_nodesAB before.
     *
     * @return start for iteration through all nodes on <code>B</code>
     */
    nodes_of_one_side_iterator nodes_of_sideB_begin() const;

    /**
     * End-Iterator for iteration through all nodes which belong to side
     * <code>B</code>,
     * It is only valid if enabled with @ref
     * ratio_cut_partition#store_nodesAB before.
     *
     * @return end for iteration through all nodes on <code>B</code>
     */
    nodes_of_one_side_iterator nodes_of_sideB_end() const;

    /**
     * Resets ratio_cut_partition, i.e. prepares the algorithm to be
     * applied to another graph.
     *
     * @see algorithm#reset
     */
    virtual void reset();
protected:
    /**
     * @internal
     */
    enum direction_type {LEFT_SHIFT = 2, RIGHT_SHIFT = 3};

    /**
     * @internal
     * <code>true</code>, iff user enabled storing of cut-edges with
     * @ref ratio_cut_partition#store_cut_edges.
     */
    bool enable_cut_edges_storing;
		
    /**
     * @internal
     * List of edges which belong to the cut.
     */
	edges_t cut_edges;

    /**
     * @internal
     * <code>true</code>, iff user enabled storing of nodes with @ref
     * ratio_cut_partition#store_nodesAB.
     */
    bool enable_nodesAB_storing;
		
    /**
     * @internal
     * List of nodes which belong to side <code>A</code>.
     */
	nodes_t nodesA;

    /**
     * @internal
     * List of nodes which belong to side <code>A</code>.
     */
	nodes_t nodesB;

    /**
     * @internal
     * Corresponds to s in [WeiChe91].
     */
    node source_node;

    /**
     * @internal
     * Corresponds to t in [WeiChe91].
     */
    node target_node;

    /**
     * @internal
     * <code>true</code>, iff user has executed @ref
     * ratio_cut_partition#set_vars before @ref ratio_cut_partition#
     * check and @ref ratio_cut_partition#run.
     */
    bool set_vars_executed;

    /**
     * @internal
     * <code>true</code>, iff user has provided <code>source_node</code>
     * and <code>target_node</code>, <code>false</code> else.
     */
    bool provided_st;

    /**
     * @internal
     * <code>true</code>, iff user has provided <code>init_side</code>
     * with @ref ratio_cut_partition#set_vars, <code>false</code>
     * otherwise.
     */
    bool provided_initial_part;

    /**
     * @internal
     * <code>true</code>, iff user has provided <code>fixed</code> with
     * @ref ratio_cut_partition#set_vars, <code>false</code> otherwise.
     */
    bool provided_fix;
		
    /**
     * @internal
     * Contains information where a node is fixed.
     */
    node_map<fix_type> fixed;

    /**
     * @internal
     * <code>LEFT</code> if @ref ratio_cut_partition#left_shift_op has
     * computed last cut, <code>RIGHT</code> else.
     */
    direction_type direction;

    /**
     * @internal
     * Contains the weight of each node.
     * Corresponds to w(v) in [Leng90].
     */
    node_map<int> node_weight;

    /**
     * @internal
     * Contains the weight of each edge.
     * Corresponds to c(e) in [Leng90].
     */
    edge_map<int> edge_weight;

    /**
     * @internal
     * Contains the maximum weight of an edge in <code>G</code>.
     * (maximum of <code>edge_weight[...]</code>)
     */
    int max_edge_weight;

    /**
     * @internal
     * Contains the sum over all vertex weights on side <code>A</code>.
     * Corresponds to w(A) in [Leng90].
     */
    int node_weight_on_sideA;

    /**
     * @internal
     * Contains the sum over all vertex weights on side <code>B</code>.
     * Corresponds to w(B) in [Leng90].
     */
    int node_weight_on_sideB;
		
    /**
     * @internal
     * Counts nodes on side <code>A</code>.
     */
    int nodes_on_sideA;
		
    /**
     * @internal
     * Counts nodes on side <code>B</code>.
     */
    int nodes_on_sideB;
		
    /**
     * @internal
     * Contains information about the current side of a node.
     */
    node_map<side_type> side;
		
    /**
     * @internal
     * Corresponds to CELL array in [FidMat82]
     */
	node_map<nodes_t::iterator> position_in_bucket;
		
    /**
     * @internal
     * Contains the maximal number of adjacent to a node.
     */
    int max_vertex_degree;

    /**
     * @internal
     * Contains how many nodes an edge has on side <code>A</code>.
     */
    edge_map<int> aside;

    /**
     * @internal
     * Contains how many nodes an edge has on side <code>B</code>.
     */
    edge_map<int> bside;

    /**
     * @internal
     * Contains the unlocked nodes of an edge on side <code>A</code>.
     * (max. 2)
     */
	edge_map<nodes_t> unlockedA;

    /**
     * @internal
     * Contains the unlocked nodes of an edge on side <code>B</code>.
     * (max. 2)
     */
	edge_map<nodes_t> unlockedB;

    /**
     * @internal
     * Corresponds to D value in Leng[90].
     */
    node_map<int> gain_value;

    /**
     * @internal
     * <code>true</code>, iff <code>bucketA</code> is empty.
     */
    bool bucketA_empty;

    /**
     * @internal
     * <code>true</code>, iff <code>bucketB</code> is empty.
     */
    bool bucketB_empty;

    /**
     * @internal
     * Contains the maximum gain value of a node in
     * <code>bucketA</code>.
     */
    int max_gainA;

    /**
     * @internal
     * Contains the maximum gain value of a node in
     * <code>bucketB</code>.
     */
    int max_gainB;
		
    /**
     * @internal
     * Like a hash table over the <code>gain_value</code> of each node
     * on side <code>A</code>. (open hashing, collisions in gain buckets
     * are organized through LIFO lists)
     */
	std::vector<nodes_t> bucketA;

    /**
     * @internal
     * Like a hash table over the <code>gain_value</code> of each node
     * on side <code>B</code>. (open hashing, collisions in gain buckets
     * are organized through LIFO lists)
     */
	std::vector<nodes_t> bucketB;

    /**
     * @internal
     * Sum over all <code>edge_costs[e]</code> where edge e is an
     * element of the cut.
     */
    int cur_cutsize;

    /**
     * @internal
     * Cut ratio as defined in [WeiChe91].
     */
    double cur_cutratio;

    /**
     * @internal
     * Fix <code>FIXA</code> nodes on side <code>A</code> and <code>FIXB
     * </code> nodes on side <code>B</code>.
     */
    void divide_up(const graph& G);

    /**
     * @internal
     * Makes <code>G</code> connected for the run of this algorithm.
     * This is done by introducing edges with weight 0 since Ratio Cut
     * works well on connected graphs only.
     */
	void make_connected(graph& G, edges_t& artificial_edges);
		
    /**
     * @internal
     * Deletes the edges introduced in @ref ratio_cut_partition#
     * make_connected.
     */
	void restore(graph& G, edges_t& artificial_edges);
		
    /**
     * @internal
     * Corresponds to phase 1 in [WeiChe91].
     */
    void initialization(const graph& G);

    /**
     * @internal
     * Initialization of the data structure for each step.
     */
    void init_data_structure(const graph& G);

    /**
     * @internal
     * Computes initial gain_value for each node and inserts it in the
     * corresponding bucket data structure.
     */
    void init_filling_buckets(const graph& G);

    /**
     * @internal
     * Compute initial gain of a node on side <code>A</code>.
     * @return initial gain_value of a node on side <code>A</code>
     */
    int inital_gain_of_node_on_sideA(const node cur_node);

    /**
     * @internal
     * Compute initial gain of a node on side <code>B</code>.
     * @return initial gain_value of a node on side <code>B</code>
     */
    int inital_gain_of_node_on_sideB(const node cur_node);

    /**
     * @internal
     * Computes some maximum variables.
     */
    void init_variables(const graph& G);
		
    /**
     * @internal
     * Computes <code>max_vertex_degree</code>.
     */
    void compute_max_vertex_degree(const graph& G);

    /**
     * @internal
     * Compute source seed [WeiChe91].
     */
    void determine_source_node(const graph& G);

    /**
     * @internal
     * Compute target seed [WeiChe91].
     */
    void compute_target_node(const graph& G);

    /**
     * @internal
     * Corresponds to right shifting operation as defined in [WeiChe91].
     * Moves nodes from side <code>A</code> to <code>B</code>.
     */
    void right_shift_op(const graph& G);

    /**
     * @internal
     * Corresponds to left shifting operation as defined in [WeiChe91].
     * Moves nodes from side <code>B</code> to <code>A</code>.
     */
    void left_shift_op(const graph& G);

    /**
     * @internal
     * Moves <code>max_gain</code> node from side <code>A</code> to
     * <code>B</code>.
     * @return <code>true</code> if vertex stored in parameter <code>
     * moved_node</code> has been found
     */
    bool move_vertex_A2B(const graph& G, node& moved_node);

    /**
     * @internal
     * Moves <code>max_gain node</code> from side B to A.
     * @return <code>true</code> if vertex stored in parameter <code>
     * moved_node</code> has been found
     */
    bool move_vertex_B2A(const graph& G, node& moved_node);

    /**
     * @internal
     * Selects node with highest ratio_gain
     */
	node compute_highest_ratio_node(nodes_t node_list);

    /**
     * @internal
     * Computes <code>cut_ratio</code>.
     * @return <code>cut_ratio</code> with cutsize <code>cur_cutsize
     * </code> and current side weights <code>node_weight_on_sideA
     * </code> and <code>node_weight_on_sideB</code>
     */
    double cutratio();
				
    /**
     * @internal
     * Corresponds to r(i) in [WeiChe91].
     * @return ratio gain of a node <code>cur_node</code> on side <code>
     * A</code>
     */
    double ratio_of_node_A2B(const node cur_node);

    /**
     * @internal
     * Corresponds to r(i) in [WeiChe91].
     * @return ratio gain of a node <code>cur_node</code> on side <code>
     * B</code>
     */
    double ratio_of_node_B2A(const node cur_node);
		
    /**
     * @internal
     * Transform a range from [-a..+a] to [0..2a].
     * (reverse to @ref ratio_cut_partition#range_up)
     */
    inline int range_up(const int gain_value) const;

    /**
     * @internal
     * Transform a range from [0..2a] to [-a..+a].
     * (reverse to @ref ratio_cut_partition#range_down)
     */
    inline int range_down(const int index) const;

    /**
     * @internal
     * Executed, if <code>cur_node</code> is chosen to move from side
     * <code>A</code> to <code>B</code>.
     */
    void update_data_structure_A2B(const node cur_node,
	const bool init_mode);
		
    /**
     * @internal
     * Executed, if <code>cur_node</code> is chosen to move from side
     * <code>B</code> to <code>A</code>.
     */
    void update_data_structure_B2A(const node cur_node,
	const bool init_mode);
		
    /**
     * @internal
     * Reorganizes <code>bucketA</code> if a nodes gain of it has been
     * changed.
     */
    void update_bucketA(const node cur_node, const int old_gain,
	const int new_gain, const bool init_mode);

    /**
     * @internal
     * Reorganizes <code>bucketB</code> if a nodes gain of it has been
     * changed.
     */
    void update_bucketB(const node cur_node, const int old_gain,
	const int new_gain, const bool init_mode);
		
    /**
     * @internal
     * Recomputes <code>max_gainA</code> or <code>max_gainB</code>
     * respectively.
     */
    void update_max_gain(const side_type side);

    /**
     * @internal
     * Do some garbage collection.
     */
    void clean_step(const graph& G);

    /**
     * @internal
     * Copies side node maps.
     */
    void copy_side_node_map(const graph& G, node_map<side_type>& dest,
	const node_map<side_type> source) const;
			
    /**
     * @internal
     * Corresponds to phase 2 in [WeiChe91].
     */
    void iterative_shifting(const graph& G);

    /**
     * @internal
     * Corresponds to phase 3 in [WeiChe91].
     */
    void group_swapping(const graph& G);

    /**
     * @internal
     * Moves nodes in group swapping phase.
     * @return <code>true</code> on improvement, <code>false</code>
     * else
     */
    bool move_manager(const graph& G);

    /**
     * @internal
     * Moves a single node.
     * @return <code>true</code> if vertex stored in parameter <code>
     * moved_node</code> has been found
     */
    bool move_vertex(const graph& G, node& moved_node);

    /**
     * @internal
     * Computes list <code>cut_edges</code>.
     */
    void compute_cut_edges(const graph& G);

    /**
     * @internal
     * Computes lists <code>nodesA</code> and <code>nodesB</code>.
     */
    void compute_nodesAB(const graph& G);
private:
#ifdef _DEBUG
    /**
     * @internal
     * Prints content of bucketA with associated gain values.
     */
    void print_bucketA();

    /**
     * @internal
     * Prints content of bucketB with associated gain values.
     */
    void print_bucketB();
#endif	// _DEBUG
};

__GTL_END_NAMESPACE

#endif // GTL_RATIO_CUT_PARTITION_H

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
