/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//	fm_partition.h
//
//==========================================================================
// $Id: fm_partition.h,v 1.8 2003/01/31 08:15:05 chris Exp $

#ifndef GTL_FM_PARTITION_H
#define GTL_FM_PARTITION_H

#include <GTL/GTL.h>
#include <GTL/graph.h>
#include <GTL/node_map.h>
#include <GTL/edge_map.h>
#include <GTL/algorithm.h>

__GTL_BEGIN_NAMESPACE


/**
 * @short Heuristic graph bi-partitioning algorithm (Fiduccia-Mattheyses).
 *
 * This class implements a heuristic graph bi-partitioning algorithm, based
 * on iterative movement, proposed by C. M. Fiduccia and R. M. Mattheyses
 * in 1982.
 *
 * <p> In the case E is the set of edges of the graph, the algorithm needs
 * <code>O(|E|)</code> time to proceed.
 *
 * @see ratio_cut_partition
 */
class GTL_EXTERN fm_partition : public algorithm
{
public:
    /**
     * Return type of @ref fm_partition#get_side_of_node.
     *
     * @see fm_partition#A
     * @see fm_partition#B
     */
    typedef int side_type;

    /**
     * <code>A</code> means the node is on side A.
     *
     * @see fm_partition#side_type
     */
    const static side_type A;

    /**
     * <code>B</code> means the node is on side B.
     *
     * @see fm_partition#side_type
     */
    const static side_type B;

    /**
     * Fix type of each node (needed with @ref fm_partition#set_vars).
     *
     * @see fm_partition#FIXA
     * @see fm_partition#FIXB
     * @see fm_partition#UNFIXED
     */
    typedef short int fix_type;

    /**
     * <code>FIXA</code> means fix node on side <code>A</code>.
     *
     * @see fm_partition#set_vars
     */
    const static fix_type FIXA;

    /**
     * <code>FIXB</code> means fix node on side <code>B</code>.
     *
     * @see fm_partition#fixe_type
     */
    const static fix_type FIXB;

    /**
     * <code>UNFIXED</code> means node is free.
     *
     * @see fm_partition#fixe_type
     */
    const static fix_type UNFIXED;

    /**
     * Default constructor.
     *
     * @see fm_partition#fixe_type
     */
    fm_partition();

    /**
     * Destructor.
     *
     * @see algorithm#~algorithm
     */
    virtual ~fm_partition();

    /**
     * Sets variables.
     * Must be executed before @ref fm_partition#check!
     *
     * @param G undirected graph
     * @param node_weight weight of each node
     * @param edge_weight weight of each edge
     * @see fm_partition#check
     */
    void set_vars(const graph& G, const node_map<int>& node_weight,
	const edge_map<int>& edge_weight);

    /**
     * Sets variables.
     * Must be executed before @ref fm_partition#check!
     * In order to get good results, <code>init_side</code> should
     * almost be in balance.
     *
     * @param G undirected graph
     * @param node_weight weight of each node
     * @param edge_weight weight of each edge
     * @param init_side initial bi-partitioning
     * @see fm_partition#check
     */
    void set_vars(const graph& G, const node_map<int>& node_weight,
	const edge_map<int>& edge_weight,
	const node_map<side_type>& init_side);

    /**
     * Sets variables.
     * Must be executed before @ref fm_partition#check!
     *
     * @param G undirected graph
     * @param node_weight weight of each node
     * @param edge_weight weight of each edge
     * @param fixed fixed nodes
     * @see fm_partition#check
     */
    void set_vars(const graph& G, const node_map<int>& node_weight,
	const edge_map<int>& edge_weight,
	const node_map<fix_type>& fixed);

    /**
     * Sets variables.
     * Must be executed before @ref fm_partition#check!
     * In order to get good results, <code>init_side</code> should
     * almost be in balance. Fixed nodes are on their fix side, their
     * initial side is overwritten then.
     *
     * @param G undirected graph
     * @param node_weight weight of each node
     * @param edge_weight weight of each edge
     * @param init_side initial bi-partitioning
     * @param fixed fixed nodes
     * @see fm_partition#check
     */
    void set_vars(const graph& G, const node_map<int>& node_weight,
	const edge_map<int>& edge_weight,
	const node_map<side_type>& init_side,
	const node_map<fix_type>& fixed);

    /**
     * Enables the storing of cut-edges. If enabled the list of
     * cut-edges can be traversed using @ref
     * fm_partition#cut_edges_iterator.
     *
     * @param set if <code>true</code> cut_edges will be stored
     * @see fm_partition#cut_edges_begin
     * @see fm_partition#cut_edges_end
     */
    void store_cut_edges(const bool set);

    /**
     * Enables the storing of nodes on their side. If enabled the nodes
     * of each side can be traversed using @ref
     * fm_partition#nodes_on_one_side_iterator.
     *
     * @param set if <code>true</code> nodes will be stored on their sides
     * @see fm_partition#nodes_of_sideA_begin
     * @see fm_partition#nodes_of_sideA_end
     * @see fm_partition#nodes_of_sideB_begin
     * @see fm_partition#nodes_of_sideB_end
     */
    void store_nodesAB(const bool set);

    /**
     * Checks whether following preconditions are satisfied:
     * <ul>
     * <li> @ref fm_partition#set_vars has been executed before.
     * <li> graph <code>G</code> is undirected.
     * <li> only node_weights >= 0 are applied.
     * <li> only edge_weights >= 0 are applied.
     * </ul>
     * 
     * @param G graph
     * @return <code>algorithm::GTL_OK</code> on success,
     * <code>algorithm::GTL_ERROR</code> otherwise
     * @see fm_partition#set_vars
     * @see algorithm#check
     */
    virtual int check(graph& G);

    /**
     * Computes a partitioning with <code>G</code>, that means a
     * division of its vertices in two sides <code>fm_partition::A
     * </code> and <code>fm_partition::B</code>.
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
     * Gets the number of passes needed to create a bi-partition with
     * this heuristic.
     *
     * @return number of passes
     */
    int get_needed_passes();
		
    /**
     * Gets side of the node after bi-partitioning.
     * 
     * @param n node of graph @c G
     * @return <code>fm_partition::A</code> if <code>n</code> lies on
     * side <code>A</code>, <code>fm_partition::B</code> otherwise
     */
    side_type get_side_of_node(const node& n) const;

    /**
     * Gets side of the node after bi-partitioning.
     * 
     * @param n node of graph @c G
     * @return <code>fm_partition::A</code> if <code>n</code> lies on
     * side <code>A</code>, <code>fm_partition::B</code> otherwise
     * @see fm_partition#get_side_of_node
     */
    side_type operator [](const node& n) const;
				
    /**
     * Gets the sum of all node weights from nodes on side A.
     *
     * @param G graph
     * @return <code>node_weight_on_sideA</code>
     */
    int get_weight_on_sideA(const graph& G) const;

    /**
     * Gets the sum of all node weights from nodes on side B.
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
     * fm_partition#store_cut_edges before.
     *
     * @return start for iteration through all cut edges
     */
    cut_edges_iterator cut_edges_begin() const;

    /**
     * End-Iterator for iteration through all edges which belong to the
     * cut.
     * It is only valid if enabled with @ref
     * fm_partition#store_cut_edges before.
     *
     * @return end for iteration through all cut-edges
     */
    cut_edges_iterator cut_edges_end() const;
		
    /**
     * Iterator type of nodes of a side.
     */
	typedef nodes_t::const_iterator nodes_of_one_side_iterator;

    /**
     * Iterate through all nodes which belong to side <code>A</code>.
     * It is only valid if enabled with @ref
     * fm_partition#store_nodesAB before.
     *
     * @return start for iteration through all nodes on <code>A</code>
     */
    nodes_of_one_side_iterator nodes_of_sideA_begin() const;

    /**
     * End-Iterator for iteration through all nodes which belong to side
     * <code>A</code>.
     * It is only valid if enabled with @ref
     * fm_partition#store_nodesAB before.
     *
     * @return end for iteration through all nodes on <code>A</code>
     */
    nodes_of_one_side_iterator nodes_of_sideA_end() const;

    /**
     * Iterate through all nodes which belong to side <code>B</code>,
     * It is only valid if enabled with @ref
     * fm_partition#store_nodesAB before.
     *
     * @return start for iteration through all nodes on <code>B</code>
     */
    nodes_of_one_side_iterator nodes_of_sideB_begin() const;

    /**
     * End-Iterator for iteration through all nodes which belong to side
     * <code>B</code>,
     * It is only valid if enabled with @ref
     * fm_partition#store_nodesAB before.
     *
     * @return end for iteration through all nodes on <code>B</code>
     */
    nodes_of_one_side_iterator nodes_of_sideB_end() const;

    /**
     * Resets fm_partition, i.e. prepares the algorithm to be applied
     * to another graph.
     *
     * @see algorithm#reset
     */
    virtual void reset();
protected:
    /**
     * @internal
     * <code>true</code>, iff user enabled storing of cut-edges with
     * @ref fm_partition#store_cut_edges.
     */
    bool enable_cut_edges_storing;
		
    /**
     * @internal
     * List of edges which belong to the cut.
     */
	edges_t cut_edges;

    /**
     * @internal
     * <code>true</code>, iff user enabled storing of nodes with
     * @ref fm_partition#store_nodesAB.
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
     * <code>true</code>, iff user has executed @ref fm_partition#
     * set_vars before @ref fm_partition#check and @ref fm_partition#
     * run.
     */
    bool set_vars_executed;
		
    /**
     * @internal
     * <code>true</code>, iff user has provided <code>init_side</code>
     * with @ref fm_partition#set_vars, <code>false</code> otherwise.
     */
    bool provided_initial_part;
		
    /**
     * @internal
     * <code>true</code>, iff user has provided <code>fixed</code> with
     * @ref fm_partition#set_vars, <code>false</code> otherwise.
     */
    bool provided_fix;
		
    /**
     * @internal
     * Contains information where a node is fixed.
     */
    node_map<fix_type> fixed;

    /**
     * @internal
     * Contains the weight of each node.
     * Corresponds to w(v) in [Leng90].
     */
    node_map<int> node_weight;

    /**
     * @internal
     * Contains the maximum weight of a node in <code>G</code>.
     * (maximum of <code>node_weight[...]</code>)
     */
    int max_node_weight;

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
     * Contains the sum over all vertex weights in <code>G</code>.
     * Corresponds to w(V) in [Leng90].
     */
    int total_node_weight;

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
     * Number of needed passes.
     */
    int no_passes;

    /**
     * @internal
     * Fix <code>FIXA</code> nodes on side <code>A</code> and <code>FIXB
     * </code> nodes on side <code>B</code>.
     */
    void divide_up(const graph& G);

    /**
     * @internal
     * Hides self loops of <code>G</code>.
     */
    void hide_self_loops(graph& G);

    /**
     * @internal
     * Computes <code>max_edge_weight</code>, <code>max_node_weight
     * </code> and <code>total_node_weight</code>.
     */
    void init_variables(const graph& G);
		
    /**
     * @internal
     * Divides nodes of <code>G</code> arbitrary into two sides <code>A
     * </code> and <code>B</code>. Here, <code>side</code> will be
     * filled with an arbitrary feasible solution.
     */
    void create_initial_bipart(const graph& G);

    /**
     * @internal
     * Shuffles order of <code>node_vector</code> with size <code>
     * vector_size</code>.
     */
    void shuffle_vector(const int vector_size,
		std::vector<graph::node_iterator>& node_vector);
		
    /**
     * @internal
     * Computes <code>max_vertex_degree</code>.
     */
    void compute_max_vertex_degree(const graph& G);

    /**
     * @internal
     * Runs as much passes as needed.
     */
    void pass_manager(const graph& G);

    /**
     * @internal
     * Copies side node maps.
     */
    void copy_side_node_map(const graph& G, node_map<side_type>& dest,
	const node_map<side_type> source) const;
			
    /**
     * @internal
     * Initialization of the data structure for each pass.
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
     * Moves nodes within a pass.
     */
    void move_manager(const graph& G);

    /**
     * @internal
     * Move a single node
     * @return <code>true</code> if vertex stored in parameter <code>
     * moved_node</code> has been found
     */
    bool move_vertex(const graph& G, node& moved_node);

    /**
     * @internal
     * Only valid on unlocked nodes!
     * @return <code>true</code> if a certain balance criterion can be
     * hold, <code>false</code> otherwise
     */
    bool balance_holds(const graph& G, const node cur_node);

    /**
     * @internal
     * Executed, if <code>cur_node</code> is chosen to move from side
     * <code>A</code> to <code>B</code>.
     */
    void update_data_structure_A2B(const node cur_node);

    /**
     * @internal
     * Executed, if <code>cur_node</code> is chosen to move from side
     * <code>B</code> to <code>A</code>.
     */
    void update_data_structure_B2A(const node cur_node);
		
    /**
     * @internal
     * Reorganizes <code>bucketA</code> if a nodes gain of it has been
     * changed.
     */
    void update_bucketA(const node cur_node, const int old_gain,
	const int new_gain);

    /**
     * @internal
     * Reorganizes <code>bucketB</code> if a nodes gain of it has been
     * changed.
     */
    void update_bucketB(const node cur_node, const int old_gain,
	const int new_gain);
		
    /**
     * @internal
     * Recomputes <code>max_gainA</code> or <code>max_gainB</code>
     * respectively.
     */
    void update_max_gain(const side_type side);

    /**
     * @internal
     * Transform a range from [-a..+a] to [0..2a].
     * (reverse to @ref fm_partition#range_up)
     */
    inline int range_up(const int gain_value) const;

    /**
     * @internal
     * Transform a range from [0..2a] to [-a..+a].
     * (reverse to @ref fm_partition#range_down)
     */
    inline int range_down(const int index) const;

    /**
     * @internal
     * Do some garbage collection.
     */
    void clean_pass(const graph& G);
		
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

#endif // GTL_FM_PARTITION_H

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
