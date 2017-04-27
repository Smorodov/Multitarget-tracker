/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   planarity.h
//
//==========================================================================
// $Id: planarity.h,v 1.22 2008/02/03 18:17:08 chris Exp $

#ifndef PLANARITY_H
#define PLANARITY_H

#include <GTL/GTL.h>
#include <GTL/graph.h>
#include <GTL/algorithm.h>
#include <GTL/st_number.h>
#include <GTL/embedding.h>
#include <GTL/biconnectivity.h>
#include <GTL/pq_node.h>

__GTL_BEGIN_NAMESPACE

/**
 * $Date: 2008/02/03 18:17:08 $ 
 * $Revision: 1.22 $
 * 
 * @brief Tests if a %graph can be drawn on a plane without any %edge
 * crossings 
 * 
 * This class implements the Lempel-Even-Cederbaum %planarity test using
 * PQ-trees. In case the %graph is planar a planar embedding is obtained,
 * i.e. for each %node in the %graph an ordered adjacency list is calculated,
 * such that there exists a planar drawing in which all adjacent edges
 * around a %node apply to this order.
 *
 * If the %graph is not planar Kuratowski's famous theorem states that it
 * must contain a subgraph hoemeomorphic to either K5 (the complete %graph
 * with five nodes) or K3,3 (the complete bipartite %graph with three nodes
 * each side).  In this case the nodes and edges of the tested %graph that
 * form either of these two are calculated.
 *
 * In case the %graph is planar and has @f$N@f$ nodes the algorithm needs
 * @f$\mathcal{O}(N)@f$ time for the test (including the planar embedding).
 * In case the %graph isn't planar it needs at most @f$\mathcal{O}(E)@f$
 * time if @f$E@f$ is the number of edges for both the test and the
 * detection of K5 or K3,3.
 */
class GTL_EXTERN planarity : public algorithm 
{
public:
    /**
     * @brief Creates an object of the planarity test %algorithm.
     *
     * @sa algorithm
     */
    planarity();
    
    /**
     * @brief Destructor
     */
    ~planarity();

    /**
     * @brief Checks whether planarity test can be applied to @p G.
     * 
     * This should return always @c GTL_OK. There aren't any
     * restrictions on @p G, even multiple edges and selfloops
     * are tolerated. 
     * 
     * @note Selfloops and multiple edges will not be added to
     * the planar embedding. planar_embedding::selfloops and
     * planar_embedding::multiple_edges can be used to get
     * these.
     *
     * @param G arbitrary %graph
     *
     * @retval GTL_OK if %planarity test can be applied
     * @retval GTL_ERROR if not
     *
     * @sa algorithm#check
     */
    int check(graph& G);
    
    /**
     * @brief Runs planarity test on @p G.
     * 
     * This should return always @c GTL_OK. The return value only
     * tracks errors that might occur, it is definitly @em not
     * the result of the test itself. The result of the test is
     * stored in a member variable and can be accessed via
     * #is_planar.
     *
     * @param G arbitrary %graph
     *
     * @retval GTL_OK if %planarity test was sucessfully applied 
     * @retval GTL_ERROR if not
     *
     * @sa algorithm::run
     */
    int run(graph& G);
    
    /**
     * @brief Resets algorithm object, such that it can  be applied to
     * another graph.
     *
     * @sa algorithm::reset
     */
    void reset();
    
    /**
     * @brief If @p p is true a planar embedding will be calculated in 
     * the next run. 
     *  
     * @param p @c true iff embedding should be calculated
     *
     * @sa #get_embedding
     * @sa planar_embedding
     */
    void calc_embedding(bool p)
    {
	emp = p;
	if (!emp) kup = false;
    }

    /** 
     * @brief Returns true if a planar embedding will be calculated in 
     * the next run.
     *  
     * @retval true iff embedding will be calculated
     *
     * @sa #get_embedding
     * @sa planar_embedding
     */
    bool calc_embedding () const
	{ return emp; }

   /** 
     * @brief If @p p is true the obstructions to %planarity will be
     * calculated in the next %run. 
     * 
     * This implies the calculation of an embedding.
     *  
     * @param p @c true iff obstructions to %planarity should be calculated
     *
     * @sa #get_obstruction_edges
     * @sa #get_obstruction_nodes
     */
    void calc_obstruction(bool p)
    {
	kup = p;
	if (kup) emp = true;
    }

    /**
     * @brief Returns true if the obstructions to %planarity will be
     * calculated in the next %run. 
     *  
     * @retval true iff obstructions to %planarity will be calculated
     *
     * @sa #get_obstruction_edges
     * @sa #get_obstruction_nodes
     */
    bool calc_obstruction() const
    {
	return kup;
    }

    /**
     * @brief Determines the strategy used to test a graph which is not
     * biconnected.
     * 
     * If this is enabled the graph will be made biconnected by
     * adding some new edges. This is usually faster than testing
     * the biconnected components one by one, which is done if
     * this option is disabled. By default this is enabled. 
     * 
     * @note This is not fully tested, i.e. at the moment this
     * feature should be used only for the test without embedding
     * or kuratowski graphs.
     *
     * @param p true iff %graph should be made biconnected
     *
     * @sa biconnectivity::make_biconnected
     */ 
    void make_biconnected(bool p) 
    {
	bip = p;
    }
    
    /**
     * @brief Returns strategy for testing graphs, which are not
     * biconnected.
     * 
     * @retval true iff graph will be made biconnected before test
     *
     * @sa biconnectivity#make_biconnected
     */
    bool make_biconnected() const 
    {
	return bip;
    }

    /**
     * @brief Result of last test.
     *
     * @retval true iff %graph in last %run was planar.
     */
    bool is_planar() const
    {
	return planar;
    }
    
    /**
     * @brief If %graph in last #run was planar a planar embedding is
     * calculated during the reductions. This function gives access to it.
     *
     * @return planar embedding of %graph in last %run
     *
     * @sa #calc_embedding
     */
    planar_embedding& get_embedding()
    {
	return embedding;
    }

    /**
     * @brief Returns the edges of a subgraph homeomorphic to
     * either K3,3 or K5 if %graph in last %run was not planar.
     * 
     * @return edges of subgraph homeomorphic to either K3,3 or K5
     *
     * @sa #get_obstruction_nodes
     * @sa #calc_obstruction
     */
	edges_t& get_obstruction_edges()
    {
	return ob_edges;
    }

    /**
     * @brief Returns the nodes of a subgraph homeomorphic to
     * either K3,3 or K5 if %graph in last %run was not planar.
     *
     * @return nodes of subgraph homeomorphic to either K3,3 or K5
     *
     * @sa #get_obstruction_edges
     * @sa #calc_obstruction
     */
	nodes_t& get_obstruction_nodes()
    {
	return ob_nodes;
    }
private:
    /**
     * @internal
     * Main procedure for planarity test. Assumes @p G to be undirected and
     * biconnected. Used to test whether the biconnected components of a
     * %graph are planar. 
     *
     * @param G biconnected, undirected graph
     * @param em planar embedding (should be empty)
     *
     * @retval true if @c G is planar
     */ 
    bool run_on_biconnected(graph& G, planar_embedding& em);

    /**
     * @internal
     * Adds the embedding for component @c G to the embedding of the whole
     * %graph.
     *
     * @param G biconnected graph 
     * @param em embedding obtained through testing @p G
     */
    void add_to_embedding(graph& G, planar_embedding& em);

    /**
     * @internal
     * The so called upward embedding can be obtained from the list of edges
     * one gets in the reduction steps of the %algorithm. The only problem
     * is that some of these lists may be turned later in the algorithm.
     * This procedure corrects the reversions according to the information
     * stored in @p dirs.
     *
     * @param em embedding 
     * @param st st-numbers of biconnected %graph
     * @param dirs direction indicators obtained after each reduction
     */
    void correct_embedding(planar_embedding& em,
			   st_number& st,
			   node_map<std::list<direction_indicator> >& dirs);

    /**
     * @internal
     * After the embedding has been corrected by the above procedure, we
     * have a so called upward embedding, this means only the edges leading
     * to nodes with smaller st-number than itself are in the adjacency list
     * for some node. This procedure extends the upward embedding @p em to a
     * full embedding. This is a recursive procedure  (well basically it's a
     * DFS starting at the %node with the highest st-number).
     *
     * @param n current node (used for recursion)
     * @param em embedding (at the beginning an upward embedding)
     * @param mark marks used nodes in DFS.
     * @param upward_begin marks the beginning of the upward embedding 
     */
    void extend_embedding(
	node n,
	planar_embedding& em,
	node_map<int>& mark,
	node_map<symlist<edge>::iterator >& upward_begin);
	
    /**
     * @internal
     * Make @p G the component specified in @p it by hiding everything not
     * in this subgraph. For the sake of efficiency the whole graph is
     * hidden at the beginning and then only what is in this component is 
     * restored.
     *
     * @param G whole graph; partially hidden 
     * @param it component to highlight
     *
     * @sa graph::hide
     */
    void switch_to_component(graph& G,
			     biconnectivity::component_iterator it);
 
    /**
     * @internal
     * Main procedure for detecting K5 or K3,3. Many cases have to be taken
     * into account so it is split in a lot of subroutines decribed below.
     *
     * @param G biconnected graph.
     * @param st st-numbers of @p G
     * @param act node for which the reduction failed 
     * @param fail (PQ-) node at which no matching could be applied
     * @param failed_at_root @c true iff @p fail is the root of the
     *        pertinent subtree.
     * @param em planar embedding obtained up to the moment the matchings
     *        stopped 
     * @param dirs direction indicators obtained up to the moment the
     *        matchings stopped
     * @param PQ tree
     */
    void examine_obstruction(graph& G,
			     st_number& st,
			     node act,
			     pq_node* fail,
			     bool failed_at_root,
			     planar_embedding& em,
				 node_map<std::list<direction_indicator> >& dirs,
			     pq_tree* PQ); 

    /**
     * @internal
     * Calculates a DFS-tree for the so called bush-form for the node with 
     * st-number @p stop, i.e. the induced subgraph consisting of all nodes
     * with st-number smaller than @p stop and all edges from one of these
     * to a higher numbered node lead to a virtual node with that number
     * (there may be duplicates).
     *
     * @param act used in recursion; starts with node numbered 1
     * @param mark marks for DFS; initially for all nodes 0
     * @param st st-numbers for graph
     * @param stop lowest st-number of virtual nodes
     * @param to_father stores the edge to predecessor of each node
     */
    void dfs_bushform(node act,
		      node_map<int>& mark,
		      st_number& st,
		      int stop,
		      node_map<edge>& to_father);

    
    /**
     * @internal
     * In case the reduction failed at a Q-node the boundary of the
     * biconnected component the Q-node represents can be obtained from @p
     * em.
     * No return value is needed, since all the edges on the boundary are
     * added to the obstruction edges (although some of them have to be
     * deleted in some cases).
     *
     * @param n node with lowest st-number in biconnected component 
     * @param em planar embedding (at least for this component)
     */
    void attachment_cycle (node n, planar_embedding& em);

    /**
     * @internal
     * Marks all neighbors of leaves in the subtree rooted at @p n.
     * In some cases where the reduction fails at a Q-node, which is not the
     * root of the pertinent subtree, an adjacent edge of the node for which
     * the reduction failed, which does not lead to that component has to be
     * found.
     *
     * @param n root of subtree
     * @param mark edges in subtree recieve 1, all other are unchanged.
     */
    void mark_all_neighbors_of_leaves (pq_node* act, node_map<int>& mark);
    
    /**
     * @internal
     * Searches one full and one empty leaf beneath @p partial. The join of
     * these leaves and the node on the boundary @p v to which @p partial is
     * attached is added to the obstruction nodes. All edges that form this
     * join are added to the obstruction edges.
     *
     * @param partial partial %node 
     * @param mark nodes already used 
     * @param to_father predecessor relation in DFS tree
     * @param v node on the boundary 
     * @return empty leaf
     */
    pq_leaf* run_through_partial(q_node* partial,
				 node_map<int>& mark,
				 node_map<edge>& to_father,
                                 node v);

    /**
     * @internal
     * Uses @p to_father to determine an already marked predecessor.
     *
     * @param act node
     * @param mark nodes already used
     * @param to_father predecessor relation in DFS tree
     *
     * @return marked node
     */
    node up_until_marked(node act,
			 node_map<int>& mark,
			 node_map<edge>& to_father);

    /**
     * @internal
     * Always uses a adjacent node with higher st-number as predecessor.
     * Searches marked predecessor.
     *
     * @param act node
     * @param mark nodes already used
     * @param st used to determine predecessor
     *
     * @return marked node
     */
    node up_until_marked(node act,
			 node_map<int>& mark,
			 st_number& st);

    /**
     * @internal
     * Assumes that @p n is non empty. Searches full leaf beneath @p n.
     *
     * @param n (PQ-) node
     *
     * @return full leaf in subtree of @p n
     */
    pq_leaf* search_full_leaf (pq_node* n);
	
    /**
     * @internal
     * Assumes that @p n is non full. Searches empty leaf beneath @p n.
     *
     * @param n (PQ-) node
     *
     * @return empty leaf in subtree of @p n
     */
    pq_leaf* search_empty_leaf(pq_node* n);

    /**
     * @internal
     * Reduction failed at a P-%node, which had at least three pertial
     * sons.
     *
     * @param p_fail P-%node at which reduction failed
     * @param act node for which reduction failed
     * @param _st st-numbers of graph
     * @param to_father predecessors in DFS-tree of bushform
     * @param G graph tested
     */
    void case_A(p_node* p_fail,
		node act,
		st_number& _st,
		node_map<edge> to_father,
		graph& G);

    /**
     * @internal
     * Reduction failed at a P-%node, which isn't the root of the pertinent
     * subtree and had at least two partial children.
     *
     * @param p_fail P-%node at which reduction failed
     * @param act node for which reduction failed
     * @param _st st-numbers of graph
     * @param to_father predecessors in DFS-tree of bushform
     * @param G graph tested
     */
    void case_B(p_node* p_fail,
		node act,
		st_number& _st,
		node_map<edge> to_father,
		graph& G);

    /**
     * @internal
     * Reduction failed at a Q-node, such that there exist children a < b <
     * c and a and c are both non-empty and b is non-full.
     *
     * @param nodes nodes on the boundary of @p q_fail to which the sons a,
     *        b, c are attached. 
     * @param leaves leaves in the subtrees of a, b, c. For a and c full 
     *        leaves and an empty one for b.
     * @param _st st-numbers of graph
     * @param to_father predecessors in DFS-tree of bushform
     * @param G graph tested
     * @param q_fail Q-node at which reduction failed
     */
    void case_C(node* nodes,
		pq_leaf** leaves,
		st_number& _st,
		node_map<edge> to_father,
		graph& G,
		q_node* q_fail);

    /**
     * @internal
     * Reduction failed at a non-root Q-node, such that there exist children
     * a < b < c and a and c are both non-full and b is non-empty.
     *
     * @param nodes nodes on the boundary of @p q_fail to which the sons a,
     *        b, c are attached. 
     * @param leaves leaves in the subtrees of a, b, c. For a and c full
     *        leaves and an empty one for b.
     * @param _st st-numbers of graph
     * @param to_father predecessors in DFS-tree of bushform
     * @param G graph tested
     * @param q_fail Q-node at which reduction failed
     */
    void case_D(node* nodes,
		pq_leaf** leaves,
		st_number& _st,
		node_map<edge> to_father,
		graph& G,
		q_node* q_fail);

    /**
     * @internal
     * Reduction failed at a non-root Q-node which has only two children, 
     * both partial.
     *
     * @param nodes nodes on the boundary of @p q_fail to which the two
     *        partial sons are attached. 
     * @param leaves two leaves in each subtree of a partial son. One full
     *        other empty.
     * @param _st st-numbers of graph
     * @param to_father predecessors in DFS-tree of bushform
     * @param G graph tested
     * @param q_fail Q-node at which reduction failed
     */
    void case_E(node* nodes,
		pq_leaf** leaves,
		st_number& _st,
		node_map<edge> to_father,
		graph& G,
		q_node* q_fail);

#ifdef _DEBUG
    /**
     * @internal
     */
    void write_bushform(graph& G, st_number& _st, int k, const char* name,
                        const node_map<int>& mark, const node_map<edge>& to_father);

    /**
     * @internal
     */    
	void write_node(std::ostream& os, int id, int label, int mark);
#endif
    
    /**
     * @internal
     */
	edges_t ob_edges;
    
    /**
     * @internal
     */
	nodes_t ob_nodes;
    
    /**
     * @internal
     */
    planar_embedding embedding;
    
    /**
     * @internal
     */
    bool planar;
    
    /**
     * @internal
     */
    bool emp;
    
    /**
     * @internal
     */
    bool kup;
    
    /**
     * @internal
     */
    bool bip;
};

__GTL_END_NAMESPACE

#endif // PLANARITY_H

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
