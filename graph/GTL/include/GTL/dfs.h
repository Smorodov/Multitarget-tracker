/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   dfs.h
//
//==========================================================================
// $Id: dfs.h,v 1.25 2003/03/24 15:58:54 raitner Exp $

#ifndef GTL_DFS_H
#define GTL_DFS_H

#include <GTL/GTL.h>
#include <GTL/algorithm.h>

__GTL_BEGIN_NAMESPACE

/**
 * $Date: 2003/03/24 15:58:54 $
 * $Revision: 1.25 $
 * 
 * @brief Depth-First-Search (DFS) %algorithm 
 * 
 * Encapsulates the DFS %algoritm together with all the data
 * produced by a run of DFS. Since there exits so much different
 * things which one might want to calculate during a DFS this
 * class provides basically two different customization
 * features. First it is possible to take influence on the
 * behaviour of this %algortihm by changing some of the following
 * options:
 *  - dfs::start_node 
 *    (default: an arbitrary %node will be chosen)
 *  - dfs::scan_whole_graph states whether BFS will be 
 *    continued in the unused part of the %graph, if not all
 *    nodes were touched at the end of DFS started at the start-%node.
 *    (default: disabled)
 *  - dfs::calc_comp_num toggle storing of completion-numbers 
 *    for each %node, i.e. a numbering which reflects the order in which 
 *    nodes were @em finished. (default: disabled)
 *  - dfs::store_preds toggle storing the predecessor of each
 *    %node, i.e. the father in DFS-tree. (default: disabled)
 *  - dfs::store_non_tree_edges toggle storing of all non-tree-edges
 *    (tree-edges are always stored) in a list and thus enable or disable
 *    iteration through all non-tree-edges.
 *    (default: disabled)
 * 
 * But the trouble with most DFS-%algorithm is that one always
 * wants to add a little bit of code somewhere in the
 * %algorithm. And then there are only two ways to get this
 * done. The more efficient one (in terms of runtime) is to
 * implement the DFS anew and add the new code where
 * necessary. The other way (which is more efficient in terms of
 * code-writing) is to take the %algorithm as provided and run
 * through the list of nodes it returns (resulting in an extra
 * factor of 2).
 *
 * Our DFS-%algoritm class provides a new method to add small
 * pieces of code to the %algorithm: Handler. These are virtual
 * functions called at well-defined, important states of the
 * %algorithm (e.g. before a new recursive call). So the only
 * thing to do is to derive your extended DFS from this class and
 * to override the handlers where needed. In detail there are the
 * following handler supported (have a look at the source code
 * for details):
 *  - dfs::init_handler
 *  - dfs::end_handler 
 *  - dfs::entry_handler 
 *  - dfs::leave_handler
 *  - dfs::before_recursive_call_handler 
 *  - dfs::after_recursive_call_handler 
 *  - dfs::old_adj_node_handler
 *  - dfs::new_start_handler
 *
 * @em Please @em note: We do @em not claim that this set of handlers 
 * is sufficient in any way. So if you believe that some new handler is 
 * needed urgently please let us know.
 *
 * There is a lot of information stored during DFS (e.g. nodes in
 * dfs-order, list of non-tree-edges). Some of it can be obtained directly
 * by using the corresponding member-function (e.g. dfs::dfs_num),
 * but all information that can be thought of as a list (e.g. nodes in
 * dfs-order) can be accessed through iterators. In detail these are (of
 * course depending on what options are chosen!): 
 *  - dfs::dfs_iterator
 *  - dfs::tree_edges_iterator
 *  - dfs::non_tree_edges_iterator
 *  - dfs::roots_iterator
 */
class GTL_EXTERN dfs : public algorithm 
{
public:
    /**
     * @brief Constructor.
     */
    dfs ();


    /**
     * @brief Destructor.
     */
    virtual ~dfs ();
    
    int run (graph& G);

    /**
     * @brief Checks whether the preconditions for DFS are
     * satisfied.
     *
     * Currently there aren't any restricitions for the DFS
     * %algorithm.
     *
     * @param G graph.
     * @retval algorithm::GTL_OK if %algorithm can be applied
     * @retval algorithm::GTL_ERROR otherwise.
     */
    virtual int check (graph& G);

    virtual void reset ();


    //---------------------------------------------------------------------
    //   Parameters
    //---------------------------------------------------------------------
    
    /**
     * @brief Sets start-%node for DFS. 
     *
     * @param n start-node.
     */
    void start_node (const node& n) 
	{ start = n; }

    /**
     * @brief Returns start-%node for DFS.
     *
     * @return start-%node.
     */
    node start_node () const {return start;}
    
    /**
     * @brief Enables or disables scanning of the whole %graph. 
     * 
     * If enabled and the DFS started at the given start-%node
     * stops without having touched all nodes, it will be
     * continued with the next unused %node, and so on until all
     * nodes were used. This makes sure that for every %node
     * #dfs_number is defined.
     *
     * On the other hand, if this feature is disabled, one
     * will be able to check what nodes can be reached, when
     * starting a DFS at the start-%node, because for those not
     * reached #dfs_number will be 0.
     * 
     * @param set if true enable scanning the whole graph.
     * @sa dfs::roots_begin
     * @sa dfs::roots_end
     */
    void scan_whole_graph (bool set) {whole_graph = set;}
    
    /**
     * @brief Returns true iff the whole graph will be scanned.
     * 
     * @retval true iff the  whole graph will be scanned.
     * @sa dfs::roots_begin
     * @sa dfs::roots_end
     */
    bool scan_whole_graph () const {return whole_graph;}

    /**
     * @brief Enables or Disables the calculation of the completion number.
     *
     * @param set if true completion-numbers will be calculated.
     * @sa dfs::comp_num
     */
    void calc_comp_num (bool set);

    /**
     * @brief Returns true iff completion-numbers will be calculated.
     * 
     * @retval true iff completion-numbers will be calculated.
     * @sa dfs::comp_num
     */
    bool calc_comp_num () const {return comp_number != 0;}


    /**
     * @brief Enables or disables the storing of predecessors. 
     * 
     * If enabled for every %node the predecessor in DFS will be
     * stored.
     *
     * @param set if true predecessors will be stored.
     * @sa dfs::father
     */
    void store_preds (bool set);

    /**
     * @brief Returns true iff the storing of predecessors is enabled.
     * 
     * @retval true iff the storing of predecessors is enabled.
     * @sa dfs::father
     */
    bool store_preds () const {return preds != 0;}
    
    /**
     * @brief Enables the storing of back-edges. 
     * 
     * If enabled the list of non-tree-edges can be traversed in
     * the order they occured using #non_tree_edges_iterator.
     *
     * @param set if true non_tree_edges will be stored.
     * @sa dfs::non_tree_edges_begin
     * @sa dfs::non_tree_edges_end
     */
    void store_non_tree_edges (bool set);

    /**
     * @brief Returns true iff the storing of non-tree-edges is enabled.
     * 
     * @return true iff the storing of non-tree-edges is enabled.
     * @sa dfs::non_tree_edges_begin
     * @sa dfs::non_tree_edges_end
     */
    bool store_non_tree_edges () const {return back_edges != 0;}

    //---------------------------------------------------------------------
    //   Access 
    //----------------------------------------------------------------------

    /**
     * @brief Checks whether %node @a n was reached in last DFS.
     *
     * @param n %node to be checked.
     * @return true iff @a n was reached.
     */
    bool reached (const node& n) const
	{return dfs_number[n] != 0;}

    /**
     * @brief DFS-Number of @a n. 
     * 
     * Please note that DFS-Number 0 means that this %node wasn't
     * reached.
     *
     * @param n %node.
     * @return DFS-Number of @a n.
     */
    int dfs_num (const node& n) const 
	{return dfs_number[n];}

   /**
    * @brief DFS-Number of @a n. 
    *
    * Please note that DFS-Number 0 means that this %node wasn't
    * reached.
    *
    * @param n %node.
    * @return DFS-Number of @a n.
    */
    int operator[] (const node& n) const 
	{return dfs_number[n];}

    /**
     * @brief Completion-number of %node @a n, if enabled in last
     * run.
     *
     * @param n %node.
     * @return Completion-number of @a n.
     * @sa dfs::calc_comp_num
     */
    int comp_num (const node& n) const
	{assert (comp_number); return (*comp_number)[n];}

    /**
     * @brief Returns father of node @a n in DFS-forest. 
     * 
     * If @a n is a root in the forest or wasn't reached the
     * return value is @c node().
     *
     * @param n %node.
     * @return Father of @a n.
     * @sa dfs::store_preds
     */    
    node father (const node& n) const
	{assert (preds); return (*preds)[n];}

    /**
     * @brief Iterator for the tree edges of the DFS-tree.
     */
	typedef edges_t::const_iterator tree_edges_iterator;

    /**
     * @brief Iterate through all edges picked in last DFS. 
     * 
     * Please note that this edges not always form a tree. In
     * case the %graph is not (strongly) connected they form a
     * forest.
     * 
     * @return start for iteration through all edges followed in DFS.
     */
    tree_edges_iterator tree_edges_begin () const 
	{return tree.begin();}

    /**
     * @brief End-iterator for iteration through all edges picked in last DFS.
     *
     * @return end for iteration through all edges followed in DFS.
     */
    tree_edges_iterator tree_edges_end () const
	{return tree.end();}

    /**
     * @brief Iterator for the (reached) nodes in DFS-order.
     */
	typedef nodes_t::const_iterator dfs_iterator;

    /**
     * @brief Iterate through all (reached) nodes in DFS-order.
     *
     * @return start for iteration through all nodes in DFS-order.
     */
    dfs_iterator begin () const 
	{return dfs_order.begin();}

    /**
     * @brief End-Iterator for iteration through all (reached)
     * nodes in DFS-order.
     *
     * @return end for iteration through all (reached) nodes
     */
    dfs_iterator end () const 
	{return dfs_order.end();}

    /**
     * @brief Iterator for the non-tree-edges
     */
	typedef edges_t::const_iterator non_tree_edges_iterator;

    /**
     * @brief Iterate through all non-tree-edges (if enabled).
     *
     * @return start for iteration through all non-tree-edges.
     * @sa dfs::store_non_tree_edges
     */
    non_tree_edges_iterator non_tree_edges_begin () const 
	{assert (back_edges);  return back_edges->begin(); }

    /**
     * @brief End-iterator for iteration through all
     * non-tree-edges (if enabled).
     *
     * @return end for iteration through all non-tree-edges.
     * @sa dfs::store_non_tree_edges
     */
    non_tree_edges_iterator non_tree_edges_end () const 
	{assert (back_edges); return back_edges->end(); }

    /**
     * @brief Iterator for the roots of the DFS-forest.
     */
	typedef std::list<dfs_iterator>::const_iterator roots_iterator;

    /**
     * @brief Iterator pointing towards the first root in the DFS-forest.
     * 
     * <em>Please note</em> that intstead of pointing directly
     * towards the node (i.e. @c *it is of type node) the
     * iterator points towards a #dfs_iterator, which represents
     * the root (i.e. @c *it is of type #dfs_iterator).
     * 
     * Using this technique makes it possible not only to obtain
     * all the roots in the forest, but also the whole trees
     * associated with each one. This can be achieved because a
     * #root_iterator specifies the exact position of the root in
     * the DFS-ordering and by definition of DFS all the
     * descendents of the root, i.e. the whole tree, will come
     * later in DFS, such that by incrementing the #dfs_iterator,
     * a #roots_iterator points at, one can traverse the whole
     * tree with this given root.
     * 
     * Of course if the root isn't the last node in the
     * DFS-forest on will also traverse all following trees, but
     * since the first node of such a tree one will discover is
     * its root, the successor of the #roots_iterator can be used
     * as end-iterator.
     * 
     * @return start for iteration through all roots in DFS-forest.
     * @sa dfs::scan_whole_graph
     */
    roots_iterator roots_begin () const 
	{return roots.begin();}

    /**
     * @brief Iterator pointing to the end of all roots.
     * 
     * @return end for iteration through all roots in DFS-forest.
     * @sa dfs::scan_whole_graph
     */
    roots_iterator roots_end () const 
	{return roots.end();}

    /**
     * @brief Number of nodes reached in last DFS.
     *
     * @return number of reached nodes.
     * @sa dfs::scan_whole_graph
     */
    int number_of_reached_nodes () const
	{return reached_nodes;}


    //-----------------------------------------------------------------------
    //   Handler - for customization purposes
    //-----------------------------------------------------------------------

    /**
     * @brief Handler called before the start of DFS.
     *
     * @param G %graph for which DFS was invoked.
     */
    virtual void init_handler (graph& /*G*/) {}
    
    /**
     * @brief Handler called at the end of DFS.
     *
     * @param G %graph for which DFS was invoked.
     */
    virtual void end_handler (graph& /*G*/) {}

    /**
     * @brief Handler called when touching %node @a n.
     *
     * @param G %graph for which DFS was invoked.
     * @param n actual %node.
     * @param f predecessor.
     */
    virtual void entry_handler (graph& /*G*/, node& /*n*/, node& /*f*/) {}

    /**
     * @brief Handler called after all the adjacent edges of @a n
     * have been examined.
     *
     * @param G %graph for which DFS was invoked.
     * @param n actual %node.
     * @param f predecessor.
     */
    virtual void leave_handler (graph& /*G*/, node& /*n*/, node& /*f*/) {}

    /**
     * @brief Handler called when a unused %node @a n connected to the
     * actual %node by @a e is found. 
     *
     * @param G %graph for which DFS was invoked.
     * @param e %edge connecting the actual %node to the unused one.
     * @param n unused %node.
     */
    virtual void before_recursive_call_handler (graph& /*G*/, edge& /*e*/, node& /*n*/) {}
    
    /**
     * @brief Handler called after the %algorithm returns from the
     * subtree starting at @a n connected to the actual %node by
     * @a e.
     *
     * @param G %graph for which DFS was invoked.
     * @param e %edge connecting the actual %node to the unused one.
     * @param n unused %node.
     */
    virtual void after_recursive_call_handler (graph& /*G*/, edge& /*e*/, node& /*n*/) {}
    
    /**
     * @brief Handler called when a already marked %node @a n connected 
     * to the actual %node by @a e is found during the search of all 
     * adjacent edges of the actual %node.
     *
     * @param G %graph for which DFS was invoked.
     * @param e %edge connecting the actual %node to the old one.
     * @param n used %node.
     */
    virtual void old_adj_node_handler (graph& /*G*/, edge& /*e*/, node& /*n*/) {}

    /**
     * @brief Called when DFS is started with start-%node @a
     * n. 
     * 
     * This is particularly useful when DFS was invoked with the
     * #scan_whole_graph option.
     *
     * @param G %graph for which DFS was invoked.
     * @param n start-%node.
     */
    virtual void new_start_handler (graph& /*G*/, node& /*n*/) { };

private:

    /**
     * @internal
     */
    void dfs_sub (graph&, node&, node&);

protected:    

    //----------------------------------------------------------------------
    //   Data
    //----------------------------------------------------------------------

    /**
     * @internal
     */
    int act_dfs_num;
    /**
     * @internal
     */
    int act_comp_num;
    /**
     * @internal
     */
	edges_t tree;
    /**
     * @internal
     */
	nodes_t dfs_order;
    /**
     * @internal
     */
    node_map<int> dfs_number;
    /**
     * @internal
     */
    int reached_nodes;
    /**
     * @internal
     */
    edge_map<int>* used;
    /**
     * @internal
     */
	std::list<dfs_iterator> roots;
    

    //-----------------------------------------------------------------------
    // Optional 
    //-----------------------------------------------------------------------
    
    /**
     * @internal
     */
    node_map<int>* comp_number;
    /**
     * @internal
     */
    node_map<node>* preds;
    /**
     * @internal
     */
	edges_t* back_edges;
    /**
     * @internal
     */
    node start;
    /**
     * @internal
     */
    bool whole_graph;
};

__GTL_END_NAMESPACE

#endif // GTL_DFS_H

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
