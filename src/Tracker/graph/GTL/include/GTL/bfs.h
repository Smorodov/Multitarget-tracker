/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   bfs.h
//
//==========================================================================
// $Id: bfs.h,v 1.14 2003/03/24 15:58:54 raitner Exp $

#ifndef GTL_BFS_H
#define GTL_BFS_H

#include <GTL/GTL.h>
#include <GTL/algorithm.h>
#include <GTL/node_map.h>

#include <deque>

__GTL_BEGIN_NAMESPACE

/**
 * $Date: 2003/03/24 15:58:54 $
 * $Revision: 1.14 $
 * 
 * @brief Breadth-First-Search (BFS) %algorithm.
 *
 * Encapsulates the BFS %algorithm together with all data
 * produced by it. There are a few parameters, which on the one
 * hand influence the behaviour of BFS (e.g. bfs::start_node) and
 * on the other hand toggle the storing of extra information,
 * such as the level-number of each %node. In detail these are: 
 *  - bfs::start_node 
 *    (default: an arbitrary %node will be chosen)
 *  - bfs::scan_whole_graph states whether BFS will be 
 *    continued in the unused part of the %graph, if not all
 *    nodes were touched at the end of BFS started at the start-%node.
 *    (default: disabled)
 *  - bfs::calc_level toggle storing of level-numbers for each
 *    %node, i.e. its distance from the start-%node.
 *    (default: disabled)
 *  - bfs::store_preds toggle storing the predecessor of each
 *    %node, i.e. the father in the BFS-tree. (default: disabled)
 *  - bfs::store_non_tree_edges toggle storing of all non_tree_edges
 *    (tree_edges are always stored) in a list and thus enable or disable
 *    iteration through all non_tree_edges.
 *    (default: disabled)
 *
 * @em Please @em note that the %algorithm always starts with the
 * given start-%node (if none was given, the first %node is chosen
 * and stored, thus after BFS the root of the tree is always
 * accesible via bfs::start_node) and continues until no more
 * unused nodes are reachable from already used ones. Thus if the
 * %graph isn't connected not @em all nodes will be reached. If 
 * bfs::scan_whole_graph isn't set the BFS stops here. If it is
 * set, the BFS will be continued with the next unused %node and
 * so on until all nodes were used.
 *
 * For further customization a few virtual functions, so called
 * handler, are called at crucial stages of the %algorithm. In
 * this basic implementation all of these handler are empty. So
 * if one wants to add only a few lines of code (e.g. some new
 * numbering) he is likely to take this class as base-class and
 * override the handler where neccessary. In detail these are
 * (please look at the source code to see where they are called):
 *    -  bfs::init_handler
 *    -  bfs::end_handler 
 *    -  bfs::popped_node_handler 
 *    -  bfs::finished_node_handler 
 *    -  bfs::unused_node_handler 
 *    -  bfs::used_node_handler 
 *    -  bfs::new_start_handler 
 *
 * @em Please @em note: We do @em not claim that the set of
 * handlers provided is sufficient in any way. So if you believe
 * that some new handler is needed urgently please let us know.
 *
 * There is a lot of information stored during BFS (e.g. nodes in
 * bfs-order, list of non-tree edges). Some of it can be obtained directly
 * by using the corresponding member-function (e.g.  bfs::bfs_num),
 * but all information that can be thought of as a list (e.g. nodes in
 * bfs-order) can be accessed through iterators. In detail these are (of
 * course depending on what options are chosen!): 
 *    -  bfs::bfs_iterator
 *    -  bfs::tree_edges_iterator
 *    -  bfs::non_tree_edges_iterator
 *    -  bfs::roots_iterator
 */
class GTL_EXTERN bfs : public algorithm
{
public:
    
    /**
     * @brief Constructor. 
     */
    bfs ();

    /**
     * @brief Destructor.
     */
    virtual ~bfs ();

    int run (graph& G);

    /**
     * @brief Checks whether the preconditions for BFS are satisfied. 
     *
     * Currently there aren't any restricitions for the BFS %algorithm.
     *
     * @param G graph.
     * @retval algorithm::GTL_OK if %algorithm can be applied
     * @retval algorithm::GTL_ERROR otherwise.
     */
    virtual int check (graph& /*G*/) { return GTL_OK; }

    virtual void reset ();
    
    //-----------------------------------------------------------------------
    //  Parameters
    //-----------------------------------------------------------------------

    /**
     * @brief Sets start-%node for BFS. 
     * 
     * The default start-%node is the invalid %node (node::node()),
     * in this case an arbitrary %node is chosen and stored when
     * BFS is run.
     *
     * @param n start-%node.
     */
    void start_node (const node& n) {start = n;}

    /**
     * @brief Returns start-%node for BFS.
     *
     * @return start-%node.
     */
    node start_node () const {return start;}

    /**
     * @brief Enables or disables scanning of the whole %graph. 
     * 
     * If enabled and the BFS started at the given start-%node
     * stops without having touched all nodes, it will be
     * continued with the next unused %node, and so on until all
     * nodes were used. This makes sure that for every %node
     * bfs::bfs_num is defined.
     *
     * If this feature is disabled, you are able to check what
     * nodes can be reached, when starting a BFS at the
     * start-%node, because for those not reached bfs::bfs_num
     * will be 0.
     *
     * @param set if true enable scanning the whole %graph.
     * @sa bfs::roots_begin, bfs::roots_end
     */
    void scan_whole_graph (bool set) {whole_graph = set;}
    
    /**
     * @brief Returns whether the whole graph will be scanned.
     * 
     * @retval true iff the whole graph will be scanned.
     * @sa bfs::roots_begin, bfs::roots_end
     */
    bool scan_whole_graph () const {return whole_graph;}

    /**
     * @brief Enables or disables the calculation of level-numbers for each 
     * %node. 
     * 
     * If enabled each %node gets a level-number, i.e. its
     * distance from the start-%node.
     *
     * @param set if true level-number will be calculated.
     * @sa bfs::level
     */
    void calc_level (bool set);
    
    /**
     * @brief Returns whether level-numbers will be calculated.
     * 
     * @retval true iff level-numbers will be calculated.
     * @sa bfs::level
     */
    bool calc_level () const {return level_number != 0;}

    /**
     * @brief Enables or disables the storing of non-tree-edges. 
     * 
     * If enabled all non-tree-edges will be stored in
     * the order they occured. 
     * 
     * @param set if true non-tree-edges will be stored.
     * @sa bfs::non_tree_edges_begin, bfs::non_tree_edges_end
     */
    void store_non_tree_edges (bool set);

    /**
     * @brief Returns whether the storing of non-tree-edges is
     * enabled.
     * 
     * @retval true iff the storing of non-tree-edges is enabled.
     * @sa bfs::non_tree_edges_begin, bfs::non_tree_edges_end
     */
    bool store_non_tree_edges () const {return non_tree != 0;}


    /**
     * @brief Enables or disables the storing of predecessors. 
     * 
     * If enabled for every %node the predecessor in the BFS-forest
     * will be stored.
     *
     * @param set if true predecessors will be stored.
     * @sa bfs::father
     */
    void store_preds (bool set);

    /**
     * @brief Returns whether the storing of predecessors is enabled.
     * 
     * @retval true iff the storing of predecessors is enabled.
     * @sa bfs::father
     */
    bool store_preds () const {return preds != 0;}

    /**
     * @brief Checks whether %node @a n was reached in BFS.
     *
     * @param n %node.
     * @retval true iff @a n was reached.
     */
    bool reached (const node& n) const
	{return bfs_number[n] != 0;}

    /**
     * @brief BFS-number of @a n. 
     * 
     * @em Please @em note that BFS-number 0 means that this %node wasn't
     * reached.
     *
     * @param n %node.
     * @return BFS-number of @a n.
     */
    int bfs_num (const node& n) const 
	{return bfs_number[n];}

    /**
     * @brief BFS-number of @a n. 
     * 
     * @em Please @em note that BFS-number 0 means that this %node wasn't
     * reached.
     *
     * @param n %node.
     * @return BFS-number of @a n.
     */
    int operator[] (const node& n) const 
	{return bfs_number[n];}

    /**
     * @brief Level-number of %node @a n.
     * 
     * @em Please @em note that this requires that this option
     * was enabled during last run.
     *
     * @param n node.
     * @return level-number of @a n.
     * @sa bfs::calc_level
     */
    int level (const node& n) const
	{assert (level_number); return (*level_number)[n];}

    /**
     * @brief Father of %node @a n in BFS-forest. 
     * 
     * If @a n is a root in the forest or wasn't reached the
     * return value is the invalid %node node::node().  
     * 
     * @em Please @em note that this requires that this option
     * was enabled during last run.
     *
     * @param n node.
     * @return Father of @a n.
     * @sa bfs::store_preds
     */
    node father (const node& n) const
	{assert (preds); return (*preds)[n];}

    /**
     * @brief Iterator for tree-edges. 
     */
	typedef edges_t::const_iterator tree_edges_iterator;

    /**
     * @brief Iterate through all tree-edges of last BFS. 
     * 
     * @em Please @em note that this edges not always form a
     * tree. In case the %graph is not (strongly) connected and
     * the whole graph was scanned, they form a forest.
     * 
     * @return Start for iteration through all tree-edges.
     */
    tree_edges_iterator tree_edges_begin () const 
	{return tree.begin();}

    /**
     * @brief End-iterator for iteration through all tree-edges
     * picked of last BFS.
     *
     * @return End for iteration through all tree-edges.
     */
    tree_edges_iterator tree_edges_end () const
	{return tree.end();}
   
    /**
     * @brief Iterator for nodes in BFS-order. 
     */
	typedef nodes_t::const_iterator bfs_iterator;

    /**
     * @brief Iterate through all (reached) nodes in BFS-Order.
     *
     * @return Start for iteration through all nodes in BFS-order.
     */
    bfs_iterator begin () const 
	{return bfs_order.begin();}

    /**
     * @brief End-iterator for iteration through all (reached)
     * nodes in BFS-Order.
     *
     * @return End for iteration through all (reached) nodes
     */
    bfs_iterator end () const 
	{return bfs_order.end();}

    /**
     * @brief Iterator for non-tree-edges.
     */
	typedef edges_t::const_iterator non_tree_edges_iterator;

    /**
     * @brief Iterate through all non-tree-edges (if enabled).
     *
     * @return Start for iteration through all non-tree-edges.
     * @sa bfs::store_non_tree_edges
     */
    non_tree_edges_iterator non_tree_edges_begin () const 
	{assert (non_tree);  return non_tree->begin(); }

    /**
     * @brief End-iterator for iteration through all
     * non-tree-edges (if enabled).
     *
     * @return End for iteration through all non-tree-edges.
     * @sa bfs::store_non_tree_edges
     */
    non_tree_edges_iterator non_tree_edges_end () const 
	{assert (non_tree); return non_tree->end(); }
    
    /**
     * @brief Iterator for roots of trees in BFS-forest.
     */
	typedef std::list<bfs_iterator>::const_iterator roots_iterator;

    /**
     * @brief Iterator pointing towards the first root in the
     * BFS-forest.  
     * 
     * @em Please @em note that instead of pointing directly
     * towards the %node (i.e. @c *it is of type @c node)
     * the iterator points towards a bfs-iterator, which
     * represents the root (i.e. @c *it is of type
     * @c bfs_iterator).
     * 
     * Using this technique makes it possible not only to obtain
     * all the roots in the forest, but also the whole trees
     * associated with each one. This can be achieved because a
     * @c root_iterator specifies the exact position of the root
     * in the BFS-ordering and by definition of BFS all the
     * descendents of the root, i.e. the whole tree below, will
     * come later in BFS, such that by incrementing the @c
     * bfs_iterator a @c roots_iterator refers to, one can
     * traverse the whole tree with this given root.
     *
     * Of course if the root isn't the last %node in the
     * BFS-forest all following trees also will be traversed. But
     * since the first %node of such a tree, that will be
     * discovered, is its root, the successor of the @c
     * roots_iterator can be used as end-iterator.
     * 
     * @return Start for iteration through all roots in BFS-forest.
     * @sa bfs::scan_whole_graph
     */
    roots_iterator roots_begin () const 
	{return roots.begin();}

    /**
     * @brief Iterator pointing to the end of all roots.
     * 
     * @return End for iteration through all roots in BFS-forest.
     * @sa bfs::scan_whole_graph
     */
    roots_iterator roots_end () const 
	{return roots.end();}

    /**
     * @brief Number of nodes reached in last BFS.
     *
     * @return Number of reached nodes.
     * @sa bfs::scan_whole_graph
     */
    int number_of_reached_nodes () const
	{return reached_nodes;}

    //-----------------------------------------------------------------------
    //   Handler
    //-----------------------------------------------------------------------

    /**
     * @brief Called at the start of BFS. 
     *
     * @param G %graph for which BFS was invoked.
     */
    virtual void init_handler (graph& /*G*/) { };

    /**
     * @brief Called right before the end of BFS.
     *
     * @param G %graph for which BFS was invoked.
     */
    virtual void end_handler (graph& /*G*/) { };

    /**
     * @brief Called after the %node @a n was taken out of the queue.
     * 
     * @param G %graph for which BFS was invoked.
     * @param n %node taken out of the queue.
     */
    virtual void popped_node_handler (graph& /*G*/, node& /*n*/) { };

    /**
     * @brief Called when finished with the %node @a n.

     * A %node is finished after all its neighbors have been
     * visited.
     *
     * @param G %graph for which BFS was invoked.
     * @param n finished %node.
     */
    virtual void finished_node_handler (graph& /*G*/, node& /*n*/) { };

    /**
     * @brief Called when an unused %node @a n was discovered. 
     * 
     * This means that the actual %node's @a f neighbor @a n was
     * not previously discovered.
     * 
     * @param G %graph for which BFS was invoked.
     * @param n unused %node.
     * @param f actual %node.
     */
    virtual void unused_node_handler (graph& /*G*/, node& /*n*/, node& /*f*/) { };

    /**
     * @brief Called when an used %node @a n was found. 
     * 
     * This means that the actual %node's (@a f) neighbor @a n
     * has already been discovered.
     * 
     * @param G %graph for which BFS was invoked.
     * @param n used %node.
     * @param f actual %node.
     */
    virtual void used_node_handler (graph& /*G*/, node& /*n*/, node& /*f*/) { };

    /**
     * @brief Called when BFS is started with start-%node
     * @a n. 

     * This is particularly useful when BFS was invoked with the
     * @c scan_whole_graph option.
     *
     * @param G %graph for which BFS was invoked.
     * @param n start-%node.
     * @sa bfs::scan_whole_graph
     */
    virtual void new_start_handler (graph& /*G*/, node& /*n*/) { };

private:

    void bfs_sub (graph&, const node&, edge_map<int>*);

protected:

    //-----------------------------------------------------------------------
    //   Data
    //-----------------------------------------------------------------------
    
    /**
     * @brief BFS number that will be assigned next.
     */
    int act_bfs_num;

    /**
     * @brief queue used in BFS.
     */
	std::deque<node> qu;

    /**
     * @brief List of nodes in BFS-order
     * 
     * @sa bfs::begin, bfs::end
     */
	nodes_t bfs_order;

    /**
     * @brief List of all edges of the BFS-tree
     * 
     * @sa bfs::tree_edges_begin, bfs::tree_edges_end
     */
	edges_t tree;

    /**
     * @brief Stores BFS-number of nodes. 
     */
    node_map<int> bfs_number;

    /**
     * @brief Number of nodes reached so far.
     */
    int reached_nodes;
    
    /**
     * @brief List of all roots of the BFS-tree
     * 
     * @sa bfs::roots_begin, bfs::roots_end
     */
	std::list<bfs_iterator> roots;

    //-----------------------------------------------------------------------
    //   Optional
    //-----------------------------------------------------------------------

    /**
     * @brief Stores whether whole %graph will be scanned.
     * 
     * @sa bfs::scan_whole_graph
     */
    bool whole_graph;

    /**
     * @brief Stores start %node.
     * 
     * @sa bfs:start_node
     */
    node start;

    /**
     * @brief Stores level number of each %node (if enabled)
     * 
     * @sa bfs::calc_level
     */
    node_map<int>* level_number;

    /**
     * @brief List of non-tree edges (if enabled)
     * 
     * @sa bfs::store_non_tree_edges
     */
	edges_t* non_tree;

    /**
     * @brief Stores father of each %node (if enabled)
     * 
     * @sa bfs::store_preds
     */
    node_map<node>* preds;
};

__GTL_END_NAMESPACE

#endif // GTL_BFS_H

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
