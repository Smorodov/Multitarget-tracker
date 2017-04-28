/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   dijkstra.h
//
//==========================================================================
// $Id: dijkstra.h,v 1.8 2003/02/25 09:18:19 chris Exp $

#ifndef GTL_DIJKSTRA_H
#define GTL_DIJKSTRA_H

#include <GTL/GTL.h>
#include <GTL/graph.h>
#include <GTL/node_map.h>
#include <GTL/edge_map.h>
#include <GTL/algorithm.h>

__GTL_BEGIN_NAMESPACE

/**
 * @brief Dijkstra's Algorithm for computing single source shortest path.
 *
 * This class implements Dijkstra's algorithm for computing single source
 * shortest path in @f$\mathcal{O}((|V| + |E|) log |V|)@f$ worst case.
 *
 * @sa bellman_ford
 *
 * @author Christian Bachmaier chris@infosun.fmi.uni-passau.de
 */
class GTL_EXTERN dijkstra : public algorithm
{
public:
    /**
     * @brief Iterator type for traversing %nodes on one shortest path.
     */
	typedef nodes_t::const_iterator shortest_path_node_iterator;

    /**
     * @brief Iterator type for traversing %edges on one shortest path.
     */
	typedef edges_t::const_iterator shortest_path_edge_iterator;

    /**
     * @internal
     */
    enum node_color {white, grey, black};

    /**
     * @brief Default constructor.
     *
     * Enables only the calculation of shortest paths.
     * 
     * @sa algorithm::algorithm
     */
    dijkstra();

    /**
     * @brief Destructor.
     *
     * @sa algorithm::~algorithm
     */
    virtual ~dijkstra();

    /**
     * @brief Sets source %node.
     * 
     * The default source is the invalid %node (node::node()),
     * in this case an arbitrary %node is chosen and stored when
     * this algorithm is run.
     *
     * @param n source node
     */
    void source(const node& n);

    /**
     * @brief Sets target %node. 
     * 
     * If a target is set with this method the %algorithm stops if a
     * shortest distance to @p n is found. Ohterwise shortest paths are
     * computed from source to any %node in the %graph.
     *
     * @param n target node
     */
    void target(const node& n);

    /**
     * @brief Sets weights of the edges. 
     * 
     * This method @b must be called before check run.
     *
     * @param weight weights of the %edges
     */
    void weights(const edge_map<double>& weight);
    
    /**
     * @brief Enables or disables the storing of predecessors. 
     * 
     * If enabled for every %node the predecessor on the shortest
     * path from will be stored.
     *
     * @param set @c true if predecessors should be stored
     *
     * @sa dijkstra::predecessor_node
     * @sa dijkstra::predecessor_edge
     */
    void store_preds(bool set);

    /**
     * @brief Checks whether the preconditions for Dijkstra are satisfied.
     * 
     * Necessary preconditions are:
     * - the weights of the edges are set
     * - the %graph @p G has at least one %node
     * - all %edge weights must be \f$\ge 0\f$
     * - the source %node and (if set) target %node must be found in @p G
     *
     * @param G graph
     *
     * @retval algorithm::GTL_OK if %algorithm can be applied 
     * @retval algorithm::GTL_ERROR otherwise
     *
     * @sa dijkstra::source
     * @sa dijkstra::weights
     * @sa algorithm::check
     */
    virtual int check(graph& G);
	    
    /**
     * @brief Runs shortest path %algorithm on @p G.
     *
     * This should return always algorithm::GTL_OK. The return value only
     * tracks errors that might occur.
     * Afterwards the result of the test can be accessed via access methods.
     *
     * @param G graph
     *
     * @retval algorithm::GTL_OK on success
     * @retval algorithm::GTL_ERROR otherwise
     *
     * @sa algorithm::run
     */
    int run(graph& G);

    /**
     * @brief Returns source %node.
     *
     * @return source %node
     */
    node source() const;

    /**
     * @brief Returns target %node if set, <code>node::node()</code> else.
     *
     * @return target %node
     */
    node target() const;

    /**
     * @brief Returns whether the storing of predecessors is enabled.
     * 
     * @return @c true iff the storing of predecessors is enabled
     *
     * @sa dijkstra::predecessor
     */
    bool store_preds() const;

    /**
     * @brief Returns whether @p n is reachable from source %node.
     * 
     * @param n node
     *
     * @return @c true iff @p n was reached from source
     */    
    bool reached(const node& n) const;

    /**
     * @brief Returns the distance from source %node to %node @p n.
     * 
     * @param n node
     *
     * @return distance if @p n is dijkstra::reached, <code>-1.0</code> else
     */
    double distance(const node& n) const;

    /**
     * @brief Predecessor %node of %node @p n on the shortest path from the
     * source %node.
     * 
     * If @p n is a root or wasn't reached the return value is
     * the invalid %node node::node().
     * 
     * @param n node
     *
     * @return predecessor %node of @p n
     *
     * @sa dijkstra::store_preds
     * @sa dijkstra::predecessor_edge
     *
     * @note The method requires that predecessor calculation option was
     * enabled during last run.
     */
    node predecessor_node(const node& n) const;

    /**
     * @brief Predecessor %edge of %node @p n on the shortest path from the
     * source %node.
     * 
     * If @p n is a root or wasn't reached the return value is
     * the invalid %edge edge::edge().
     *
     * @param n node
     *
     * @return predecessor %edge of @p n
     *
     * @sa dijkstra::store_preds
     * @sa dijkstra::predecessor_node
     *
     * @note The method requires that predecessor calculation option was
     * enabled during last run.
     */
    edge predecessor_edge(const node& n) const;

    /**
     * @brief Returns an iterator to the beginning (to the source %node) of
     * a shortest %node path to %node @p dest.
     *
     * @param dest target %node
     *
     * @return beginning %node iterator of a shortest path
     *
     * @note The method requires that predecessor calculation option was
     * enabled during last run. If this method is called on the shortest
     * path to @p dest for the first time (before
     * dijkstra::shortest_path_nodes_end) it needs
     * @f$\mathcal{O}(\mbox{length of this path})@f$ time.
     */
    shortest_path_node_iterator shortest_path_nodes_begin(const node& dest);

    /**
     * @brief Returns an iterator one after the end (one after
     * %node @p dest) of a shortest %node path to %node @p dest.
     *
     * @param dest target %node
     *
     * @return shortest path end %node iterator
     *
     * @note The method requires that predecessor calculation option was
     * enabled during last run. If this method is called on the shortest
     * path to @p dest for the first time (before
     * dijkstra::shortest_path_nodes_begin) it needs
     * @f$\mathcal{O}(\mbox{length of this path})@f$ time.
     */
    shortest_path_node_iterator shortest_path_nodes_end(const node& dest);

    /**
     * @brief Returns an iterator to the beginning %edge of a shortest %edge
     * path to %node @p dest.
     *
     * @param dest target %node
     *
     * @return beginning %edge iterator of a shortest path
     *
     * @note The method requires that predecessor calculation option was
     * enabled during last run. If this method is called on the shortest
     * path to @p dest for the first time (before
     * dijkstra::shortest_path_edges_end) it needs
     * @f$\mathcal{O}(\mbox{length of this path})@f$ time.
     */
    shortest_path_edge_iterator shortest_path_edges_begin(const node& dest);

    /**
     * @brief Returns an iterator one after the end of a shortest %edge path
     * to %node @p dest.
     *
     * @param dest target %node
     *
     * @return shortest path end %edge iterator
     *
     * @note The method requires that predecessor calculation option was
     * enabled during last run. If this method is called on the shortest
     * path to @p dest for the first time (before
     * dijkstra::shortest_path_edges_begin) it needs
     * @f$\mathcal{O}(\mbox{length of this path})@f$ time.
     */
    shortest_path_edge_iterator shortest_path_edges_end(const node& dest);

    /**
     * @brief Resets Dijkstra's algorithm.
     *
     * It prepares the algorithm to be applied again, possibly to another
     * graph.
     *
     * @note The weights are not reset. You can apply this algorithms
     *
     * @sa algorithm::reset
     */
    virtual void reset();
private:
    /**
     * @internal
     * Stores source.
     * 
     * @sa dijkstra::source.
     */
    node s;

    /**
     * @internal
     * Stores target.
     * 
     * @sa dijkstra::source.
     */
    node t;

    /**
     * @internal
     * Indicates whether weights were set.
     * 
     * @sa dijkstra::weights.
     */
    bool weights_set;

    /**
     * @internal
     * Indicates whether predecessors should be computed.
     * 
     * @sa dijkstra::store_preds.
     */
    bool preds_set;

    /**
     * @internal
     * Stores the weights of the %edges.
     * 
     * @sa dijkstra::weights.
     */
    edge_map<double> weight;

    /**
     * @internal
     * Stores father of each %node in shortest path tree (if enabled).
     * (default: edge() (if enabled))
     * 
     * @sa dijkstra::store_preds
     */
    node_map<edge> pred;

    /**
     * @internal
     * Indicates the current %node status.
     * (default: black)
     */
    node_map<int> mark;

    /**
     * @internal
     * Distance from source @a s.
     * (default: -1)
     * 
     * @sa dijkstra::distance.
     */
    node_map<double> dist;

    /**
     * @internal
     * Stores for every target %node a list of nodes on the shortest path
     * from source @a s to it. Filled on demand by methods creating
     * iterators.
     * (default: empty)
     *
     * @sa dijkstra::shortest_path_nodes_begin
     * @sa dijkstra::shortest_path_nodes_end
     */
	node_map<nodes_t> shortest_path_node_list;

    /**
     * @internal
     * Stores for every target node a list of edges on the shortest path
     * from source @a s to it. Filled on demand by methods creating
     * iterators.
     * (default: empty)
     *
     * @sa dijkstra::shortest_path_edges_begin
     * @sa dijkstra::shortest_path_edges_end
     */
	node_map<edges_t> shortest_path_edge_list;

    /**
     * @internal
     * Prepares the %algorithm to be applied once again.
     */
    void reset_algorithm();
    
    /**
     * @internal
     * Inits data structure.
     */
    void init(graph& G);

    /**
     * @internal
     * Fills ordered list <code>shortest_path_node_list[t]</code>
     * with nodes of shortest path from @a s to @p t.
     */
    void fill_node_list(const node& t);

    /**
     * @internal
     * Fills ordered list <code>shortest_path_edge_list[t]</code>
     * with edges of shortest path from @a s to @p t.
     */
    void fill_edge_list(const node& t);
};

__GTL_END_NAMESPACE

#endif // GTL_DIJKSTRA_H

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
