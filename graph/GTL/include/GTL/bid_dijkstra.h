/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   bid_dijkstra.h
//
//==========================================================================
// $Id: bid_dijkstra.h,v 1.3 2003/03/24 15:58:54 raitner Exp $

#ifndef GTL_BID_DIJKSTRA_H
#define GTL_BID_DIJKSTRA_H

#include <GTL/GTL.h>
#include <GTL/graph.h>
#include <GTL/node_map.h>
#include <GTL/edge_map.h>
#include <GTL/algorithm.h>

__GTL_BEGIN_NAMESPACE

/**
 * $Date: 2003/03/24 15:58:54 $
 * $Revision: 1.3 $
 * 
 * @brief Dijkstra's Algorithm for computing a shortest path from a single
 * source to a single target.
 *
 * This class implements Dijkstra's algorithm in a bidirectional manner for
 * computing a shortest path from a single source to a single target in
 * \f$\mathcal{O}((|V| + |E|) log |V|)\f$ worst case.
 *
 * @sa dijkstra
 * @sa bellman_ford
 *
 * @author Christian Bachmaier chris@infosun.fmi.uni-passau.de
 */
class GTL_EXTERN bid_dijkstra : public algorithm
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
    bid_dijkstra();

    /**
     * @brief Destructor.
     *
     * @sa algorithm::~algorithm
     */
    virtual ~bid_dijkstra();

    /**
     * @brief Sets source and target %node.
     *
     * Must be executed every time before check and run of this %algorithm.
     * 
     * @param s source %node
     * @param t target %node
     */
    void source_target(const node& s, const node& t);

    /**
     * @brief Sets weights of the edges. 
     * 
     * This method @b must be called before check and run. 
     *
     * @param weight weights of the %edges
     */
    void weights(const edge_map<double>& weight);
    
    /**
     * @brief Enables or disables the storing of the shortest path. 
     * 
     * If enabled for every %node and edge on the shortest path from source
     * to target will be stored.
     *
     * @param set true if path should be stored
     *
     * @sa dijkstra::predecessor_node
     * @sa dijkstra::predecessor_edge
     */
    void store_path(bool set);

    /**
     * @brief Checks whether the preconditions for bidirectional Dijkstra are
     * satisfied.
     * 
     * The Precondition are that the weights of the edges have been set and
     * that the graph has at least one %node. Additionally all %edge weights
     * must be \f$\ge 0\f$ and and source and target %nodes must be found in
     * @p G.
     *
     * @param G graph
     *
     * @retval algorithm::GTL_OK if %algorithm can be applied 
     * @retval algorithm::GTL_ERROR otherwise
     *
     * @sa dijkstra::source
     * @sa dijkstra::weigths
     * @sa algorithm::check
     */
    virtual int check(graph& G);
	    
    /**
     * @brief Runs shortest path algorithm on @p G.
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
     * @brief Returns whether the storing of the shortest path is enabled.
     * 
     * @return @c true iff the storing of path is enabled.
     *
     * @sa dijkstra::predecessor
     */
    bool store_path() const;

    /**
     * @brief Returns whether target is reachable from source.
     *
     * @return @c true iff target was reached from source
     */    
    bool reached() const;

    /**
     * @brief Returns the distance from source %node to target %node.
     *
     * @return distance if target is bid_dijkstra::reached, <code>-1.0</code>
     * else
     */
    double distance() const;

    /**
     * @brief Returns an iterator to the beginning (to the source %node) of
     * the shortest %node path to target %node.
     *
     * @return beginning %node iterator of the shortest path
     *
     * @sa bid_dijkstra::store_path
     *
     * @note The method requires that path calculation option was
     * enabled during last run.
     */
    shortest_path_node_iterator shortest_path_nodes_begin();

    /**
     * @brief Returns an iterator one after the end (one after target
     * %node) of the shortest %node path to target %node.
     *
     * @return shortest path end %node iterator
     *
     * @sa bid_dijkstra::store_path
     *
     * @note The method requires that path calculation option was
     * enabled during last run.
     */
    shortest_path_node_iterator shortest_path_nodes_end();

    /**
     * @brief Returns an iterator to the beginning %edge of the shortest
     * %edge path to target %node.
     *
     * @sa bid_dijkstra::store_path
     *
     * @return beginning %edge iterator of the shortest path
     *
     * @note The method requires that path calculation option was
     * enabled during last run.
     */
    shortest_path_edge_iterator shortest_path_edges_begin();

    /**
     * @brief Returns an iterator one after the end of a shortest %edge path
     * to target %node.
     *
     * @sa bid_dijkstra::store_path
     *
     * @return shortest path end %edge iterator
     *
     * @note The method requires that predecessor calculation option was
     * enabled during last run.
     */
    shortest_path_edge_iterator shortest_path_edges_end();

    /**
     * @brief Resets Dijkstra's bidirectional algorithm.
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
     * @sa bid_dijkstra::source.
     */
    node s;

    /**
     * @internal
     * Stores target.
     * 
     * @sa bid_dijkstra::source.
     */
    node t;

    /**
     * @internal
     * Indicates whether weights were set.
     * 
     * @sa bid_dijkstra::weights.
     */
    bool weights_set;

    /**
     * @internal
     * Indicates whether predecessors should be computed.
     * 
     * @sa bid_dijkstra::store_preds.
     */
    bool path_set;

    /**
     * @internal
     * Stores the weights of the %edges.
     * 
     * @sa bid_dijkstra::weights.
     */
    edge_map<double> weight;

    /**
     * @internal
     * Stores distance between @s and @t.
     * (default: -1.0)
     */
    double dist;

    /**
     * @internal
     * Stores if @a t can be reached from @s.
     * (default: false)
     */
    bool reached_t;

    /**
     * @internal
     * Stores predecessor of each %node in shortest path.
     * (default: edge() (if enabled))
     */
    node_map<edge> pred;

    /**
     * @internal
     * Stores successor of each %node in shortest path tree.
     * (default: edge() (if enabled))
     */
    node_map<edge> succ;

    /**
     * @internal
     * Indicates the current %node status.
     * (default: black)
     */
    node_map<int> source_mark;

    /**
     * @internal
     * Indicates the current %node status.
     * (default: black)
     */
    node_map<int> target_mark;

    /**
     * @internal
     * Distance from source @a s.
     * (default: -1.0)
     */
    node_map<double> source_dist;

    /**
     * @internal
     * Distance to target @a t.
     * (default: -1.0)
     */
    node_map<double> target_dist;

    /**
     * @internal
     * Stores for target %node @a t a list of nodes on the shortest path
     * from source @a s to it.
     * (default: empty)
     *
     * @sa dijkstra::shortest_path_nodes_begin
     * @sa dijkstra::shortest_path_nodes_end
     */
	nodes_t shortest_path_node_list;

    /**
     * @internal
     * Stores for target %node @a t a list of edges on the shortest path
     * from source @a s to it.
     * (default: empty)
     *
     * @sa dijkstra::shortest_path_edges_begin
     * @sa dijkstra::shortest_path_edges_end
     */
	edges_t shortest_path_edge_list;

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
     * Fills ordered lists @a shortest_path_node_list and @a
     * shortest_path_edge_list with nodes respective edges of shortest path
     * from @a s to @a t. Calculates distance.
     *
     * @param n first white node of the two directions
     */
    void fill_node_edge_lists(const node& n);
};

__GTL_END_NAMESPACE

#endif // GTL_BID_DIJKSTRA_H

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
