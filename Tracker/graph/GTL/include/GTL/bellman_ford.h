/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   bellman_ford.h
//
//==========================================================================
// $Id: bellman_ford.h,v 1.5 2003/03/24 15:58:54 raitner Exp $ 

#ifndef GTL_BELLMAN_FORD_H
#define GTL_BELLMAN_FORD_H

#include <GTL/GTL.h>
#include <GTL/algorithm.h>
#include <GTL/node_map.h>
#include <GTL/edge_map.h>

__GTL_BEGIN_NAMESPACE


/**
 * $Date: 2003/03/24 15:58:54 $
 * $Revision: 1.5 $
 *
 * @brief Bellman Ford %algorithm.
 *
 * Implementation of the single source shortest path due to
 * Bellman and Ford. Unlike Dijkstra's SSSP %algorithm this one
 * allows negative edge weights, as long as there are no cycles
 * with negative weight. If there are negative cycles this
 * implementation finds them.
 */ 

class GTL_EXTERN bellman_ford : public algorithm 
{
public:

    /**
     * @brief Constructor. 
     */
    bellman_ford();

    /**
     * @brief Destructor.
     */
    virtual ~bellman_ford();

    /**
     * @brief Checks whether the preconditions for Bellman Ford
     * are satisfied.
     * 
     * The Precondition are that the weights of the edges
     * have been set and that the graph has at least one node.
     *
     * @param G graph.
     * @retval algorithm::GTL_OK if %algorithm can be applied
     * @retval algorithm::GTL_ERROR otherwise.
     */
    int check (graph& G);

    int run (graph& G);

    /**
     * @brief Resets the algorithm. 
     * 
     * The weights are not reset. You can apply this algorithms
     * twice without setting the weights for the second call.
     */
    void reset ();

    /**
     * @brief Sets source. 
     * 
     * The default source is the invalid %node (node::node()),
     * in this case an arbitrary %node is chosen and stored when
     * this algorithm is run.
     *
     * @param n source.
     */
    void source (const node& n) {s = n;}    

    /**
     * @brief Returns source.
     *
     * @return source.
     */
    node source () const {return s;}

    /**
     * @brief Sets weights of the edges. 
     * 
     * This method @b must be called before run. 
     *
     * @param w weights of the edges.
     */
    void weights (const edge_map<double>& weight) {w = weight; vars_set = true; }
    
    /**
     * @brief Enables or disables the storing of predecessors. 
     * 
     * If enabled for every %node the predecessor on the shortest
     * path from will be stored.
     *
     * @param set if true predecessors will be stored.
     * @sa bellman_ford::predecessor_node,
     * bellman_ford::predecessor_edge
     */
    void store_preds (bool set);

    /**
     * @brief Returns whether the storing of predecessors is enabled.
     * 
     * @retval true iff the storing of predecessors is enabled.  
     * 
     * @sa bellman_ford::predecessor_node,
     * bellman_ford::predecessor_edge
     */
    bool store_preds () const {return preds != 0;}

    /**
     * @brief Returns whether is reachable from source.
     * 
     * @param n node
     */    
    bool reached (const node& n) const {return !inf[n];}

    /**
     * @brief Returns the distance from source to @a n
     * 
     * @param n node
     */
    double distance (const node& n) const {return d[n];}

    /**
     * @brief edge to predecessor of %node @a n on the shortest
     * path from source
     * 
     * If @a n is a root or wasn't reached the return value is
     * the invalid %edge edge::edge().
     * 
     * @em Please @em note that this requires that this option
     * was enabled during last run.
     *
     * @param n node.
     * @return predecessor of @a n.
     * @sa bellman_ford::store_preds
     */
    edge predecessor_edge (const node& n) const
	{assert (preds); return (*preds)[n];}

    /**
     * @brief predecessor of %node @a n on the shortest
     * path from source
     * 
     * If @a n is a root or wasn't reached the return value is
     * the invalid %node node::node().
     * 
     * @em Please @em note that this requires that this option
     * was enabled during last run.
     *
     * @param n node.
     * @return predecessor of @a n.
     * @sa bellman_ford::store_preds
     */
    node predecessor_node (const node& n) const
	{edge e = predecessor_edge(n); return e == edge() ? node() : e.opposite(n); }

    /**
     * @brief Returns whether there is a cycle with negative
     * weight.
     */
    bool negative_cycle() const
	{return cycle;}

private:

    
    /** 
     * @brief Main method for Bellman Ford
     * 
     * @param e edge to be relaxed
     */    
    void relax (const edge& e, bool dir);

    /**
     * @brief Stores source.
     * 
     * @sa bellman_ford::source.
     */
    node s;

    /**
     * @brief Stores the weights of the edges.
     * 
     * @sa bellman_ford::weights.
     */
    edge_map<double> w;
    
    /**
     * @brief Indicates whether weights were set.
     * 
     * @sa bellman_ford::weights.
     */
    bool vars_set; 

    /**
     * @brief distance from source s.
     * 
     * @sa bellman_ford::distance.
     */
    node_map<double> d; 

    /**
     * @brief Indicates whether the node has distance infinity
     * 
     * @sa bellman_ford::distance.
     */
    node_map<bool> inf;

    /**
     * @brief Stores father of each %node (if enabled)
     * 
     * @sa bellman_ford::store_preds
     */
    node_map<edge>* preds;

    /**
     * @brief Indicates whether there is a cycle with negative
     * weight
     * 
     * @sa bellman_ford::negative_cycle.
     */
    bool cycle; 
};

__GTL_END_NAMESPACE

#endif // GTL_BELLMAN_FORD_H
