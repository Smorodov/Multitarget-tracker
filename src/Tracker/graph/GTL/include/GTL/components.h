/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   components.h
//
//==========================================================================
// $Id: components.h,v 1.5 2003/04/03 11:44:42 raitner Exp $

#ifndef GTL_COMPONENTS_H
#define GTL_COMPONENTS_H

#include <GTL/GTL.h>
#include <GTL/dfs.h>

#include <list>

__GTL_BEGIN_NAMESPACE
/**
 * @brief Connected components algorithm
 */
class GTL_EXTERN components : public dfs 
{
public:
    /**
     * @brief Creates connected components algorithm object.
     *
     * @sa dfs::dfs
     */
    components ();

    /**
     * @brief Destroys connected components algorithm object.
     *
     * @sa dfs::~dfs
     */
    virtual ~components () {}

    /**
     * @brief Checks whether the connected components algorithm can be applied
     *
     * Necessary preconditions:
     * - G is undirected.
     * - scanning of whole graph is enabled.
     * - DFS may be applied 
     * 
     * @param G graph.
     * @return algorithm::GTL_OK if connected components can be computed for G.
     * @sa dfs::scan_whole_graph
     */
    virtual int check (graph& G);

    virtual void reset ();

    /**
     * @internal
     */
	typedef std::list<std::pair<nodes_t, edges_t> >::iterator component_iterator;

    /**
     * @brief Start iteration over all components (if enabled during
     * last call to run).

     * Components are represented as a pair consisting of
     * a list of nodes and a list of edges,
     * i.e. if @c it is of type @c component_iterator
     * then @c *it is of type
     * @c pair&lt;list&lt;node&gt;,list&lt;edge&gt;&nbsp;&gt;. 
     *
     * @return iterator to first component 
     */
    component_iterator components_begin ()
	{ return comp.begin(); }


    /**
     * @brief End of iteration over all components.
     *
     * @return end of iteration over biconnected components
     * @sa biconnectivity::store_components
     */
    component_iterator components_end ()
	{ return comp.end(); }

    /**
     * @brief Number of components detected during the last run.
     * 
     * @return number of components.
     */
    int number_of_components () const
	{return num_of_components; }

    //-----------------------------------------------------------------------
    //   Handler used to extend dfs to biconnectivity
    //-----------------------------------------------------------------------
    /**
     * @internal
     */
    virtual void before_recursive_call_handler (graph&, edge&, node&);

    /**
     * @internal
     */
    virtual void old_adj_node_handler (graph&, edge&, node&);

    /**
     * @internal
     */
    virtual void new_start_handler (graph&, node&);    


protected:

    /**
     * @internal
     */
    int num_of_components;
    /**
     * @internal
     */
	std::list<std::pair<nodes_t, edges_t> > comp;
    /**
     * @internal
     */
    component_iterator li;
};

__GTL_END_NAMESPACE

#endif // GTL_BICONNECTIVITY_H

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
