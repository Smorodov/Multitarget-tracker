/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   biconnectivity.h
//
//==========================================================================
// $Id: biconnectivity.h,v 1.18 2003/03/26 13:37:14 raitner Exp $

#ifndef GTL_BICONNECTIVITY_H
#define GTL_BICONNECTIVITY_H

#include <GTL/GTL.h>
#include <GTL/dfs.h>

#include <list>
#include <stack>

__GTL_BEGIN_NAMESPACE

/**
 * $Date: 2003/03/26 13:37:14 $
 * $Revision: 1.18 $
 * 
 * @brief Biconnectivity-test and low-numbers.
 *
 * Obviously there is a close relationship between DFS and the testing of
 * biconnectivity. Thus this test takes advantage of the possibility to 
 * add pieces of code to the DFS-class in order to calculate the
 * low-numbers. 
 * 
 * As default no biconnected components will be stored and no edges 
 * will be added to make the graph biconnected. The test will run on the
 * whole graph, even if it is not connected. 
 */

class GTL_EXTERN biconnectivity : public dfs 
{
public:
    /**
     * @brief Creates biconnectivity algorithm object.
     * 
     * @see dfs::dfs
     */
    biconnectivity ();

    /**
     * @brief Destroys biconnectivity algorithm object.
     *
     * @see dfs::~dfs
     */
    virtual ~biconnectivity () {}

    /**
     * @brief Checks whether the algorithm can be applied.
     * 
     * Necessary preconditions:
     *   - G is undirected.
     *   - storing of predecessors is enabled.
     *   - DFS may be applied 
     * 
     * @param G graph.
     * @return algorithm::GTL_OK if binconnectivity-test can
     * be applied to @a G.
     * @sa dfs::scan_whole_graph, dfs::store_preds
     */
    virtual int check (graph& G);

    virtual void reset ();

    /**
     * @brief low-number. 
     *
     * @param n node.
     * @return low-number of n.
     */
    int low_number (const node& n) const 
	{return low_num[n];}

    /**
     * @brief Biconnectivity-test. 
     * 
     * @return true iff graph is biconnected.
     */
    bool is_biconnected () const 
	{return num_of_components == 1;}

    /**
     * @brief Returns whether the storing of components is enabled.
     * 
     * @return true iff storing of components is enabled.
     * @sa biconnectivity::components_begin, biconnectivity::components_end
     */
    bool store_components () const
	{ return store_comp; }

    /**
     * @brief Enables or disables the storing of biconnected components.
     *
     * If this feature is enabled, the whole graph will be scanned
     * in order to get all the biconnected components even if the graph
     * isn't connected. By default this feature is disabled.
     * 
     * @param set if true each biconnected component will be stored.
     * @sa biconnectivity::components_begin, biconnectivity::components_end
     */
    void store_components (bool set) 
	{ store_comp  = set; if (set) scan_whole_graph (set); }
    
    /**
     * @brief If enabled edges will be added to the graph in order to make it 
     * biconnected, if cutpoints are discovered.
     * 
     * The list of added edges can be accessed via additional_begin and
     * additional_end.
     *
     * @param set if true additional edges will we inserted
     *    to make the graph biconnected.
     * @sa biconnectivity::additional_begin, biconnectivity::additional_end
     */
    void make_biconnected (bool set) 
	{ add_edges = set; if (set) scan_whole_graph (set); }
    
    /**
     * @brief Returns whether addition of edges neccessary to make graph
     * biconnected is enabled. 
     * 
     * @return true iff addition edges is enabled.
     * @sa biconnectivity::additional_begin, biconnectivity::additional_end
     */
    bool make_biconnected () const 
	{ return add_edges; }
    
    /**
     * @brief Begin of edges added to make graph biconnected.
     * 
     * @return begin of additional edges
     * @sa biconnectivity::make_biconnected
     */
	edges_t::iterator additional_begin()
	{ return additional.begin (); }

    /**
     * @brief End of edges added to make graph biconnected
     * 
     * @return end of additional edges
     * @sa biconnectivity::make_biconnected
     */
	edges_t::iterator additional_end()
	{ return additional.end (); }
    
    /**
     * @internal
     */
	typedef nodes_t::iterator cutpoint_iterator;

    /**
     * @brief Start iteration over all cutpoints found.
     *
     * A cutpoints is a node whose removal will disconnect the graph,
     * thus a graph with no cutpoints is biconnected and vice versa.
     * 
     * @return iterator to first cutpoint.
     * @sa biconnectivity::cut_points_end
     */
    cutpoint_iterator cut_points_begin () 
	{ return cut_points.begin(); }

    /**
     * @brief End of iteration over all cutpoints.
     * 
     * @return one-past-the-end iterator.
     * @sa biconnectivity::cut_points_begin
     */
    cutpoint_iterator cut_points_end () 
	{ return cut_points.end(); }


    /**
     * @internal
     */
	typedef std::list<std::pair<nodes_t, edges_t> >::iterator component_iterator;

    /**
     * @brief Start iteration over all biconnected components (if enabled during
     * last call to run).
     *
     * Components are represented as a pair consisting of
     * a list of nodes and a list of edges,
     * i.e. if it is of type component_iterator
     * then *it is of type 
     * pair&lt;list&lt;node&gt;,list&lt;edge&gt; &gt;. 
     *
     * @return iterator to first component
     * @sa biconnectivity::store_components
     */
    component_iterator components_begin ()
	{ return components.begin(); }


    /**
     * @brief End of iteration over all biconnected components.
     *
     * @return end of iteration over biconnected components
     * @sa biconnectivity::store_components
     */
    component_iterator components_end ()
	{ return components.end(); }

    /**
     * @brief Number von biconnected components detected during the last run.
     * 
     * @return number of biconnected components.
     */
    int number_of_components () const
	{return num_of_components; }

    //-----------------------------------------------------------------------
    //   Handler used to extend dfs to biconnectivity
    //-----------------------------------------------------------------------
    /**
     * @internal
     */
    virtual void init_handler (graph&);

    /**
     * @internal
     */
    virtual void entry_handler (graph&, node&, node&);

    /**
     * @internal
     */
    virtual void before_recursive_call_handler (graph&, edge&, node&);

    /**
     * @internal
     */
    virtual void after_recursive_call_handler (graph&, edge&, node&);

    /**
     * @internal
     */
    virtual void old_adj_node_handler (graph&, edge&, node&);

    /**
     * @internal
     */
    virtual void new_start_handler (graph&, node&);    

    /**
     * @internal
     */
    virtual void leave_handler (graph&, node&, node&);    

    /**
     * @internal
     */
    virtual void end_handler (graph&);    


protected:
    /**
     * @internal
     */
	edges_t self_loops;

    /**
     * @internal
     */
    node_map<component_iterator> in_component;

    /**
     * @internal
     */
    node_map<int> low_num;
    /**
     * @internal
     */
    int num_of_components;
    /**
     * @internal
     */
    bool store_comp;
    /**
     * @internal
     */
    bool add_edges;
    /**
     * @internal
     */
    node last;
    /**
     * @internal
     */
	std::stack<node> node_stack;
    /**
     * @internal
     */
	std::stack<edge> edge_stack;
    /**
     * @internal
     */
	std::list<std::pair<nodes_t, edges_t> > components;
    /**
     * @internal
     */
	nodes_t cut_points;
    /**
     * @internal
     */
    node_map<int> cut_count;
    /**
     * @internal
     */
	edges_t additional;
    /**
     * @internal
     */
    node_map<node> first_child;
};

__GTL_END_NAMESPACE

#endif // GTL_BICONNECTIVITY_H

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
