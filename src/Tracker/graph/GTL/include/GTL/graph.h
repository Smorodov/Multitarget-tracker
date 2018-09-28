/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   graph.h
//
//==========================================================================
// $Id: graph.h,v 1.43 2002/11/06 08:49:35 raitner Exp $

#ifndef GTL_GRAPH_H
#define GTL_GRAPH_H

#include <GTL/GTL.h>
#include <GTL/node.h>
#include <GTL/edge.h>
#include <GTL/edge_map.h>
#include <GTL/node_map.h>
#include <GTL/gml_parser.h>

#include <iostream>
#include <string>

__GTL_BEGIN_NAMESPACE

/**
 * $Date: 2002/11/06 08:49:35 $
 * $Revision: 1.43 $
 * 
 * @brief A directed or undirected graph.
 * 
 * A graph G=(V,E) consists of a set of nodes
 *  V and a set of edges  E , where every
 * edge can be viewed as a (ordered) pair of nodes  (u,v) 
 * connecting source  u  with target  v .
 * Obviously this implies a direction on the edges, which is why we 
 * call these graphs directed (this is the default). A graph can be made 
 * undirected by just ignoring the (implicit) direction. 
 *
 * @see     node
 * @see     edge
 */

class GTL_EXTERN graph
{
public:
    //================================================== Con-/Destructors

    /**
     * Generates an empty graph, i.e. without any nodes and any edges.
     */
    graph();

    /**
     * Copy constructor. <em>Please note:</em> This will generate an 
     * isomorpic copy of <code>G</code>. Although this graph will look 
     * like <code>G</code> it is not <em>physically</em> the same. 
     * Especially it consists of nodes and edges, which of course have 
     * counterparts in <code>G</code>, but are different. This means 
     * that the nodes (edges) in the copy have undefined behaviour if 
     * used within a @ref node_map (@ref edge_map ) of the original graph.
     *
     * @param <code>G</code> graph
     */
    graph (const graph& G);
    
    /**
     * Makes new graph isomorphic to the subgraph induced by <code>nodes</code>.
     * The same restriction as for the ordinary copy constructor applies to
     * this one. 
     *
     * @param <code>G</code> graph
     * @param <code>nodes</code> nodes of <code>G</code>, which form 
     *        the induced subgraph this graph will be isomorphic to.
     */
	graph(const graph& G, const nodes_t& nodes);
    
    /**
     * Makes new graph isomorphic to the subgraph induced by the nodes
     * in the range from <code>it</code> to <code>end</code>
     * The same restriction as for the ordinary copy constructor applies to
     * this one. 
     *
     * @param <code>G</code> graph
     * @param <code>it</code> beginning of nodes
     * @param <code>end</code> end of nodes     
     */
    graph (const graph& G, 
		nodes_t::const_iterator it,
		nodes_t::const_iterator end);
    
    /**
     * Destructor. Deletes all nodes and edges.
     */
    virtual ~graph();
    
    //================================================== Directed/Undirected

    /**
     * Makes graph directed. 
     */
    void make_directed();

    /**
     * Makes graph undirected.
     */
    void make_undirected();

    //================================================== Tests / Information
    
    /**
     * Test whether the graph is directed.
     *
     * @return true iff the graph is directed.
     */
    bool is_directed() const;

    /**
     * Test whether the graph is undirected.
     *
     * @return true iff the graph is undirected
     */
    bool is_undirected() const;

    /**
     * Checks if for all edges <var>(v, w)</var> the reverse edge
     * <var>(w,v)</var> is present, too. Additionally the reverse of some
     * edge <code>e</code> will be stored as <code>rev[e]</code>. If there
     * is no reverse edge of <code>e</code> <code>rev[e]</code> will be the
     * invalid edge <code>edge()</code>.
     * 
     * @param <code>rev</code> map associating every edge with its
     *   reverse edge.
     * @return true iff every edge has a reverse edge.  
     */
    bool is_bidirected(edge_map<edge>& rev) const;

    /**
     * Test whether the graph is connected
     *
     * @return true iff the graph is connected
     * @see dfs
     * @see bfs 
     */
    bool is_connected() const;

    /**
     * Test whether the graph is acyclic
     *
     * @return true iff the graph contains no cycles
     * @see topsort
     */
    bool is_acyclic() const;

    /**
     * Returns the number of nodes in the graph.
     *
     * @return number of nodes
     */
    int number_of_nodes() const;

    /**
     * Returns the number of (visible) edges in the graph
     *
     * @return number of edges
     */
    int number_of_edges() const;

    /**
     * Returns a center of the graph which is defined as a node with
     * maximum excentricity.
     *
     * @return one node of the graph center
     */
    node center() const;

    //================================================== Creation
    
    /**
     * Adds a new node.
     *
     * @return new node.  
     */
    virtual node new_node();
    
    /**
     * Adds new edge from <code>s</code> to
     * <code>t</code>. 
     * 
     * <p>
     * <em>Precondition:</em> <code>s,t</code> are valid nodes in this graph.
     *
     * @param <code>s</code> source of new edge
     * @param <code>t</code> target of new edge 
     * @return new edge. 
     */
    virtual edge new_edge(node s, node t);

    /**
     * @internal
     */
	virtual edge new_edge(const nodes_t &sources, const nodes_t &targets);

    //================================================== Deletion

    /**
     * Deletes node <code>n</code>, and thus all edges incident with
     * <code>n</code>.
     *
     * <p>
     * <em>Precondition:</em> <code>n</code> is a valid <em>visible</em> node 
     *    in this graph
     *
     * @param <code>n</code> visible node to be deleted 
     */
    void del_node(node n);

    /**
     * @deprecated
     * Deletes all visible nodes, i.e. the hidden ones stay.
     */
    void del_all_nodes(); 

    /**
     * Deletes edge <code>e</code>.
     *
     * <p>
     * <em>Precondition:</em> <code>e</code> is a valid <em>visible</em> edge 
     *    in this graph.
     *
     * @param <code>e</code> edge to be deleted
     */
    void del_edge(edge e);

    /**
     * @deprecated
     * Deletes all visible edges, i.e. the hidden ones stay.
     */    
    void del_all_edges(); 

    /**
     * Deletes all nodes and edges, even the hidden ones
     */
    void clear();

    //================================================== Iterators

    /**
     * @internal
     */
	typedef nodes_t::const_iterator node_iterator;
    /**
     * @internal
     */
	typedef edges_t::const_iterator edge_iterator;
    
    /**
     * Iterate through all nodes in the graph.
     *
     * @return start for iteration through all nodes in the graph.
     */
    node_iterator nodes_begin() const;
    
    /**
     * Iterate through all nodes in the graph.
     *
     * @return end for iteration through all nodes in the graph.
     */
    node_iterator nodes_end() const;
    
    /**
     * Iterate through all edges in the graph.
     *
     * @return start for iteration through all edges in the graph.
     */
    edge_iterator edges_begin() const;
    
    /**
     * Iterate through all edges in the graph.
     *
     * @return end for iteration through all edges in the graph.
     */
    edge_iterator edges_end() const;

    //================================================== get nodes/edges

    /**
     * @deprecated
     * @return a list of all nodes of the graph
     */
	nodes_t all_nodes() const;

    /**
     * @deprecated
     * @return a list of all edges of the graph
     */
	edges_t all_edges() const;
 
    /**
     * @deprecated
     */
    node choose_node () const;
    
    //================================================== Hide / Restore

    /**
     * Hides an edge. 
     *
     * <p>
     * <em>Precondition:</em> <code>e</code> is a valid edge in this graph
     *
     * @param <code>e</code> edge to be hidden
     */
    void hide_edge (edge e);
    
    /**
     * Restores a hidden edge
     * 
     * <p>
     * <em>Precondition:</em> <code>e</code> is a valid edge in this graph
     *
     * @param <code>e</code> hidden edge 
     */
    void restore_edge (edge e);

    /**
     * Hides a node. <em>Please note:</em> all the edges incident with 
     * <code>n</code> will be hidden, too. All these edges are returned 
     * in a list.
     *
     * <p>
     * <em>Precondition:</em> <code>n</code> is a valid node in this graph
     *
     * @param <code>e</code> node to be hidden
     * @return list of implicitly hidden, incident edges
     */
	edges_t hide_node(node n);

    /**
     * Restores a hidden node. This only restores the node itself. It
     * doesn't restore the incident edges, i.e. you will have to restore
     * all the edges you get returned when calling @ref graph#hide_node
     * yourself.  
     * 
     * <p>
     * <em>Precondition:</em> <code>n</code> is a valid node in this graph
     * @param <code>n</code> hidden node
     */
    void restore_node (node n);

    /**
     * Hides all nodes <em>not</em> contained in <code>subgraph_nodes</code>, i.e.
     * (the visible part of) the graph is the induced subgraph with
     * respect to the nodes in <code>subgraph_nodes</code>. It is allowed
     * to apply this function recursively, i.e. one may call
     * <code>induced_subgraph</code> on a graph that is already a induced
     * subgraph.
     * 
     * @param <code>subgraph_nodes</code> nodes of subgraph.
     * @see graph#restore_graph
     */
	void induced_subgraph(nodes_t& subgraph_nodes);

    /**
     * Restores all hidden nodes and edges
     * This means that, although the nodes
     * and edges got hidden at different times, they will be restored all
     * together.
     * 
     * @see graph#induced_subgraph
     * @see graph#hide_edge
     * @see graph#hide_node
     */
    void restore_graph ();

    //================================================== Others

    /**
     * @deprecated
     * inserts for all edges of the graph a reverse edge
     * NOTE: this functions does NOT care about existing reverse edges
     */
	edges_t insert_reverse_edges();
    
    //================================================== I/O

    /**
     * Load graph from a file in GML-format. The optional
     * parameter <code>preserve_ids</code> controls whether to
     * give the nodes the same ids as in the GML file. You can enable this 
     * for debugging but you should disable it for final releases since 
     * it may make <code>node_map</code> unecessarily large.  
     * 
     * @param <code>filename</code> file in GML-format.  
     * @param <code>preserve_ids</code> if true all the nodes
     * will get the same id as in the GML file. If false (default) 
     * the nodes will be numbered consecutively beginning with 0. However 
     * the order of the nodes in the GML file will be preserved. 
     * @return detailed error description (hopefully GML_OK). For details 
     *   see @ref GML_error#err_num.
     */
    
	GML_error load(const std::string& filename, bool preserve_ids = false)
	{ return load (filename.c_str(), preserve_ids); }

    
    /**
     * Load graph from a file in GML-format. The optional
     * parameter <code>preserve_ids</code> controls whether to
     * give the nodes the same ids as in the GML file. You can enable this 
     * for debugging but you should disable it for final releases since 
     * it may make <code>node_map</code> unecessarily large.  
     *
     * @param <code>filename</code> file in GML-format.
     * @param <code>preserve_ids</code> if true all the nodes
     * will get the same id as in the GML file. If false (default) 
     * the nodes will be numbered consecutively beginning with 0. However 
     * the order of the nodes in the GML file will be preserved.
     * @return detailed error description (hopefully GML_OK). For details 
     *   see @ref GML_error#err_num.
     */
    
    GML_error load (const char* filename, bool preserve_ids = false);

    /**
     * Save graph to file <code>filename</code> in GML-format, i.e.
     * <code>graph [ node [ id # ] ... edge [ source # target #] ... ]</code>
     *
     * @param <code>filename</code>
     * @return 0 on error 1 otherwise
     */

    int save (const char* filename) const;

    /**
     * Saves graph to stream <code>file</code> in GML-format.
     *
     * @param <code>file</code> output stream defaults to cout.
     */
    
	void save(std::ostream* file = &std::cout) const;

    //================================================== Node handlers
    
    /**
     * Virtual function called before a new node is created;
     * can be redefined in a derived class for customization
     *
     * @see graph#new_node
     */
    virtual void pre_new_node_handler() {}

    /**
     * Virtual function called after a new node was created;
     * can be redefined in a derived class for customization
     *
     * @param <code>n</code> created node
     * @see graph#new_node
     */
    virtual void post_new_node_handler(node /*n*/) {}

    /**
     * Virtual function called before a node is deleted;
     * can be redefined in a derived class for customization
     *
     * @param <code>n</code> node deleted afterwards 
     * @see graph#del_node 
     */
    virtual void pre_del_node_handler(node /*n*/) {}     	

    /**
     * Virtual function called after a node was deleted;
     * can be redefined in a derived class for customization
     *
     * @see graph#del_node
     */
    virtual void post_del_node_handler() {}        	

    /**
     * Virtual function called before a node gets hidden; 
     * can be redefined in a derived class for customization
     * 
     * @param <code>n</code> node to be hidden
     * @see graph#hide_node
     */
    virtual void pre_hide_node_handler(node /*n*/) {}          

    /**
     * Virtual function called after a node got hidden;
     * can be redefined in a derived class for customization
     * 
     * @param <code>n</code> hidden node
     * @see graph#hide_node
     */
    virtual void post_hide_node_handler(node /*n*/) {}         

    /**
     * Virtual function called before a node is restored;
     * can be redefined in a derived class for customization
     * 
     * @param <code>n</code> node to be restored
     * @see graph#restore_node
     */
    virtual void pre_restore_node_handler(node /*n*/) {}       
 
    /**
     * Virtual function called after a node was restored;
     * can be redefined in a derived class for customization
     * 
     * @param <code>n</code> restored node
     * @see graph#restore_node
     */
   virtual void post_restore_node_handler(node /*n*/) {}

   //================================================== Edge handlers

    /**
     * Virtual function called before a new edge is inserted;
     * can be redefined in a derived class for customization
     *
     * @param <code>s</code> source of edge created afterwards
     * @param <code>t</code> target of edge created afterwards
     * @see graph#new_edge
     */
    virtual void pre_new_edge_handler(node /*s*/, node /*t*/) {}    

    /**
     * Virtual function called after a new edge was inserted;
     * can be redefined in a derived class for customization
     *
     * @param <code>e</code> created edge 
     * @see graph#new_edge
     */
    virtual void post_new_edge_handler(edge /*e*/) {}          

    /**
     * Virtual function called before a edge is deleted;
     * can be redefined in a derived class for customization
     * 
     * @param <code>e</code> edge to be deleted
     * @see graph#del_edge
     */
    virtual void pre_del_edge_handler(edge /*e*/) {}           

    /**
     * Virtual function called after a edge was deleted;
     * can be redefined in a derived class for customization
     * 
     * @param <code>s</code> source of edge deleted
     * @param <code>t</code> target of edge deleted
     * @see graph#del_edge
     */
    virtual void post_del_edge_handler(node, node) {}
    
    /**
     * Virtual function called before a edge gets hidden; 
     * can be redefined in a derived class for customization
     * 
     * @param <code>e</code> edge to be hidden
     * @see graph#hide_edge
     */
    virtual void pre_hide_edge_handler(edge /*e*/) {}          

    /**
     * Virtual function called after a edge got hidden;
     * can be redefined in a derived class for customization
     * 
     * @param <code>e</code> hidden edge
     * @see graph#hide_edge
     */
    virtual void post_hide_edge_handler(edge /*e*/) {}         

    /**
     * Virtual function called before a edge is restored;
     * can be redefined in a derived class for customization
     * 
     * @param <code>e</code> edge to be restored
     * @see graph#restore_edge
     */
    virtual void pre_restore_edge_handler(edge /*e*/) {}       
 
    /**
     * Virtual function called after a edge was restored;
     * can be redefined in a derived class for customization
     * 
     * @param <code>e</code> restored edge
     * @see graph#restore_edge
     */
   virtual void post_restore_edge_handler(edge /*e*/) {}

    //================================================== Global handlers
    
    /**
     * Virtual function called before performing clear;
     * can be redefined in a derived class for customization. 
     * <em>Please note:</em> Although nodes and edges are deleted 
     * during @ref graph#clear this is not achieved by calling 
     * @ref graph#del_node and @ref graph#del_edge, which is why 
     * the correspondig handler will not be called. 
     *
     * @see graph#clear
     */
    virtual void pre_clear_handler()  {}

    /**
     * Virtual function called after the graph was cleared;
     * can be redefined in a derived class for customization
     * <em>Please note:</em> Although nodes and edges are deleted 
     * during @ref graph#clear this is not achieved by calling 
     * @ref graph#del_node and @ref graph#del_edge, which is why 
     * the correspondig handler will not be called. 
     *
     * @see graph#clear
     */ 
   virtual void post_clear_handler() {} 

    /**
     * Virtual function called before performing make_directed
     * (only if graph was undirected)
     * can be redefined in a derived class for customization
     *
     * @see graph#make_directed
     */
    virtual void pre_make_directed_handler()  {}

    /**
     * Virtual function called after performing make_directed;
     * (only if graph was undirected)
     * can be redefined in a derived class for customization
     *
     * @see graph#make_directed
     */
    virtual void post_make_directed_handler()  {}

    /**
     * Virtual function called before performing make_undirected;
     * (only if graph was directed)
     * can be redefined in a derived class for customization
     *
     * @see graph#make_undirected
     */
    virtual void pre_make_undirected_handler()  {}

    /**
     * Virtual function called after performing make_undirected;
     * (only if graph was directed)
     * can be redefined in a derived class for customization
     *
     * @see graph#make_undirected
     */
    virtual void post_make_undirected_handler()  {}


    //================================================== I/O - Handler 

    /**
     * Called before writing the graph key to <code>os</code>. This can be
     * used to write top-level keys that should appear before the graph in
     * the file.
     * 
     * @param <code>os</code> output stream.
     * @see graph#save 
     */
	virtual void pre_graph_save_handler(std::ostream* /*os*/) const { };

    /**
     * Called before the closing bracket of the list belonging to the
     * graph key is written. This can be used to write information that
     * belong to the graph, and thus should appear within the list
     * associated with the graph key.
     *
     * @param <code>os</code> output stream.
     * @see graph#save 
     */
	virtual void save_graph_info_handler(std::ostream*) const { };
    
    /**
     * Called before the closing bracket of the list belonging to the key
     * of node <code>n</code> is written. This can be used to write
     * information belonging to the node <code>n</code> and thus should
     * appear within the list associated with this node.
     * 
     * @param <code>os</code> output stream.
     * @see graph#save 
     */
	virtual void save_node_info_handler(std::ostream*, node) const { };

    /**
     * Called before the closing bracket of the list belonging to the key
     * of edge <code>e</code> is written. This can be used to write
     * information belonging to the edge <code>e</code> and thus should
     * appear within the list associated with this edge.
     * 
     * @param <code>os</code> output stream.
     * @see graph#save
     */
	virtual void save_edge_info_handler(std::ostream*, edge) const { };

    /**
     * Called after writing the graph key to <code>os</code>. This can be
     * used to write top-level keys that should appear after the graph in
     * the file.
     * 
     * @param <code>os</code> output stream.
     * @see graph#save 
     */
	virtual void after_graph_save_handler(std::ostream*) const { };

    /**
     * Called after the graph is completely built. The topmost list
     * of key-value-pairs is passed to this handler. NB: This list 
     * also contains the graph key, which was used to build the graph. 
     * 
     * @param <code>list</code> pointer to the list of key-value pairs at
     *                          top level
     * @see graph#load 
     */
    virtual void top_level_key_handler (GML_pair* list);

    /**
     * Called after a node is created. The whole list of key-value-pairs 
     * belonging to this node is passed to this handler together with the 
     * node itself.
     *
     * @param <code>n</code> node parsed 
     * @param <code>list</code> pointer to the list of key-value-pairs of
     *                          this node.
     * @see graph#load 
     */
    virtual void load_node_info_handler (node n, GML_pair* list );
    
    /**
     * Called after an edge is created. The whole list of key-value-pairs 
     * belonging to this edge is passed to this handler together with the 
     * edge itself.
     *
     * @param <code>e</code> edge parsed 
     * @param <code>list</code> pointer to the list of key-value-pairs of
     *                          this edge.
     * @see graph#load 
     */
    virtual void load_edge_info_handler (edge e, GML_pair* list);

    /**
     * Called after the graph is completely built. The whole list for 
     * the graph key used to build this graph is passed to this handler.
     *
     * @param <code>list</code> pointer to the list of key-value-pairs of
     *                          the graph.
     * @see graph#load 
     */
    virtual void load_graph_info_handler (GML_pair* list);

private:

    //================================================== Flags
    
    mutable bool directed;

    //================================================== Visible Nodes/Edges
    
	nodes_t nodes;
	edges_t edges;
    int nodes_count, edges_count;

    //================================================== Hidden Nodes/Edges
    
	nodes_t hidden_nodes;
	edges_t hidden_edges;
    int hidden_nodes_count, hidden_edges_count; 

    //================================================== Node/edge numbering

    int new_node_id();
    int new_edge_id();

    //================================================== Copy 
    
    void copy (const graph& G, 
		nodes_t::const_iterator it,
		nodes_t::const_iterator end);

public: // needs to be public, because template friends are not possible
    /**
     * @internal
     */
    int number_of_ids(node) const;
    
    /**
     * @internal
     */
    int number_of_ids(edge) const;
    
private:
	std::list<int> free_node_ids;
	std::list<int> free_edge_ids;
    int free_node_ids_count, free_edge_ids_count;

    //================================================== utilities
    
	void del_list(nodes_t &);
	void del_list(edges_t &);

	GTL_EXTERN friend std::ostream& operator<< (std::ostream& os, const graph& G);
};

__GTL_END_NAMESPACE

//--------------------------------------------------------------------------
//   Iteration
//--------------------------------------------------------------------------

#define forall_nodes(v,g) GTL_FORALL(v,g,graph::node_iterator,nodes_)
#define forall_edges(v,g) GTL_FORALL(v,g,graph::edge_iterator,edges_)
    
#endif // GTL_GRAPH_H

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
