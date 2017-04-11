/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   node.h
//
//==========================================================================
// $Id: node.h,v 1.20 2003/11/27 13:36:56 raitner Exp $

#ifndef GTL_NODE_H
#define GTL_NODE_H

#include <GTL/GTL.h>
#include <GTL/edge.h>

#include <list>

__GTL_BEGIN_NAMESPACE

//--------------------------------------------------------------------------
//   For MSVC 5.0 node.h has to be included before {node,edge}_data.h.
//   So we only declare node_data here
//--------------------------------------------------------------------------

class node_data;

//--------------------------------------------------------------------------
// The first alternative is correct. The second one is a workaround
// for compilers that don't support namespaces and use the SGI STL
// (i.e. gcc/egcs)
//--------------------------------------------------------------------------

#ifdef __GTL_USE_NAMESPACES

class node;
typedef std::iterator<std::bidirectional_iterator_tag, edge> bi_iter_edge;
typedef std::iterator<std::bidirectional_iterator_tag, node> bi_iter_node;

#else

class node;
typedef bidirectional_iterator<edge,ptrdiff_t> bi_iter_edge;
typedef bidirectional_iterator<node,ptrdiff_t> bi_iter_node;

#endif // __GTL_USE_NAMESPACES

//--------------------------------------------------------------------------
//   nodes
//--------------------------------------------------------------------------

/**
 * @short A node in a graph
 */
class GTL_EXTERN node
{
public:
    /**
     * Default constructor. Creates an invalid node. 
     * The only way to obtain a valid node is through @ref
     * graph#new_node Example:
     * <pre>
     *   graph g;
     *   node n;
     *
     *   n = g.new_node();
     * </pre>
     *
     * @see graph#new_node
     */
    node();

    /**
     * Returns the degree of the node, i. e.
     * @ref node#outdeg + @ref node#indeg .
     */
    int degree() const;

    /**
     * Returns the out degree of the node, i. e. the number of outgoing edges.
     */
    int outdeg() const;

    /**
     * Returns the in degree of the node, i. e. the number of incoming edges.
     */
    int indeg() const;

    /**
     * @internal
     */
    int id() const;
    
    /**
     * Returns the node on the opposite side of <code>e</code>.
     * 
     * @param e an edge incident to the node
     */
    const node& opposite(edge e) const;
    
    /**
     * @internal
     */
    nodes_t opposites(edge) const;

    /**
     * Returns true iff node is hidden.
     *
     * @return true iff node is hidden.
     * @see graph#hide_edge
     * @see graph#restore_edge
     */
    bool is_hidden () const;

    /**
     * Returns the excentricity of the node, i.e. the maximum graph-theoretic
     * distance to another node
     *
     * @return excentricity of node. 
     */
    int excentricity() const;
    
    //================================================== Iterator types

    /**
     * @internal
     */
    typedef edges_t::const_iterator in_edges_iterator;
    /**
     * @internal
     */
    typedef edges_t::const_iterator out_edges_iterator;
    
    /**
     * @internal
     */
    class inout_edges_iterator;

    /**
     * @internal
     */
    class adj_nodes_iterator;

    /**
     * @internal
     */
    class adj_edges_iterator;

    //================================================== Iterators

    /**
     * Iterate through all adjacent nodes.
     *
     * @return start for iteration through all adjacent nodes
     */
    adj_nodes_iterator adj_nodes_begin() const;

    /**
     * Iterate through all adjacent nodes.
     *
     * @return end for iteration through all adjacent nodes
     */
    adj_nodes_iterator adj_nodes_end() const;

    /**
     * Iterate through all adjacent edges.
     *
     * @return start for iteration through all adjacent edges
     */
    adj_edges_iterator adj_edges_begin() const;

    /**
     * Iterate through all adjacent edges.
     *
     * @return end for iteration through all adjacent edges
     */
    adj_edges_iterator adj_edges_end() const;

    /**
     * Iterate through all incoming edges.
     *
     * @return start for iteration through all incoming edges
     */
    in_edges_iterator in_edges_begin() const;

    /**
     * Iterate through all incoming edges.
     *
     * @return end for iteration through all incoming edges
     */
    in_edges_iterator in_edges_end() const;

    /**
     * Iterate through all outgoing edges.
     *
     * @return start for iteration through all outgoing edges
     */
    out_edges_iterator out_edges_begin() const;

    /**
     * Iterate through all outgoing edges.
     *
     * @return end for iteration through all outgoing edges
     */
    out_edges_iterator out_edges_end() const;

    /**
     * Iterate through all incoming <em>and</em> outgoing edges.
     *
     * @return start for iteration through all incoming and outgoing edges
     */
    inout_edges_iterator inout_edges_begin() const;

    /**
     * Iterate through all incoming <em>and</em> outgoing edges.
     *
     * @return end for iteration through all incoming and outgoing edges
     */
    inout_edges_iterator inout_edges_end() const;

    //================================================== Implementation
    
private:
    node_data *data;
    
    bool is_directed() const;
    bool is_undirected() const;

    friend class graph;
    friend class edge;
    friend class adj_edges_iterator;

    GTL_EXTERN friend bool operator==(node, node);
    GTL_EXTERN friend bool operator!=(node, node);
    GTL_EXTERN friend bool operator<(node, node);
	GTL_EXTERN friend std::ostream& operator<< (std::ostream& os, const node& n);
};

/**
 * @short Iterator for adjacent edges of a node
 */
class GTL_EXTERN node::adj_edges_iterator : public bi_iter_edge
{
public:
	
    // constructor
    adj_edges_iterator();
    adj_edges_iterator(node, bool);

    // comparibility
    bool operator==(const adj_edges_iterator&) const;
    bool operator!=(const adj_edges_iterator&) const;

    // operators
    adj_edges_iterator &operator++();
    adj_edges_iterator operator++(int);
    adj_edges_iterator &operator--();
    adj_edges_iterator operator--(int);

    // dereferencing
    const edge& operator*() const;
    const edge* operator->() const;

private:
    in_edges_iterator akt_edge[2], last_edge[2], begin_edge[2];
    int inout;     // in=0, out=1
    bool directed; // graph directed ??
};
    
/**
 * @short Iterator for all incident edges of a node
 */
class GTL_EXTERN node::inout_edges_iterator : public bi_iter_edge
{
public:

    // constructor
    inout_edges_iterator();
    inout_edges_iterator(node n, bool start);

    // comparibility
    bool operator==(const inout_edges_iterator&) const;
    bool operator!=(const inout_edges_iterator&) const;

    // operators
    inout_edges_iterator &operator++();
    inout_edges_iterator operator++(int);
    inout_edges_iterator &operator--();
    inout_edges_iterator operator--(int);

    // dereferencing
    const edge& operator*() const;
    const edge* operator->() const;
	
private:
    in_edges_iterator akt_edge[2], last_edge, begin_edge;
    int inout;     // in=0, out=1
};

/**
 * @short Iterator for adjacent nodes of a node
 */
class GTL_EXTERN node::adj_nodes_iterator : public bi_iter_node
{
public:

    // constructor
    adj_nodes_iterator();
    adj_nodes_iterator(const node&, bool);

    // comparibility
    bool operator==(const adj_nodes_iterator&) const;
    bool operator!=(const adj_nodes_iterator&) const;

    // operators
    adj_nodes_iterator &operator++();
    adj_nodes_iterator operator++(int);
    adj_nodes_iterator &operator--();
    adj_nodes_iterator operator--(int);

    // dereferencing
    const node& operator*() const;
    const node* operator->() const;

private:
    adj_edges_iterator akt_edge;
    node int_node;
};


__GTL_END_NAMESPACE

//--------------------------------------------------------------------------
//   Iteration
//--------------------------------------------------------------------------

// #define forall_adj_nodes(v,w)	GTL_FORALL(v,w,node::adj_nodes_iterator,adj_nodes_)
#define forall_out_edges(e,v)	GTL_FORALL(e,v,node::out_edges_iterator,out_edges_)
#define forall_in_edges(e,v)	GTL_FORALL(e,v,node::in_edges_iterator,in_edges_)
#define forall_inout_edges(e,v)	GTL_FORALL(e,v,node::inout_edges_iterator,inout_edges_)
#define forall_adj_edges(e,v)	GTL_FORALL(e,v,node::adj_edges_iterator,adj_edges_)
    
#endif // GTL_NODE_H

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
