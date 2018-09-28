/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   node_map.h
//
//==========================================================================
// $Id: node_map.h,v 1.8 2005/06/14 12:22:12 raitner Exp $

#ifndef GTL_NODE_MAP_H
#define GTL_NODE_MAP_H

#include <GTL/GTL.h>
#include <GTL/node.h>
#include <GTL/ne_map.h>

__GTL_BEGIN_NAMESPACE

class graph;

/**
 * @short A specialized map with nodes as keys
 *
 * A <code>node_map</code> is a specialized and optimized map
 * implementation with nodes as keys. Using a <code>node_map</code> is
 * the standard way to attach user defined information to 
 * the nodes of a <code>graph</code>.
 *
 * An example of usage:
 * <pre>
 *   graph g;
 *
 *   node v1 = g.new_node();
 *   node v2 = g.new_node();
 *
 *   node_map&lt;string&gt; label(g, "Default Label");
 *
 *   label[v1] = "v1";
 *   label[v2] = "v2";
 *
 *   assert(label[v1] != label[v2]);
 * </pre>
 *
 * The nodes used as keys for a <code>node_map</code> MUST be nodes
 * of the same graph. If you want to use nodes from different graphs, use
 * a <code>map&lt;node,T&gt;</code> instead. A graph and a copy of it are
 * considered to be different.
 *
 * Most of the functionality of <code>node_map</code> is inherited from
 * @ref ne_map.
 *
 * @see edge_map
 */
template <class T, class Alloc = std::allocator<T> >
class node_map : public ne_map<node, T, graph, Alloc>
{
public:

    /**
     * Constructs an empty <code>node_map</code> not associated with any
     * <code>graph</code>. You may (but need not) call
     * <code>ne_map::init(const graph &, T)</code> to associate it to
     * a <code>graph</code>.
     */
    node_map() : ne_map<node, T, graph, Alloc>() {};

    /**
     * Constructs a <code>node_map</code> associated to the graph
     * <code>g</code>.
     * The value associated to each node in <code>g</code> is set to
     * <code>t</code>.
     */
    explicit node_map(const graph &g, T t=T()) : ne_map<node, T, graph, Alloc>(g,t) {};
};

__GTL_END_NAMESPACE

#endif // GTL_NODE_MAP_H

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
