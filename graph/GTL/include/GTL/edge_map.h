/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   edge_map.h
//
//==========================================================================
// $Id: edge_map.h,v 1.8 2005/06/14 12:22:12 raitner Exp $

#ifndef GTL_EDGE_MAP_H
#define GTL_EDGE_MAP_H

#include <GTL/GTL.h>
#include <GTL/edge.h>
#include <GTL/ne_map.h>

__GTL_BEGIN_NAMESPACE

class graph; 

/**
 * @short A specialized map with edges as keys
 *
 * A <code>edge_map</code> is a specialized and optimized map
 * implementation with edges as keys. Using a <code>edge_map</code> is
 * the standard way to attach user defined information to 
 * the edges of a <code>graph</code>.
 *
 * An example of usage:
 * <pre>
 *   graph g;
 *
 *   node v1 = g.new_node();
 *   node v2 = g.new_node();
 *   edge e = g.new_edge(v1, v2);
 *
 *   edge_map&lt;string&gt; label(g, "Default Label");
 *
 *   label[e] = "An edge";
 *
 *   assert(label[e] == "An edge");
 * </pre>
 *
 * The edges used as keys for a <code>edge_map</code> MUST be edges
 * of the same graph. If you want to use edges from different graphs, use
 * a <code>map&lt;edge,T&gt;</code> instead. A graph and a copy of it are
 * considered to be different.
 *
 * Most of the functionality of <code>edge_map</code> is inherited from
 * @ref ne_map.
 *
 * @see node_map
 */
template <class T, class Alloc = std::allocator<T> >
class edge_map : public ne_map<edge, T, graph, Alloc>
{
public:

    /**
     * Constructs an empty <code>edge_map</code> not associated with any
     * <code>graph</code>. You may (but need not) call
     * <code>ne_map::init(const graph &, T)</code> to associate it to
     * a <code>graph</code>.
     */
    edge_map() : ne_map<edge, T, graph, Alloc>() {};
    
    /**
     * Constructs a <code>edge_map</code> associated to the graph
     * <code>g</code>.
     * The value associated to each edge in <code>g</code> is set to
     * <code>t</code>.
     */
     explicit edge_map(const graph &g, T t=T()) : 
        ne_map<edge, T, graph, Alloc>(g,t) {};
};

__GTL_END_NAMESPACE

#endif // GTL_EDGE_MAP_H

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
