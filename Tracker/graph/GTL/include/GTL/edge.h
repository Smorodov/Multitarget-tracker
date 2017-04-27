/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   edge.h
//
//==========================================================================
// $Id: edge.h,v 1.15 2001/04/17 14:35:25 raitner Exp $

#ifndef GTL_EDGE_H
#define GTL_EDGE_H

#include <GTL/GTL.h>

#include <list>
#include <ostream>

__GTL_BEGIN_NAMESPACE
 
//--------------------------------------------------------------------------
//   For MSVC 5.0 edge.h has to be included before node.h and
//   {node,edge}_data.h. So we only declare needed classnames here
//--------------------------------------------------------------------------

class node;
typedef std::list<node> nodes_t;

class edge_data;

//--------------------------------------------------------------------------
//   edge
//--------------------------------------------------------------------------

/**
 * @short An edge in a graph
 */
class GTL_EXTERN edge
{
public:
    /**
     * Default constructor. Creates an invalid edge. 
     * The only way to obtain a valid edge is through @ref
     * graph#new_edge. Example:
     * <pre>
     *   graph g;
     *   node n1, n2;
     *   edge e;
     *
     *   n1 = g.new_node();
     *   n2 = g.new_node();
     *   e = g.new_edge(n1, n2);
     * </pre>
     *
     * @see graph#new_edge
     */
    edge();

    /**
     * Returns the source node of the edge.
     * 
     * @return source
     */
    node source() const;

    /**
     * Returns the target node of the edge.
     * 
     * @return target
     */
    node target() const;
	const node& target_() const;

    /**
     * Changes the direction of this edge. 
     */
    void reverse ();

    /**
     * Makes <code>n</code> the source of this edge. Takes O(1) time. 
     * 
     * @param <code>n</code> new source 
     */
    void change_source (node n);

    /**
     * Makes <code>n</code> the target of this edge. Takes O(1) time. 
     * 
     * @param <code>n</code> new target 
     */
    void change_target (node n);

    /**
     * Returns the node opposite to <code>n</code> referring to
     * this edge.
     * 
     * @param <code>n</code> a node incident to this edge
     */
    const node& opposite(node n) const;

    /**
     * @internal
     */
    nodes_t sources() const;

    /**
     * @internal
     */
	nodes_t targets() const;

    /**
     * @internal
     */
    int id() const;

    
    /**
     * Returns true iff node is hidden. 
     * 
     * @return true iff node is hidden.
     * @see graph#hide_edge
     * @see graph#restore_edge
     */
    bool is_hidden () const;


    //================================================== Implementation
    
private:
    edge_data *data;
    
    void remove_from(int where) const; // 0 = sources, 1 == targets

    friend class graph;
    friend class node;
    
    GTL_EXTERN friend bool operator==(edge, edge);
    GTL_EXTERN friend bool operator!=(edge, edge);
    GTL_EXTERN friend bool operator<(edge, edge);
	GTL_EXTERN friend std::ostream& operator<< (std::ostream& os, const edge& e);
};

typedef std::list<edge> edges_t;

__GTL_END_NAMESPACE

#endif // GTL_EDGE_H

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
