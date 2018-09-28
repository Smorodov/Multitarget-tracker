/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   ne_map.h - common implementation of node_map and edge_map
//
//==========================================================================
// $Id: ne_map.h,v 1.20 2005/06/14 12:22:12 raitner Exp $

#ifndef GTL_NE_MAP_H
#define GTL_NE_MAP_H

#include <GTL/GTL.h>

#include <vector>
#include <cassert>

//--------------------------------------------------------------------------
//   Class declaration
//--------------------------------------------------------------------------

__GTL_BEGIN_NAMESPACE

/**
 * @short Baseclass for node_map and edge_map
 *
 * ne_map is the common implementation of <code>@ref node_map </code>
 * and <code>@ref edge_map </code> and cannot be used directly.
 */

template <class Key, class Value, class Graph, class Alloc = std::allocator<Value> > 
class ne_map
{
protected:
    
    //================================================== Constructors

    /**
     * Constructs an empty <code>ne_map</code> not associated to any
     * <code>graph</code>.
     */
    ne_map();

    /**
     * Constructs a <code>ne_map</code> associated to the
     * <code>graph g</code>. The value associated to each key is set
     * to <code>def</code>.
     * You may (but need not) call
     * <code>ne_map::init(const graph &, T)</code> to associate it to
     * a <code>graph</code>.
     *
     * @param <code>g</code>   associated <code>graph</code>
     * @param <code>def</code> default value
     */
    explicit ne_map(const Graph &g, Value def=Value());

    //================================================== Operations
    
public:

    /**
     * Initializes the ne_map to hold information for the elements
     * of graph g. def is the value associated with all elements.
     *
     * @param <code>g</code>   associated <code>graph</code>
     * @param <code>def</code> default value
     */
    void init(const Graph &, Value def=Value());

    /**
     * @internal
     */
#if defined(__GTL_MSVCC) && _MSC_VER < 1310
    typedef Value& value_reference;
#else
	typedef typename std::vector<Value, Alloc>::reference value_reference;
#endif

    /**
     * @internal
     */
#if defined(__GTL_MSVCC) && _MSC_VER < 1310
    typedef const Value& const_value_reference;
#else
	typedef typename std::vector<Value, Alloc>::const_reference const_value_reference;
#endif
    
    /**
     * Read/write accessor function to the value associated with
     * <code>key</code>.
     * Use this function to change the value of an element in the
     * <code>ne_map</code>. Assume that <code>ne</code> is a
     * <code>ne_map&lt;int&gt;</code>. Then you can assign the value
     * 5 to <code>key</code> with:
     * <pre>
     *   ne[key] = 5;
     * </pre>
     *
     * If there is no entry in the <code>ne_map</code> associated
     * with <code>key</code>, one is created.
     *
     * @param key Key of the Entry to change
     * @return a reference to the value associated to <code>key</code>.	
     */
    value_reference operator[](Key key);

    /**
     * Read-only accessor function to the value associated with
     * <code>key</code>.
     * Use this function to read the value of an element in the
     * <code>ne_map</code>. Assume that <code>ne</code> is a
     * <code>ne_map&lt;int&gt;</code>. Then you can print the value
     * associated with <code>key</code> with:
     * <pre>
     *   cout << ne[key];
     * </pre>
     *
     * @param key Key of the Entry to look up
     * @return a const reference to the value associated to
     * <code>key</code>.	
     */
    const_value_reference operator[](Key key) const;

    /**
     * Erases a elements of this nodemap
     */
    void clear ();

    //================================================== Implementation
    
private:
	std::vector<Value, Alloc> data;
};

// Implementation Begin

template <class Key, class Value, class Graph, class Alloc>
  ne_map<Key,Value,Graph,Alloc>::ne_map()
{
}

template <class Key, class Value, class Graph, class Alloc>
ne_map<Key,Value,Graph,Alloc>::ne_map(const Graph &g, Value t2) :
    data(g.number_of_ids(Key()), t2)
{
}

template <class Key, class Value, class Graph, class Alloc>
void ne_map<Key,Value,Graph,Alloc>::init(const Graph &g, Value t2)
{
    int n = g.number_of_ids(Key());
    data.resize(n);
    fill_n(data.begin(), n, t2);
}

template <class Key, class Value, class Graph, class Alloc>
typename ne_map<Key,Value,Graph,Alloc>::value_reference ne_map<Key,Value,Graph,Alloc>::operator[](Key t1)
{
    if(t1.id() >= (signed)data.size())
    {
	if (t1.id() >= (signed)data.capacity()) {
	    data.reserve((6 * t1.id()) / 5 + 1);
	}

	data.insert(data.end(), t1.id()+1-data.size(), Value());
    }
    return data.operator[](t1.id());
}

template <class Key, class Value, class Graph, class Alloc>
typename ne_map<Key,Value,Graph,Alloc>::const_value_reference ne_map<Key,Value,Graph,Alloc>::operator[](Key t1) const
{
    assert(t1.id() < (signed)data.size());
    return data.operator[](t1.id());
}

template <class Key, class Value, class Graph, class Alloc>
void ne_map<Key,Value,Graph,Alloc>::clear ()
{
    data.clear();
}

// Implementation End

__GTL_END_NAMESPACE

#endif // GTL_NE_MAP_H

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
