/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   edge.cpp
//
//==========================================================================
// $Id: edge.cpp,v 1.17 2001/11/07 13:58:09 pick Exp $

#include <GTL/node_data.h>
#include <GTL/edge_data.h>
#include <cassert>
#include <iostream>

__GTL_BEGIN_NAMESPACE

//--------------------------------------------------------------------------
//   edge
//--------------------------------------------------------------------------

edge::edge() :
    data(0)
{
}

GTL_EXTERN std::ostream& operator<< (std::ostream& os, const edge& e) {
    if (e != edge ()) {
	return os << e.source() << "-->" << e.target();
    } else {
	return os << "UNDEF";
    }
}

node edge::source() const
{
    return data->nodes[0].front();
}

node edge::target() const
{
    return data->nodes[1].front();
}

const node& edge::target_() const
{
	return data->nodes[1].front();
}

void edge::change_source (node new_source)
{
    //
    // First delete this edge from source's adjacency list
    // and clear the list of sources 
    //
    
	nodes_t::iterator the_nodes = data->nodes[0].begin();
	nodes_t::iterator the_nodes_end = data->nodes[0].end();

    while(the_nodes != the_nodes_end)
    {
	the_nodes->data->edges[1].erase (data->adj_pos[0].front());
	data->adj_pos[0].pop_front();
	
	the_nodes = data->nodes[0].erase (the_nodes);
    }

    //
    // Just to be sure :)
    // 
    
    assert (data->nodes[0].empty());
    assert (data->adj_pos[0].empty());

    //
    // insert this edge in the list of outgoing edges of new_source 
    //

    data->adj_pos[0].push_back(new_source.data->edges[1].insert (
	new_source.data->edges[1].end(), *this));

    //
    // make new_source a source of this node.
    //

    data->nodes[0].push_back (new_source);
}


void edge::change_target (node new_target) {
    //
    // First delete this edge from target's adjacency list
    // and clear the list of targets
    //
    
	nodes_t::iterator the_nodes = data->nodes[1].begin();
	nodes_t::iterator the_nodes_end = data->nodes[1].end();

    while(the_nodes != the_nodes_end)
    {
	the_nodes->data->edges[0].erase (data->adj_pos[1].front());
	data->adj_pos[1].pop_front();
	
	the_nodes = data->nodes[1].erase (the_nodes);
    }

    //
    // Just to be sure :)
    // 

    assert (data->nodes[1].empty());
    assert (data->adj_pos[1].empty());

    //
    // insert this edge in the list of incoming edges of new_target 
    //

    data->adj_pos[1].push_back(new_target.data->edges[0].insert (
	new_target.data->edges[0].end(), *this));

    //
    // make new_target a target of this node.
    //

    data->nodes[1].push_back (new_target);
}


void edge::reverse () 
{
    //
    // First delete this edge from all adjacency lists
    //
    
	nodes_t::iterator the_nodes = data->nodes[0].begin();
	nodes_t::iterator the_nodes_end = data->nodes[0].end();

    while(the_nodes != the_nodes_end)
    {
	the_nodes->data->edges[1].erase (data->adj_pos[0].front());
	data->adj_pos[0].pop_front();

	++the_nodes;
    }

    the_nodes = data->nodes[1].begin();
    the_nodes_end = data->nodes[1].end();

    while(the_nodes != the_nodes_end)
    {
	the_nodes->data->edges[0].erase (data->adj_pos[1].front());
	data->adj_pos[1].pop_front();

	++the_nodes;
    }

    //
    // Now the lists of positions in the adjacency - lists should be empty 
    //
    
    assert (data->adj_pos[0].empty());
    assert (data->adj_pos[1].empty());

    //
    // Now insert this edge reversed
    //

    the_nodes = data->nodes[1].begin();
    the_nodes_end = data->nodes[1].end();

    while(the_nodes != the_nodes_end)
    {
	data->adj_pos[0].push_back(the_nodes->data->edges[1].insert (
	    the_nodes->data->edges[1].end(), *this));

	++the_nodes;
    }

    the_nodes = data->nodes[0].begin();
    the_nodes_end = data->nodes[0].end();

    while(the_nodes != the_nodes_end)
    {
	data->adj_pos[1].push_back(the_nodes->data->edges[0].insert (
	    the_nodes->data->edges[0].end(), *this));

	++the_nodes;
    }

    //
    // swap nodes[0] and nodes[1]
    // 
    
	nodes_t tmp = data->nodes[0];
    data->nodes[0] = data->nodes[1];
    data->nodes[1] = tmp;
}

    

nodes_t edge::sources() const
{
    return data->nodes[0];
}

nodes_t edge::targets() const
{
    return data->nodes[1];
}

int edge::id() const
{
    return data->id;
}

bool edge::is_hidden () const
{
    return data->hidden;
}

void edge::remove_from(int where) const
{
	nodes_t::iterator the_nodes = data->nodes[where].begin();
	nodes_t::iterator the_nodes_end = data->nodes[where].end();

	std::list<edges_t::iterator>::iterator the_adj_pos = data->adj_pos[where].begin();

	while (the_nodes != the_nodes_end)
	{
		the_nodes->data->edges[1 - where].erase(*the_adj_pos);

		++the_nodes;
		++the_adj_pos;
	}
}

const node& edge::opposite(node n) const
{
    // not implemented for hypergraphs
    assert(data);

    node& s = *(data->nodes[0].begin());
    if (n == s)
	return *(data->nodes[1].begin());
    else
	return s;
}

GTL_EXTERN bool operator==(edge e1, edge e2)
{
    return e1.data == e2.data;
}

GTL_EXTERN bool operator!=(edge e1, edge e2)
{
    return e1.data != e2.data;
}

GTL_EXTERN bool operator<(edge e1, edge e2)
{
    return e1.data < e2.data;
}

__GTL_END_NAMESPACE

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
