/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   node.cpp
//
//==========================================================================
// $Id: node.cpp,v 1.18 2001/11/07 13:58:10 pick Exp $

#include <GTL/node_data.h>
#include <GTL/edge_data.h>
#include <GTL/graph.h>
#include <GTL/bfs.h>

#include <cassert>
#include <iostream>

__GTL_BEGIN_NAMESPACE

node::node() :
    data(0)
{
}

GTL_EXTERN std::ostream& operator<< (std::ostream& os, const node& n) {
    if (n != node()) {
	return os << "[" << n.id() << "]";
    } else {
	return os << "[ UNDEF ]";
    }
}

node::adj_nodes_iterator node::adj_nodes_begin() const
{ 
    return node::adj_nodes_iterator(*this, true);
}

node::adj_nodes_iterator node::adj_nodes_end() const
{
    return node::adj_nodes_iterator(*this, false);
}

node::adj_edges_iterator node::adj_edges_begin() const
{
    return node::adj_edges_iterator(*this, true);
}

node::adj_edges_iterator node::adj_edges_end() const
{
    return node::adj_edges_iterator(*this, false);
}

node::inout_edges_iterator node::inout_edges_begin() const
{
    return node::inout_edges_iterator(*this, true);
}

node::inout_edges_iterator node::inout_edges_end() const
{
    return node::inout_edges_iterator(*this, false);
}

node::in_edges_iterator node::in_edges_begin() const
{
    return data->edges[0].begin();
}

node::in_edges_iterator node::in_edges_end() const
{
    return data->edges[0].end();
}

node::out_edges_iterator node::out_edges_begin() const
{
    return data->edges[1].begin();
}

node::out_edges_iterator node::out_edges_end() const
{
    return data->edges[1].end();
}

int node::degree() const
{
    return outdeg() + indeg();
}

int node::outdeg() const
{
    return data->edges[1].size();
}

int node::indeg() const
{
    return data->edges[0].size();
}

int node::id() const
{
    return data->id;
}

bool node::is_directed() const
{
    return data->owner->is_directed();
}

bool node::is_undirected() const
{
    return data->owner->is_undirected();
}

const node& node::opposite(edge e) const
{
    // not implemented for hypergraphs
    assert(e.data);

    node& s = *(e.data->nodes[0].begin());
    if (*this == s)
	return *(e.data->nodes[1].begin());
    else
	return s;
}

nodes_t node::opposites(edge) const
{
    // not implemented yet
	return nodes_t(); // to avoid compiler warnings
}

bool node::is_hidden () const
{
    return data->hidden;
}

int node::excentricity() const
{
    bfs b;
    b.start_node(*this);
    b.calc_level(true);
    b.run(*data->owner);

    node last_node = *(--b.end());
    
    return b.level(last_node);
}

GTL_EXTERN bool operator==(node v1, node v2)
{
    return v1.data == v2.data;
}

GTL_EXTERN bool operator!=(node v1, node v2)
{
    return v1.data != v2.data;
}

GTL_EXTERN bool operator<(node v1, node v2)
{
    return v1.data < v2.data;
}

//--------------------------------------------------------------------------
//   adj_edges_iterator
//--------------------------------------------------------------------------

node::adj_edges_iterator::adj_edges_iterator()
{
}

node::adj_edges_iterator::adj_edges_iterator(node n, bool start)
{
    // iterators that are used everytime
    last_edge[0] = n.out_edges_end();
    last_edge[1] = n.in_edges_end();
    directed  = n.is_directed();
    if (!directed)
    {
	begin_edge[0] = n.out_edges_begin();
	begin_edge[1] = n.in_edges_begin();
    }

    // set at start or end
    if (start)
    {
	inout = 0;
	akt_edge[0] = n.out_edges_begin();
	if (!directed)
	{
	    akt_edge[1] = n.in_edges_begin();
	    if (akt_edge[0] == last_edge[0])
		inout = 1;
	}
    }
    else
    {
	inout = directed ? 0 : 1;
	akt_edge[0] = n.out_edges_end();
	if (!directed)
	    akt_edge[1] = n.in_edges_end();
    }
}

bool node::adj_edges_iterator::operator==(const
					  node::adj_edges_iterator& i) const
{
    return i.akt_edge[i.inout] == akt_edge[inout];
}

bool node::adj_edges_iterator::operator!=(const
					  node::adj_edges_iterator& i) const
{
    return i.akt_edge[i.inout] != akt_edge[inout];
}

node::adj_edges_iterator& node::adj_edges_iterator::operator++()
{
    if (directed)
	++akt_edge[inout];
    else
    {
	if (inout == 0)
	{
	    ++akt_edge[0];
	    if (akt_edge[0] == last_edge[0])
		++inout;
	}
	else // inout == 1
	{
	    if (akt_edge[1] == last_edge[1])
	    {
		inout = 0;
		akt_edge[0] = begin_edge[0];
		akt_edge[1] = begin_edge[1];
		if (begin_edge[0] == last_edge[0])
		    inout = 1;
	    }
	    else
		++akt_edge[inout];
	}
    }
    return *this;
}

node::adj_edges_iterator node::adj_edges_iterator::operator++(int)
{
    node::adj_edges_iterator tmp = *this;
    operator++();
    return tmp;
}

node::adj_edges_iterator& node::adj_edges_iterator::operator--()
{
    if (!directed && inout == 1 && akt_edge[1] == begin_edge[1])
	inout = 0;
    --akt_edge[inout];
    return *this;
}

node::adj_edges_iterator node::adj_edges_iterator::operator--(int)
{
    node::adj_edges_iterator tmp = *this;
    operator--();
    return tmp;
}

const edge& node::adj_edges_iterator::operator*() const
{
    return *akt_edge[inout];
}

const edge* node::adj_edges_iterator::operator->() const
{
    return akt_edge[inout].operator->();
}

//--------------------------------------------------------------------------
//   inout_edges_iterator
//--------------------------------------------------------------------------

node::inout_edges_iterator::inout_edges_iterator()
{
}

node::inout_edges_iterator::inout_edges_iterator(node n, bool start)
{
    // iterators that are used everytime
    last_edge = n.in_edges_end();
    begin_edge  = n.out_edges_begin();

    // set at start or end
    if (start)
    {
	inout = 0;
	akt_edge[0] = n.in_edges_begin();
	akt_edge[1] = n.out_edges_begin();
	if (akt_edge[0] == last_edge)
	    inout = 1;
    }
    else
    {
	inout = 1;
	akt_edge[0] = n.in_edges_end();
	akt_edge[1] = n.out_edges_end();
    }
}

bool node::inout_edges_iterator::operator==(const
				         node::inout_edges_iterator& i) const
{
    return i.akt_edge[i.inout] == akt_edge[inout];
}

bool node::inout_edges_iterator::operator!=(const
					 node::inout_edges_iterator& i) const
{
    return i.akt_edge[i.inout] != akt_edge[inout];
}

node::inout_edges_iterator& node::inout_edges_iterator::operator++()
{
    ++akt_edge[inout];
    if ((akt_edge[inout] == last_edge) && (inout==0))
	    ++inout;
    return *this;
}

node::inout_edges_iterator node::inout_edges_iterator::operator++(int)
{
    node::inout_edges_iterator tmp = *this;
    operator++();
    return tmp;
}

node::inout_edges_iterator& node::inout_edges_iterator::operator--()
{
    if (inout == 1 && (akt_edge[1] == begin_edge))
	inout = 0;
    --akt_edge[inout];
    return *this;
}

node::inout_edges_iterator node::inout_edges_iterator::operator--(int)
{
    node::inout_edges_iterator tmp = *this;
    operator--();
    return tmp;
}

const edge& node::inout_edges_iterator::operator*() const
{
    return *akt_edge[inout];
}

const edge* node::inout_edges_iterator::operator->() const
{
    return akt_edge[inout].operator->();
}

//--------------------------------------------------------------------------
//   adj_nodes_iterator
//--------------------------------------------------------------------------

node::adj_nodes_iterator::adj_nodes_iterator()
{
}

node::adj_nodes_iterator::adj_nodes_iterator(const node& n, bool start)
{
    int_node = n;
    if (start)
	akt_edge = n.adj_edges_begin();
    else
	akt_edge = n.adj_edges_end();
}

bool node::adj_nodes_iterator::operator==(const
					  node::adj_nodes_iterator& i) const
{
    return i.akt_edge == akt_edge;
}

bool node::adj_nodes_iterator::operator!=(const
					  node::adj_nodes_iterator& i) const
{
    return i.akt_edge != akt_edge;
}

node::adj_nodes_iterator& node::adj_nodes_iterator::operator++()
{
    ++akt_edge;
    return *this;
} 

node::adj_nodes_iterator node::adj_nodes_iterator::operator++(int)
{
    node::adj_nodes_iterator tmp = *this;
    operator++();
    return tmp;
}

node::adj_nodes_iterator& node::adj_nodes_iterator::operator--()
{
    --akt_edge;
    return *this;
}

node::adj_nodes_iterator node::adj_nodes_iterator::operator--(int)
{
    node::adj_nodes_iterator tmp = *this;
    operator--();
    return tmp;
}

const node& node::adj_nodes_iterator::operator*() const
{
    return int_node.opposite(*akt_edge);
}

const node* node::adj_nodes_iterator::operator->() const
{
    return &(int_node.opposite(*akt_edge));
}

__GTL_END_NAMESPACE

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
