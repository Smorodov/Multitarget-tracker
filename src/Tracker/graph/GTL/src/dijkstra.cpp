/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   dijkstra.cpp 
//
//==========================================================================
//$Id: dijkstra.cpp,v 1.6 2002/12/23 13:46:41 chris Exp $

#include <GTL/dijkstra.h>
#include <GTL/bin_heap.h>

#include <cstdlib>
#include <iostream>
#include <cassert>

__GTL_BEGIN_NAMESPACE

/**
 * @internal
 * Binary predicate that compares two nodes according to their distance.
 */
class less_dist : public std::binary_function<node, node, bool>
{
public:
    /**
     * @internal
     * Constructor sets pointer to node distances and infimum info.
     */
    less_dist(const node_map<double>* dist, const node_map<int>* mark)
    {
	this->dist = dist;
	this->mark = mark;
    }

    /**
     * @internal
     * Compares distances of @p n1 and @p n2.
     */
    bool operator()(const node n1, const node n2) const
    {
	if (((*mark)[n1] == dijkstra::black) &&
	    ((*mark)[n2] == dijkstra::black))
	{
	    return false;
	}
	else if ((*mark)[n1] == dijkstra::black)
	{
	    return false;
	}
	else if ((*mark)[n2] == dijkstra::black)
	{
	    return true;
	}
	return (*dist)[n1] < (*dist)[n2];
    }
private:
    /**
     * @internal
     * Node distances from source.
     */
    const node_map<double>* dist;

    /**
     * @internal
     * Infimum distance info (color of nodes).
     */
    const node_map<int>* mark;
};


dijkstra::dijkstra()
{
    reset_algorithm();
}


dijkstra::~dijkstra()
{
}


void dijkstra::source(const node& n)
{
    s = n;
}


void dijkstra::target(const node& n)
{
    t = n;
}


void dijkstra::weights(const edge_map<double>& weight)
{
    this->weight = weight;
    weights_set = true;
}


void dijkstra::store_preds(bool set)
{
    preds_set = set;
}


int dijkstra::check(graph& G)
{
    if ((s == node()) || (!weights_set))
    {
	return GTL_ERROR;
    }

    bool source_found = false;
    graph::node_iterator node_it;
    graph::node_iterator nodes_end = G.nodes_end();
    for (node_it = G.nodes_begin(); node_it != nodes_end; ++node_it)
    {
	if (*node_it == s)
        {
	    source_found = true;
	    break;
	}
    }
    if (!source_found)
    {
	return(GTL_ERROR);
    }

    graph::edge_iterator edge_it;
    graph::edge_iterator edges_end = G.edges_end();
    for(edge_it = G.edges_begin(); edge_it != edges_end; ++edge_it)
    {
	if (weight[*edge_it] < 0.0)
	{
	    return false;
	}
    }

    return GTL_OK;
}


int dijkstra::run(graph& G)
{
    init(G);

    less_dist prd(&dist, &mark);
    bin_heap<node, less_dist> node_heap(prd, G.number_of_nodes());
    mark[s] = grey;
    dist[s] = 0.0;
    node_heap.push(s);
    while (!node_heap.is_empty())
    {

	// debug:
	// node_heap.print_data_container();

	node cur_node = node_heap.top();
	node_heap.pop();

	// debug:
	// node_heap.print_data_container();

	mark[cur_node] = white;
	if (cur_node == t)
	{
	    // if @a t is set through #target we are ready
	    return GTL_OK;
	}

	node::adj_edges_iterator adj_edge_it;
	node::adj_edges_iterator adj_edges_end = cur_node.adj_edges_end();
	for (adj_edge_it = cur_node.adj_edges_begin();
	     adj_edge_it != adj_edges_end;
	     ++adj_edge_it)
	{
	    node op_node = adj_edge_it->opposite(cur_node);
	    if (mark[op_node] == black)
	    {
		mark[op_node] = grey;
		dist[op_node] = dist[cur_node] + weight[*adj_edge_it];
		node_heap.push(op_node);

		// debug:
		// node_heap.print_data_container();

		if (preds_set)
		{
		    pred[op_node] = *adj_edge_it;
		}
	    }
	    else if (mark[op_node] == grey)
	    {
		if (dist[op_node] > dist[cur_node] + weight[*adj_edge_it])
		{
		    dist[op_node] = dist[cur_node] + weight[*adj_edge_it];
		    node_heap.changeKey(op_node);

		    // debug:
		    // node_heap.print_data_container();

		    if (preds_set)
		    {
			pred[op_node] = *adj_edge_it;
		    }
		}
    	    }
	    else    // (mark[op_node] == white)
	    {
		// nothing to do: shortest distance to op_node is already
		//		  computed
	    }
	}
    }

    return GTL_OK;
}


node dijkstra::source() const
{
    return s;
}


node dijkstra::target() const
{
    return t;
}


bool dijkstra::store_preds() const
{
    return preds_set;
}


bool dijkstra::reached(const node& n) const
{
    return mark[n] != black;
}


double dijkstra::distance(const node& n) const
{
    return dist[n];
}


node dijkstra::predecessor_node(const node& n) const
{
    assert(preds_set);
    if ((n == s) || (!reached(n)))
    {
	return node();
    }
    return pred[n].opposite(n);
}


edge dijkstra::predecessor_edge(const node& n) const
{
    assert(preds_set);
    return pred[n];
}


dijkstra::shortest_path_node_iterator dijkstra::shortest_path_nodes_begin(
    const node& dest)
{
    assert(preds_set);
    if ((shortest_path_node_list[dest].empty()) &&
	(dest != s) &&
	(reached(dest)))
    {
	fill_node_list(dest);
    }
    return shortest_path_node_list[dest].begin();
}


dijkstra::shortest_path_node_iterator dijkstra::shortest_path_nodes_end(
    const node& dest)
{
    assert(preds_set);
    if ((shortest_path_node_list[dest].empty()) &&
	(dest != s) &&
	(reached(dest)))
    {
	fill_node_list(dest);
    }
    return shortest_path_node_list[dest].end();
}


dijkstra::shortest_path_edge_iterator dijkstra::shortest_path_edges_begin(
    const node& dest)
{
    assert(preds_set);
    if ((shortest_path_edge_list[dest].empty()) &&
	(dest != s) &&
	(reached(dest)))
    {
	fill_edge_list(dest);
    }
    return shortest_path_edge_list[dest].begin();
}


dijkstra::shortest_path_edge_iterator dijkstra::shortest_path_edges_end(
    const node& dest)
{
    assert(preds_set);
    if ((shortest_path_edge_list[dest].empty()) &&
	(dest != s) &&
	(reached(dest)))
    {
	fill_edge_list(dest);
    }
    return shortest_path_edge_list[dest].end();
}


void dijkstra::reset()
{
    reset_algorithm();
}


void dijkstra::reset_algorithm()
{
    s = node();
    t = node();
    weights_set = false;
    preds_set = false;
}


void dijkstra::init(graph& G)
{
    dist.init(G, -1.0);
    mark.init(G, black);
    
    if (preds_set)
    {
	pred.init(G, edge());
	graph::node_iterator node_it;
	graph::node_iterator nodes_end = G.nodes_end();
	for (node_it = G.nodes_begin(); node_it != nodes_end; ++node_it)
	{
	    shortest_path_node_list[(*node_it)].clear();
	    shortest_path_edge_list[(*node_it)].clear();
	}
    }
}


void dijkstra::fill_node_list(const node& dest)
{
    if ((dest == s) || (!reached(dest)))
    {
	return;
    }

    node cur_node = dest;
    while (cur_node != node())
    {
	shortest_path_node_list[dest].push_front(cur_node);
	cur_node = predecessor_node(cur_node);
    }
}


void dijkstra::fill_edge_list(const node& dest)
{
    if ((dest == s) || (!reached(dest)))
    {
	return;
    }

    node cur_node = dest;
    edge cur_edge = predecessor_edge(dest);
    while (cur_edge != edge())
    {
	shortest_path_edge_list[dest].push_front(cur_edge);
	cur_node = predecessor_node(cur_node);
	cur_edge = predecessor_edge(cur_node);
    }
}

__GTL_END_NAMESPACE

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
