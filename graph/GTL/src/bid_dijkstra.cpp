/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   bid_dijkstra.cpp 
//
//==========================================================================
//$Id: bid_dijkstra.cpp,v 1.2 2004/05/06 11:58:19 chris Exp $

#include <GTL/bid_dijkstra.h>
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
	if (((*mark)[n1] == bid_dijkstra::black) &&
	    ((*mark)[n2] == bid_dijkstra::black))
	{
	    return false;
	}
	else if ((*mark)[n1] == bid_dijkstra::black)
	{
	    return false;
	}
	else if ((*mark)[n2] == bid_dijkstra::black)
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


bid_dijkstra::bid_dijkstra()
{
    reset_algorithm();
}


bid_dijkstra::~bid_dijkstra()
{
}


void bid_dijkstra::source_target(const node& s, const node& t)
{
    this->s = s;
    this->t = t;
}


void bid_dijkstra::weights(const edge_map<double>& weight)
{
    this->weight = weight;
    weights_set = true;
}


void bid_dijkstra::store_path(bool set)
{
    path_set = set;
}


int bid_dijkstra::check(graph& G)
{
    if ((s == node()) || (t == node()) || (!weights_set))
    {
	return GTL_ERROR;
    }

    bool source_found = false;
    bool target_found = false;
    graph::node_iterator node_it;
    graph::node_iterator nodes_end = G.nodes_end();
    for (node_it = G.nodes_begin(); node_it != nodes_end; ++node_it)
    {
	if (*node_it == s)
        {
	    source_found = true;
	    if (target_found)
	    {
		break;
	    }
	}
	if (*node_it == t)
	{
	    target_found = true;
	    if (source_found)
	    {
		break;
	    }
	}
    }
    if ((!source_found) || (!target_found))
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


int bid_dijkstra::run(graph& G)
{
    init(G);

    double max_dist = 1;
    graph::edge_iterator edge_it;
    graph::edge_iterator edges_end = G.edges_end();
    for(edge_it = G.edges_begin(); edge_it != edges_end; ++edge_it)
    {
	max_dist += weight[*edge_it];
    }

    less_dist source_prd(&source_dist, &source_mark);
    less_dist target_prd(&target_dist, &target_mark);
    bin_heap<node, less_dist> source_heap(source_prd,
					  G.number_of_nodes());
    bin_heap<node, less_dist> target_heap(target_prd,
					  G.number_of_nodes());

    source_mark[s] = grey;
    source_dist[s] = 0.0;
    source_heap.push(s);
    target_mark[t] = grey;
    target_dist[t] = 0.0;
    target_heap.push(t);
    while ((!source_heap.is_empty()) || (!target_heap.is_empty()))
    {
	if (source_dist[source_heap.top()] <=
	    target_dist[target_heap.top()])
	{

	    // debug:
	    // source_heap.print_data_container();

	    node cur_node = source_heap.top();
	    source_heap.pop();

	    // debug:
	    // source_heap.print_data_container();

	    source_mark[cur_node] = white;

	    if ((target_mark[cur_node] == white) &&
		(max_dist == source_dist[cur_node] +
		    target_dist[cur_node]))
	    {
		fill_node_edge_lists(cur_node);
		break;
	    }

	    node::adj_edges_iterator adj_edge_it;
	    node::adj_edges_iterator adj_edges_end =
		cur_node.adj_edges_end();
	    for (adj_edge_it = cur_node.adj_edges_begin();
		adj_edge_it != adj_edges_end;
		++adj_edge_it)
	    {
		node op_node = adj_edge_it->opposite(cur_node);
		if (source_mark[op_node] == black)
		{
		    source_mark[op_node] = grey;
		    source_dist[op_node] = source_dist[cur_node] +
			weight[*adj_edge_it];
		    source_heap.push(op_node);

		    // debug:
		    // source_heap.print_data_container();

		    if (path_set)
		    {
			pred[op_node] = *adj_edge_it;
		    }

		    if ((target_mark[op_node] == grey) ||
			(target_mark[op_node] == white))
		    {
			if (max_dist > source_dist[op_node] +
			    target_dist[op_node])
			{
			    max_dist = source_dist[op_node] +
				target_dist[op_node];
			}
		    }
		}
		else if (source_mark[op_node] == grey)
		{
		    if (source_dist[op_node] > source_dist[cur_node] +
			weight[*adj_edge_it])
		    {
			source_dist[op_node] = source_dist[cur_node] +
			    weight[*adj_edge_it];
			source_heap.changeKey(op_node);

			// debug:
			// source_heap.print_data_container();

			if (path_set)
			{
			    pred[op_node] = *adj_edge_it;
			}

			if ((target_mark[op_node] == grey) ||
			    (target_mark[op_node] == white))
			{
			    if (max_dist > source_dist[op_node] +
				target_dist[op_node])
			    {
				max_dist = source_dist[op_node] +
				    target_dist[op_node];
			    }
			}
		    }
		}
		else    // (source_mark[op_node] == white)
		{
		    // nothing to do: shortest distance to op_node is
		    //		      already computed
		}
	    }
	}
	else	// (source_dist[source_heap.top()] >
		//  target_dist[target_heap.top()])
	{

	    // debug:
	    // target_heap.print_data_container();

	    node cur_node = target_heap.top();
	    target_heap.pop();

	    // debug:
	    // target_heap.print_data_container();

	    target_mark[cur_node] = white;

	    if ((source_mark[cur_node] == white) &&
		(max_dist == source_dist[cur_node] +
		    target_dist[cur_node]))
	    {
		fill_node_edge_lists(cur_node);
		break;
	    }

	    if (G.is_directed())
	    {
		node::in_edges_iterator in_edge_it;
		node::in_edges_iterator in_edges_end = 
		    cur_node.in_edges_end();
		for (in_edge_it = cur_node.in_edges_begin();
		     in_edge_it != in_edges_end;
		     ++in_edge_it)
		{
		    node op_node = in_edge_it->opposite(cur_node);
		    if (target_mark[op_node] == black)
		    {
			target_mark[op_node] = grey;
			target_dist[op_node] = target_dist[cur_node] +
			    weight[*in_edge_it];
			target_heap.push(op_node);

			// debug:
			// target_heap.print_data_container();

			if (path_set)
			{
			    succ[op_node] = *in_edge_it;
			}

    			if ((source_mark[op_node] == grey) ||
			    (source_mark[op_node] == white))
			{
			    if (max_dist > source_dist[op_node] +
				target_dist[op_node])
			    {
				max_dist = source_dist[op_node] +
				    target_dist[op_node];
			    }
			}
		    }
		    else if (target_mark[op_node] == grey)
		    {
			if (target_dist[op_node] > target_dist[cur_node] +
			    weight[*in_edge_it])
			{
			    target_dist[op_node] = target_dist[cur_node] +
				weight[*in_edge_it];
			    target_heap.changeKey(op_node);

			    // debug:
			    // target_heap.print_data_container();

			    if (path_set)
			    {
				succ[op_node] = *in_edge_it;
			    }

        			if ((source_mark[op_node] == grey) ||
				    (source_mark[op_node] == white))
				{
				    if (max_dist > source_dist[op_node] +
				    target_dist[op_node])
				{
				    max_dist = source_dist[op_node] +
					target_dist[op_node];
				}
			    }
			}
		    }
		    else    // (target_mark[op_node] == white)
		    {
			// nothing to do: shortest distance to op_node is
			//		  already computed
		    }
		}
	    }
	    else    // (G.is_undirected())
	    {
		node::adj_edges_iterator adj_edge_it;
		node::adj_edges_iterator adj_edges_end =
		    cur_node.adj_edges_end();
		for (adj_edge_it = cur_node.adj_edges_begin();
		     adj_edge_it != adj_edges_end;
		     ++adj_edge_it)
		{
		    node op_node = adj_edge_it->opposite(cur_node);
		    if (target_mark[op_node] == black)
		    {
			target_mark[op_node] = grey;
			target_dist[op_node] = target_dist[cur_node] +
			    weight[*adj_edge_it];
			target_heap.push(op_node);

			// debug:
			// target_heap.print_data_container();

			if (path_set)
			{
			    succ[op_node] = *adj_edge_it;
			}

    			if ((source_mark[op_node] == grey) ||
			    (source_mark[op_node] == white))
			{
			    if (max_dist > source_dist[op_node] +
				target_dist[op_node])
			    {
				max_dist = source_dist[op_node] +
				    target_dist[op_node];
			    }
			}
		    }
		    else if (target_mark[op_node] == grey)
		    {
			if (target_dist[op_node] > target_dist[cur_node] +
			    weight[*adj_edge_it])
			{
			    target_dist[op_node] = target_dist[cur_node] +
				weight[*adj_edge_it];
			    target_heap.changeKey(op_node);

			    // debug:
			    // target_heap.print_data_container();

			    if (path_set)
			    {
				succ[op_node] = *adj_edge_it;
			    }

    			    if ((source_mark[op_node] == grey) ||
				(source_mark[op_node] == white))
			    {
				if (max_dist > source_dist[op_node] +
				    target_dist[op_node])
				{
				    max_dist = source_dist[op_node] +
					target_dist[op_node];
				}
			    }
			}
		    }
		    else    // (target_mark[op_node] == white)
		    {
			// nothing to do: shortest distance to op_node is
			//		  already computed
		    }
		}
	    }
	}
    }

    return GTL_OK;
}


node bid_dijkstra::source() const
{
    return s;
}


node bid_dijkstra::target() const
{
    return t;
}


bool bid_dijkstra::store_path() const
{
    return path_set;
}


bool bid_dijkstra::reached() const
{
    return reached_t;
}


double bid_dijkstra::distance() const
{
    return dist;
}


bid_dijkstra::shortest_path_node_iterator
bid_dijkstra::shortest_path_nodes_begin()
{
    assert(path_set);
    return shortest_path_node_list.begin();
}


bid_dijkstra::shortest_path_node_iterator
bid_dijkstra::shortest_path_nodes_end()
{
    assert(path_set);
    return shortest_path_node_list.end();
}


bid_dijkstra::shortest_path_edge_iterator
bid_dijkstra::shortest_path_edges_begin()
{
    assert(path_set);
    return shortest_path_edge_list.begin();
}


bid_dijkstra::shortest_path_edge_iterator
bid_dijkstra::shortest_path_edges_end()
{
    assert(path_set);
    return shortest_path_edge_list.end();
}


void bid_dijkstra::reset()
{
    reset_algorithm();
}


void bid_dijkstra::reset_algorithm()
{
    s = node();
    t = node();
    weights_set = false;
    path_set = false;
    dist = -1.0;
    reached_t = false;
}


void bid_dijkstra::init(graph& G)
{
    source_dist.init(G, -1.0);
    source_mark.init(G, black);
    target_dist.init(G, -1.0);
    target_mark.init(G, black);
    
    if (path_set)
    {
	pred.init(G, edge());
	succ.init(G, edge());
	shortest_path_node_list.clear();
	shortest_path_edge_list.clear();
    }
}


void bid_dijkstra::fill_node_edge_lists(const node& n)
{
    reached_t = true;
    if (t == s)
    {
	return;
    }
    dist = source_dist[n] + target_dist[n];
    if (path_set)
    {
	node cur_node;
	edge cur_edge;

	cur_node = n;
	cur_edge = pred[cur_node];
	while (cur_edge != edge())
	{
	    shortest_path_edge_list.push_front(cur_edge);
	    cur_node = cur_edge.opposite(cur_node);
	    cur_edge = pred[cur_node];
	    shortest_path_node_list.push_front(cur_node);
	}
	shortest_path_node_list.push_back(n);
	cur_node = n;
	cur_edge = succ[cur_node];
	while (cur_edge != edge())
	{
	    shortest_path_edge_list.push_back(cur_edge);
	    cur_node = cur_edge.opposite(cur_node);
	    cur_edge = succ[cur_node];
	    shortest_path_node_list.push_back(cur_node);
	}
    }
}

__GTL_END_NAMESPACE

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
