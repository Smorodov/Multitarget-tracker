/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   min_tree.cpp
//
//==========================================================================
// $Id: min_tree.cpp,v 1.4 2001/11/07 13:58:10 pick Exp $

#include <GTL/graph.h>
#include <stack>
#include <queue>
#include <GTL/min_tree.h>

__GTL_BEGIN_NAMESPACE

min_tree::min_tree () { 
    is_set_distances = false;
    weight = 0;
}

int min_tree::check (graph& g) { 
    if (g.is_directed()) return GTL_ERROR;
    else if (g.number_of_nodes() < 2) return GTL_ERROR;
    else if (!g.is_connected()) return GTL_ERROR;
    else if (!is_set_distances) return GTL_ERROR;
    else return GTL_OK;
}

void min_tree::set_distances (const edge_map<int>& dist) { 
    this->dist = dist;
    is_set_distances = true;
}

std::set<edge> min_tree::get_min_tree() {
    return this->tree;
}
 
int min_tree::get_min_tree_length() { 
    int sum;
	std::set<edge>::iterator tree_it;
 
    sum = 0;

    for (tree_it = tree.begin(); tree_it != tree.end(); tree_it++)
	sum += dist[*tree_it];

    return sum;
}
 
int min_tree::run (graph& g) { 
	std::priority_queue <TSP_A_VALUE, std::vector<TSP_A_VALUE>, input_comp> node_distances;
    node::adj_edges_iterator adj_it, adj_end;
	std::set<node> tree_nodes;
	std::set<node>::iterator tree_it;
    edge curr;
    node new_node;
    graph::edge_iterator edge_it, edges_end;
    unsigned int number_of_nodes;
    int min_dist;


    // making out the start edge

    edge_it = g.edges_begin();
    edges_end = g.edges_end();

    curr = *edge_it;
    min_dist = dist[*edge_it];

    for (; edge_it != edges_end; edge_it++) { 
	if (dist[*edge_it] < min_dist) { 
	    curr = *edge_it;
	    min_dist = dist[*edge_it];
	}
    }

    tree.insert(curr);
 
    tree_nodes.insert(curr.source());
    tree_nodes.insert(curr.target());


    for (tree_it = tree_nodes.begin(); tree_it != tree_nodes.end(); tree_it++) { 
	adj_it = tree_it->adj_edges_begin();
	adj_end = tree_it->adj_edges_end();

	for (; adj_it != adj_end; adj_it++) {  
	    node_distances.push(TSP_A_VALUE(dist[*adj_it], adj_it));
	}  
    }

    // create the min_tree

    number_of_nodes = g.number_of_nodes();

    while(tree.size() < number_of_nodes - 1) { 
	curr = *((node_distances.top()).second);
  
	node_distances.pop();
 
	if (tree_nodes.find(curr.source()) != tree_nodes.end() && 
	    tree_nodes.find(curr.target()) != tree_nodes.end()) {
	} else {
	    tree.insert(curr);
	    weight += dist[curr];

	    if (tree_nodes.find(curr.source()) != tree_nodes.end()) { 
		new_node = curr.target();
	    } else { 
		new_node = curr.source();
	    }

	    tree_nodes.insert(new_node);
	    
	    adj_it = new_node.adj_edges_begin();
	    adj_end = new_node.adj_edges_end();
	    
	    for (; adj_it != adj_end; adj_it++) { 
		node_distances.push(TSP_A_VALUE(dist[*adj_it], adj_it));
	    }   
	}
    }

    return GTL_OK;
}

void min_tree::reset() {
    tree.erase (tree.begin(), tree.end());
    weight = 0;
}
    
__GTL_END_NAMESPACE           









