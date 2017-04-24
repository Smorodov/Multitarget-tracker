/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   bfs.cpp
//
//==========================================================================
// $Id: bfs.cpp,v 1.11 2001/11/07 13:58:09 pick Exp $

#include <GTL/bfs.h>
#include <GTL/edge_map.h>

__GTL_BEGIN_NAMESPACE

//--------------------------------------------------------------------------
//   Con-/Destructors
//--------------------------------------------------------------------------

bfs::bfs () : algorithm () 
{
    level_number = 0;
    preds = 0;
    non_tree = 0;
    act_bfs_num = 1;
    reached_nodes = 0;
    whole_graph = false;
}

bfs::~bfs () 
{
    if (level_number) delete level_number;
    if (preds) delete preds;
    if (non_tree) delete non_tree;
}


//--------------------------------------------------------------------------
//   Parameters
//--------------------------------------------------------------------------

void bfs::calc_level (bool set) 
{
    if (set && !level_number) {
	level_number = new node_map<int>;
    } else if (!set && level_number) {
	delete level_number;
	level_number = 0;
    }
}

void bfs::store_preds (bool set)
{
    if (set && !preds) {
	preds = new node_map<node>;
    } else if (!set && preds) {
	delete preds;
	preds = 0;
    }
}

void bfs::store_non_tree_edges (bool set) 
{
    if (set && !non_tree)
	{
		non_tree = new edges_t;
    }
	else if (!set && non_tree)
	{
		delete non_tree;
		non_tree = 0;
    }
}

//--------------------------------------------------------------------------
//   GTL_Algorithm - Interface
//--------------------------------------------------------------------------

void bfs::reset () 
{
    act_bfs_num = 1;
    tree.erase (tree.begin(), tree.end());
    bfs_order.erase (bfs_order.begin(), bfs_order.end());
    roots.erase (roots.begin(), roots.end());
    reached_nodes = 0;
    if (non_tree) {
	non_tree->erase (non_tree->begin(), non_tree->end());
    }
}


int bfs::run (graph& G) {
    
    bfs_number.init (G, 0);

    if (level_number) {
	level_number->init (G);
    }

    if (preds) {
	preds->init (G, node());
    }

    edge_map<int> *used = 0;

    if (non_tree) {
	used = new edge_map<int> (G, 0);
    }

    init_handler (G);

    //
    // Set start-node 
    // 

    if (start == node()) {
	start = G.choose_node();
    }

    new_start_handler (G, start);

    bfs_sub (G, start, used);

    node curr;

    if (whole_graph && reached_nodes < G.number_of_nodes()) {
	forall_nodes (curr, G) {
	    if (bfs_number[curr] == 0) {

		new_start_handler (G, curr);

		bfs_sub (G, curr, used);
	    }
	}
    }

    if (non_tree) {
	delete used;
    }

    end_handler (G);

    return 1;
}



//--------------------------------------------------------------------------
//   PRIVATE
//--------------------------------------------------------------------------


void bfs::bfs_sub (graph& G, const node& st, edge_map<int>* used) 
{
    qu.push_back (st);
    bfs_number[st] = act_bfs_num;
    ++act_bfs_num;

    if (level_number) {
	(*level_number)[st] = 0;
    }

    while (!qu.empty()) {
	node tmp = qu.front();
	qu.pop_front();
	++reached_nodes;
	
	if (tmp == st) {
	    roots.push_back (bfs_order.insert (bfs_order.end(), tmp));
	} else {
	    bfs_order.push_back (tmp);
	}

	popped_node_handler (G, tmp);

	node::adj_edges_iterator it = tmp.adj_edges_begin();
	node::adj_edges_iterator end = tmp.adj_edges_end();
	
	for (; it != end; ++it) {
	    edge curr = *it;
	    node opp = tmp.opposite (curr);

	    if (bfs_number[opp] == 0) {
		bfs_number[opp] = act_bfs_num;
		++act_bfs_num;
		tree.push_back (curr);
		
		if (non_tree) {
		    (*used)[curr] = 1;
		}

		if (level_number) {
		    (*level_number)[opp] = (*level_number)[tmp] + 1;
		}
	
		if (preds) {
		    (*preds)[opp] = tmp;
		}

		qu.push_back (opp);

		unused_node_handler (G, opp, tmp);

	    } else {
		if (non_tree && !(*used)[curr]) {
		    (*used)[curr] = 1;
		    non_tree->push_back(curr);
		}

		used_node_handler (G, opp, tmp);
	    }
	}

	finished_node_handler (G, tmp);
    }			
}
		
__GTL_END_NAMESPACE

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
