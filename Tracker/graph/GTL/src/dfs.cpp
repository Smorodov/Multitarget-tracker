/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   dfs.cpp
//
//==========================================================================
// $Id: dfs.cpp,v 1.18 2001/11/07 13:58:09 pick Exp $

#include <GTL/dfs.h>
#include <GTL/edge_map.h>

__GTL_BEGIN_NAMESPACE

//--------------------------------------------------------------------------
//   Con-/Destructors
//--------------------------------------------------------------------------

dfs::dfs () : algorithm () 
{
    act_dfs_num = 1;
    act_comp_num = 1;
    reached_nodes = 0;
    whole_graph = false;
    comp_number = 0;
    preds = 0;
    used = 0;
    back_edges = 0;
}

dfs::~dfs ()
{
    if (comp_number) delete comp_number;
    if (preds) delete preds;
    if (back_edges) {
	delete back_edges;
	delete used;
    } 
}

//--------------------------------------------------------------------------
//   GTL_Algorithm - Interface
//--------------------------------------------------------------------------


void dfs::reset () 
{
    act_dfs_num = 1;
    act_comp_num = 1;
    reached_nodes = 0;
    tree.erase (tree.begin(), tree.end());
    dfs_order.erase (dfs_order.begin(), dfs_order.end());
    roots.erase (roots.begin(), roots.end());
    start = node();

    if (back_edges) {
	back_edges->erase (back_edges->begin(), back_edges->end());
    }
}


int dfs::check (graph& /*G*/) 
{
    return GTL_OK;
}

int dfs::run (graph& G)
{
    //
    // initialization
    // 

    node curr;
    node dummy;
    
    dfs_number.init (G, 0);
    
    if (comp_number) {
	comp_number->init (G);
    }

    if (preds) {
	preds->init (G, node());
    }

    if (back_edges) {
	used = new edge_map<int> (G, 0);
    }

    init_handler (G);

    //
    // Set start-node 
    // 

    if (G.number_of_nodes() == 0) {
	return GTL_OK;
    }

    if (start == node()) {
	start = G.choose_node();
    } 
	
    new_start_handler (G, start);
    
    dfs_sub (G, start, dummy);

    if (whole_graph && reached_nodes < G.number_of_nodes()) {

	//
	// Continue DFS with next unused node.
	//

	forall_nodes (curr, G) {
	    if (dfs_number[curr] == 0) {
		new_start_handler (G, curr);
		dfs_sub (G, curr, dummy);
	    }
	}
    }    
    
    if (back_edges) {
	delete used;
	used = 0;
    }

    end_handler(G);

    return GTL_OK;
}    


//--------------------------------------------------------------------------
//   PRIVATE 
//--------------------------------------------------------------------------


void dfs::dfs_sub (graph& G, node& curr, node& father) 
{
    node opp;
    edge adj;
    
    if (father == node()) {	
	roots.push_back (dfs_order.insert (dfs_order.end(), curr));
    } else {
	dfs_order.push_back (curr);    
    }

    dfs_number[curr] = act_dfs_num;
    reached_nodes++;

    if (preds) {
	(*preds)[curr] = father;
    }

    entry_handler (G, curr, father);

    ++act_dfs_num;
    node::adj_edges_iterator it = curr.adj_edges_begin();
    node::adj_edges_iterator end = curr.adj_edges_end();
    
    while (it != end) {
	adj = *it;
	opp = curr.opposite(adj);

	if (dfs_number[opp] == 0) {	    
	    tree.push_back (adj);

	    if (back_edges) {
		(*used)[adj] = 1;
	    }

	    before_recursive_call_handler (G, adj, opp);
	    dfs_sub (G, opp, curr);
	    after_recursive_call_handler (G, adj, opp);

	} else {
	    if (back_edges && !(*used)[adj]) {
		(*used)[adj] = 1;
		back_edges->push_back (adj);
	    }

	    old_adj_node_handler (G, adj, opp);
	}

	++it;
    }

    leave_handler (G, curr, father);

    if (comp_number) {
	(*comp_number)[curr] = act_comp_num;
	++act_comp_num;
    }
}

//--------------------------------------------------------------------------
//   Parameters
//--------------------------------------------------------------------------

void dfs::calc_comp_num (bool set) 
{
    if (set && !comp_number) {
	comp_number = new node_map<int>;
    } else if (!set && comp_number) {
	delete comp_number;
	comp_number = 0;
    }
}

void dfs::store_preds (bool set)
{
    if (set && !preds) {
	preds = new node_map<node>;
    } else if (!set && preds) {
	delete preds;
	preds = 0;
    }
}

void dfs::store_non_tree_edges (bool set) 
{
    if (set && !back_edges)
	{
		back_edges = new edges_t;
    }
	else if (!set && back_edges)
	{
		delete back_edges;
		back_edges = 0;
    }
}
    
__GTL_END_NAMESPACE

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
