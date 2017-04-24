/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   components.cpp
//
//==========================================================================
// $Id: components.cpp,v 1.5 2001/11/07 13:58:09 pick Exp $

#include <GTL/components.h>

__GTL_BEGIN_NAMESPACE

components::components () : dfs ()
{
    scan_whole_graph (true);
    num_of_components = 0;
}

void components::reset () 
{ 
    dfs::reset ();
    comp.erase (comp.begin(), comp.end());
    num_of_components = 0;
}

int components::check (graph& G) 
{
    return G.is_undirected() && whole_graph && 
	dfs::check (G) == GTL_OK ? GTL_OK : GTL_ERROR;
}
    

//--------------------------------------------------------------------------
//   Handler
//--------------------------------------------------------------------------


void components::new_start_handler (graph& /*G*/, node& st) 
{
	li = comp.insert(comp.end(), std::pair<nodes_t, edges_t>(nodes_t(), edges_t()));
    li->first.push_back(st);
    ++num_of_components;
}

void components::before_recursive_call_handler (graph& /*G*/, edge& /*e*/, node& n)
{
    li->first.push_back(n);
    // li->second.push_back(e);    
}


void components::old_adj_node_handler (graph& /*G*/, edge& e, node& n) 
{
    node curr = n.opposite (e);

    //
    // Store backedges at lower endpoint
    //

    if (dfs_num (curr) > dfs_num (n)) { 
	li->second.push_back (e);    
    }
}


__GTL_END_NAMESPACE

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
