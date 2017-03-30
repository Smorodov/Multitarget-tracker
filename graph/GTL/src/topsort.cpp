/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   topsort.cpp
//
//==========================================================================
// $Id: topsort.cpp,v 1.7 2001/11/07 13:58:12 pick Exp $

#include <GTL/topsort.h>

__GTL_BEGIN_NAMESPACE

//--------------------------------------------------------------------------
//   algorithm - interface
//--------------------------------------------------------------------------


void topsort::reset () 
{
    dfs::reset();
    acyclic = true;
    top_order.erase (top_order.begin(), top_order.end());;
}

int topsort::check (graph& G) 
{
    return G.is_directed() ? GTL_OK : GTL_ERROR;
}



//--------------------------------------------------------------------------
//   Handler
//--------------------------------------------------------------------------


void topsort::init_handler (graph& G) 
{
    top_numbers.init (G, 0);
    act_top_num = G.number_of_nodes();
}


void topsort::leave_handler (graph& /*G*/, node& n, node& /*f*/) 
{
    top_numbers[n] = act_top_num;
    act_top_num--;
    top_order.push_front (n);
}


void topsort::old_adj_node_handler (graph& /*G*/, edge& /*adj*/, node& opp)
{
    if (top_numbers[opp] == 0) {
	acyclic = false;
    }
}

__GTL_END_NAMESPACE

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
