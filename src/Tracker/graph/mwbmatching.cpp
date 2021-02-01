// $Id: mwbmatching.cpp,v 1.3 2007/10/28 08:47:20 rdmp1c Exp $

#include "mwbmatching.h"


#include <cstdlib>
#include <cassert>
#include <iostream>
#include <queue>
#include <list>
#include <map>
#include <set>
#include <stack>

#ifdef __GNUC__
#include <algorithm>
#endif

#ifdef __BORLANDC__
	#include <values.h>
#endif
#if (defined __MWERKS__) || (defined __GNUC__)
	#include <climits>
	#define MAXINT INT_MAX
#endif

mwbmatching::mwbmatching ()
    : algorithm()
{
}

void mwbmatching::set_vars(const GTL::edge_map<int>& edge_weight)
{
    m_edge_weight = edge_weight;
	m_mwbm = 0;
    m_set_vars_executed = true;
}

int mwbmatching::check (GTL::graph& G) 
{
	if (!m_set_vars_executed)
		return(GTL_ERROR);

	if ((G.number_of_nodes() <= 1) || (!G.is_connected()) || (!G.is_directed()))
		return(GTL_ERROR);

    return GTL_OK;
}

int mwbmatching::run(GTL::graph& G)
{
	// Initialise
	m_pot.init (G, 0);
	m_free.init (G, true);
	m_dist.init (G, 0);
	
	GTL::nodes_t A;
	GTL::nodes_t B;
	GTL::node n;
	// Partition graph based on direction of edges
	forall_nodes (n, G)
	{
		if (n.outdeg() == 0)
			B.push_back (n);
		else
			A.push_back(n);
		
		m_node_from_id[n.id()] = n;
	}
	
	// Simple heuristic
	int C = 0;
	GTL::edge e;
	forall_edges (e, G)
	{
		m_edge_from_id[e.id()] = e;
		if (m_edge_weight[e] > C) 
			C = m_edge_weight[e];	
	}

	GTL::nodes_t::iterator it = A.begin();
	GTL::nodes_t::iterator end = A.end();
	while (it != end)
	{
		m_pot[*it] = C;
		it++;
	}
	
	it = A.begin();
	
	while (it != end)
	{
		if (m_free[*it])
			augment (G, *it);
		it++;
	}
	
	// Get edges in matching
	it = B.begin();
	end = B.end();
	while (it != end)
	{
		forall_out_edges (e, *it)
		{
			m_result.push_back (e);	
			m_mwbm += m_edge_weight[e];
		}
		it++;
	}
    return(GTL_OK);
}

int mwbmatching::augment(GTL::graph& G, GTL::node a)
{
	// Initialise
	m_pred.init(G, -1);
	m_pq = fh_alloc(G.number_of_nodes());
	
	m_dist[a] = 0;
	
	std::stack<GTL::node, std::vector<GTL::node> > RA;
	RA.push(a);
	std::stack<GTL::node, std::vector<GTL::node> > RB;
	
	GTL::node a1 = a;
	GTL::edge e;
	
	// Relax
	forall_adj_edges (e, a1)
	{
		const GTL::node& b = e.target_();
		long db = m_dist[a1] + (m_pot[a1] + m_pot[b] - m_edge_weight[e]);
		if (m_pred[b] == -1)
		{
			m_dist[b] = db;
			m_pred[b] = e.id();
			RB.push(b);
			
			fh_insert (m_pq, b.id(), db);	
		}
		else
		{
			if (db < m_dist[b])
			{
				m_dist[b] = db;
				m_pred[b] = e.id();

				fh_decrease_key (m_pq, b.id(), db);  
			}
		}
	}

	GTL::node best_node_in_A = a;
	long minA = m_pot[a];
	long delta = 0;
	for (;;)
	{
		// Find node with minimum distance db
		int node_id = -1;
		long db = 0;
		if (m_pq->n != 0)
		{
			node_id = fh_delete_min (m_pq);
			db = m_dist[m_node_from_id[node_id]];
		}
		
		if (node_id == -1 || db >= minA)
		{
			delta = minA;
			// augmentation by best node in A
      		augment_path_to (G, best_node_in_A);  
      		m_free[a] = false; 
      		m_free[best_node_in_A] = true; 
			break;
		}
		else
		{
			GTL::node b = m_node_from_id[node_id];
			if (m_free[b])
			{
				delta = db;
				// augmentation by path to b, so a and b are now matched
				augment_path_to (G, b);
        		m_free[a] = false;
        		m_free[b] = false;
		        break;
			}
			else
			{
				// continue shortest path computation
				e = (*b.adj_edges_begin());
				const GTL::node& a2 = e.target_();
				m_pred[a2] = e.id(); 
				RA.push(a2);
				m_dist[a2] = db; 
					  
				if (db + m_pot[a2] < minA)
				{ 
					best_node_in_A = a2;
				  	minA = db + m_pot[a2];
				}
				
				// Relax
				forall_adj_edges (e, a2)
				{
					const GTL::node& b1 = e.target_();
					long db1 = m_dist[a2] + (m_pot[a2] + m_pot[b1] - m_edge_weight[e]);
					if (m_pred[b1] == -1)
					{
						m_dist[b1] = db1;
						m_pred[b1] = e.id();
						RB.push(b1);
						
						fh_insert (m_pq, b1.id(), db1);
					}
					else
					{
						if (db1 < m_dist[b1])
						{
							m_dist[b1] = db1;
							m_pred[b1] = e.id();

							fh_decrease_key (m_pq, b1.id(), db1);  
						}
					}
				}							
			}
		}
	}
	
	while (!RA.empty())
	{
		GTL::node a_node = std::move(RA.top());
		RA.pop();
		m_pred[a_node] = -1;
		long pot_change = delta - m_dist[a_node];
		if (pot_change <= 0)
			continue;
		m_pot[a_node] = m_pot[a_node] - pot_change;		
	}
	while (!RB.empty())
	{
		GTL::node b_node = std::move(RB.top());
		RB.pop();
		m_pred[b_node] = -1;
		
		long pot_change = delta - m_dist[b_node];
		if (pot_change <= 0)
			continue;
		m_pot[b_node] = m_pot[b_node] + pot_change;
	}
	
	// Clean up
	fh_free(m_pq);
	return 0;
}

void mwbmatching::augment_path_to (GTL::graph &/*G*/, GTL::node v)
{
	auto i = m_pred[v];
	while (i != -1)
	{
		GTL::edge e = m_edge_from_id[i];
		e.reverse();
		i = m_pred[e.target()];
	}
}

GTL::edges_t MAX_WEIGHT_BIPARTITE_MATCHING(GTL::graph &G, GTL::edge_map<int> weights)
{
	GTL::edges_t L;

	mwbmatching mwbm;
	mwbm.set_vars(weights);
	
	//if (mwbm.check(G) != algorithm::GTL_OK)
	//{
	//	cout << "Maximum weight bipartite matching algorithm check failed" << endl;
		//exit(1);
	//}
	//else
	{
		if (mwbm.run(G) != GTL::algorithm::GTL_OK)
			std::cout << "Error running maximum weight bipartite matching algorithm" << std::endl;
		else
			L = mwbm.get_match();
	}
	return L;
}
