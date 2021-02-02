// $Id: mytree.cpp,v 1.2 2005/08/16 12:22:52 rdmp1c Exp $


#include "mytree.h"

#include <vector>
#include <stack>

bool is_tree (const GTL::graph& G)
{ 
	GTL::node v;
	forall_nodes(v,G)
    if ( v.indeg () > 1 ) return false; // nonunique parent
  return ( G.number_of_nodes() == G.number_of_edges() + 1
    && G.is_connected() );
}

bool MyTree::is_root( const GTL::node v ) const
{
	return (v.indeg() == 0);
}

bool MyTree::is_leaf( const GTL::node v ) const
{
	return (v.outdeg() == 0);
}

GTL::node MyTree::parent( const GTL::node v ) const
{
	if (v.indeg() == 0) return v;
	GTL::edge e = (*v.in_edges_begin());
	return e.source();
}

GTL::node MyTree::root() const
{
	GTL::node v = (*nodes_begin());
	while (!is_root(v))
		v = parent(v);
	return v;
}

void MyTree::postorder_traversal()
{ 
	std::stack < GTL::node, std::vector<GTL::node> > S;
	S.push (root());
	int num = 1;
	do {
		GTL::node v = std::move(S.top());
		S.pop();
		int n = number_of_nodes() - num++ + 1; // order in which node is visited in postorder
		order[v] = n; 
		number[n] = v;

//		cout << label[v] << " " << order[v] << endl;
		
		// store info about order here...
		
		
		
		GTL::node::adj_nodes_iterator it = v.adj_nodes_begin();
		GTL::node::adj_nodes_iterator end = v.adj_nodes_end();
		while (it != end)
		{
			S.push (*it);
			it++;
		}
	} while ( !S.empty() );
}



GTL::node MyTree::get_left_child(const GTL::node v) const
{
	return (*(v.adj_nodes_begin()));
}

GTL::node MyTree::get_right_child(const GTL::node v) const
{
	GTL::node right;
	GTL::node::adj_nodes_iterator it = v.adj_nodes_begin();
	GTL::node::adj_nodes_iterator end = v.adj_nodes_end();
	while (it != end)
	{
		right = *it;
		it++;
	}
	return right;
}

