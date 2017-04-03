// $Id: mytree.cpp,v 1.2 2005/08/16 12:22:52 rdmp1c Exp $


#include "mytree.h"

#include <vector>
#include <stack>

bool is_tree (const graph& G)
{ 
  node v;
  forall_nodes(v,G)
    if ( v.indeg () > 1 ) return false; // nonunique parent
  return ( G.number_of_nodes() == G.number_of_edges() + 1
    && G.is_connected() );
}

bool MyTree::is_root( const node v ) const
{
	return (v.indeg() == 0);
}

bool MyTree::is_leaf( const node v ) const 
{
	return (v.outdeg() == 0);
}

node MyTree::parent( const node v ) const 
{
	if (v.indeg() == 0) return v;
	edge e = (*v.in_edges_begin());
	return e.source();
}

node MyTree::root() const 
{
	node v = (*nodes_begin());
	while (!is_root(v))
		v = parent(v);
	return v;
}

void MyTree::postorder_traversal()
{ 
	std::stack < node, std::vector<node> > S;
	S.push (root());
	int num = 1;
	do {
		node v = S.top();
		S.pop();
		int n = number_of_nodes() - num++ + 1; // order in which node is visited in postorder
		order[v] = n; 
		number[n] = v;

//		cout << label[v] << " " << order[v] << endl;
		
		// store info about order here...
		
		
		
		node::adj_nodes_iterator it = v.adj_nodes_begin();
		node::adj_nodes_iterator end = v.adj_nodes_end();
		while (it != end)
		{
			S.push (*it);
			it++;
		}
	} while ( !S.empty() );
}



node MyTree::get_left_child(const node v) const 
{
	return (*(v.adj_nodes_begin()));
}

node MyTree::get_right_child(const node v) const 
{
	node right;
	node::adj_nodes_iterator it = v.adj_nodes_begin();
	node::adj_nodes_iterator end = v.adj_nodes_end();
	while (it != end)
	{
		right = *it;
		it++;
	}
	return right;
}

