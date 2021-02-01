// $Id: mytree.h,v 1.2 2005/08/16 12:22:53 rdmp1c Exp $

#ifndef MYTREE_H
#define MYTREE_H


/* This code is heavily based on the code provided by Gabriel Valiente for
   his book "Algorithms on Trees and Graphs" (Berlin: Springer-Verlag, 2002).
   I've modified it to call GTL rather than LEDA functions.
*/



/*
  To do:
  
  LCA functions
  Preorder
  Inorder
  Visitation numbers
  Bottom up
  Height
  Depth
  etc.
  
*/

#include "mygraph.h"


// Test whether graph is a tree
bool is_tree (const GTL::graph& G);

class MyTree final : public MyGraph
{
public:
	MyTree () = default;

	GTL::node parent( const GTL::node v ) const;
	GTL::node root() const;
	
	bool is_root( const GTL::node v ) const;
	bool is_leaf( const GTL::node v ) const;

	GTL::node get_left_child(const GTL::node v) const;
	GTL::node get_right_child(const GTL::node v) const;

	void postorder_traversal();
	
	int postorder (const GTL::node v) const { return order[v]; };
	
protected:
	GTL::node_map<int> order;
	std::map <int, GTL::node, std::less <int> > number;
};



#endif

