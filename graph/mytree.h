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
bool is_tree (const graph& G);

class MyTree : public MyGraph
{
public:
	MyTree () { };

	node parent( const node v ) const;
	node root() const;
	
	bool is_root( const node v ) const;
	bool is_leaf( const node v ) const;

	node get_left_child(const node v) const;
	node get_right_child(const node v) const;

	void postorder_traversal();
	
	int postorder (const node v) const { return order[v]; };
	
protected:
	node_map<int> order;
	std::map <int, node, std::less <int> > number;
};



#endif

