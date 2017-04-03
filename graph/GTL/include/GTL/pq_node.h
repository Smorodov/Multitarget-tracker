/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   pq_node.h
//
//==========================================================================
// $Id: pq_node.h,v 1.15 2003/04/03 11:48:26 raitner Exp $

#ifndef PQ_NODE_H
#define PQ_NODE_H

#include <GTL/GTL.h>
#include <GTL/symlist.h>
#include <GTL/graph.h>

#include <list>
#include <iostream>

__GTL_BEGIN_NAMESPACE

class pq_tree;
class p_node;
class q_node; 
class pq_leaf;
class direction_indicator;

/**
 * @internal
 */
class GTL_EXTERN pq_node 
{
protected:
    /**
     * @internal
     */
    typedef symlist<pq_node*>::iterator iterator;

    /**
     * @internal
     */
    enum PQ_KIND {P_NODE, Q_NODE, LEAF, DIR};

    /**
     * @internal
     */
    enum PQ_MARK {UNMARKED, QUEUED, BLOCKED, UNBLOCKED};

    /**
     * @internal
     */
    pq_node (node n_, int id_) :  pert_children(0),
				  pert_leaves(0),
				  mark (UNMARKED),
				  n (n_),
				  id (id_)
    {
    }

    /**
     * @internal
     */
    virtual ~pq_node ();

    /**
     * @internal
     * Used to identify nodes.
     */
    virtual PQ_KIND kind() const = 0;

    /**
     * @internal
     * Called whenever a son is known to be partial during reduction phase.
     */
    virtual void partial(iterator)
    {
    }

    /**
     * @internal
     * Called whenever a son is known to be full during reduction phase.
     */
    virtual void full(iterator)
    {
    }

    /**
     * @internal
     * Used to write a description of this node into a stream.
     */
	virtual void write(std::ostream&, int) = 0;

    /**
     * @internal
     * Reset node for next reduction.
     */
    virtual void clear()
    {
	mark = UNMARKED;
	pert_leaves = 0;
	pert_children = 0;
    }

    // type-casts 

    /**
     * @internal
     * Interface type-cast to P-node.
     */
    virtual p_node* P() = 0;

    /**
     * @internal
     * Interface type-cast to Q-node.
     */
    virtual q_node* Q() = 0;

    /**
     * @internal
     * Interface type-cast to direction indicator.
     */
    virtual direction_indicator* D() = 0;

    /**
     * @internal
     * Interface type-cast to PQ-leaf.
     */
    virtual pq_leaf* L() = 0;

    //
    // Data used in reductions
    //

    /**
     * @internal
     * Number of pertinent children; is calculated during bubble-up phase
     * and gets decreased whenever a pertinent child is matched in reduction
     * phase, such that it can be assured that this node is matched @em
     * after all its pertinent children were correctly matched.
     */
    int pert_children;

    /**
     * @internal
     * Number of pertinent leaves in the subtree rooted at this node; is 
     * calculated in the reduction phase and is used to determine the root 
     * of the pertinent subtree, i.e. the last node for template matchings. 
     */
    int pert_leaves;

    /**
     * @internal
     * For Q-nodes it is not acceptable to maintain father pointers for @em 
     * all sons (cf. Booth, Luecker); fortunatly this isn't neccessary and 
     * the father pointer is only valid if is_endmost is true. For the sons
     * of a Q-node is_endmost is only true for the first and the last son.
     * For the sons of P-nodes ths flag is always true.
     */
    bool is_endmost;

    /**
     * @internal
     * The main operations on PQ-trees are performed bottom up so each node
     * should know its father; Because of complexity issuses this isn't
     * always possible and  thus father is valid iff is_endmost is true.
     */
    pq_node* father;

    /**
     * @internal
     * Describes the role this node plays in the reduction at the moment;
     * four states are possible:
     * -# @c UNMARKED: node wasn't touched so far
     * -# @c BLOCKED: during bubble-up phase this node got queued, but as
     *    yet it was not possible to get a valid father pointer
     * -# @c UNBLOCKED: node was touched during bubble-up and it either had
     *    a valid father pointer or one could be borrowed from one of its
     *    siblings
     * -# @c QUEUED: node has been put into the queue
     */
    PQ_MARK mark;

    /**
     * @internal
     * List of sons.
     */
    symlist<pq_node*> sons;

    /**
     * @internal
     * Position in the list of sons of node's father.
     */
    iterator pos;

    /**
     * @internal
     * Position in the list of nodes to be cleared in reset. Each node
     * touched in #bubble-up phase is stored in the list of nodes to be
     * cleared. As they get matched in the reduction phase they are cleared
     * and deleted from this list. But even if the reduction is successful
     * not all nodes touched in the first phase really get matched.
     */
    std::list<pq_node*>::iterator lpos;

    //
    // Application specific data (should become template parameter)
    //

    /**
     * @internal
     * Node of the graph which this PQ-node represents.
     */
    node n;

    /**
     * @internal
     */
    int id;

    /**
     * @internal
     */
    node up;

    /**
     * @internal
     */
    int up_id;

    //
    // Friends
    //

    /**
     * @internal
     * Allow q_node private access.
     */
    friend class q_node;

    /**
     * @internal
     * Allow p_node private access.
     */
    friend class p_node;

    /**
     * @internal
     * Allow my_pq_tree private access.
     */
    friend class pq_tree;

    /**
     * @internal
     * Allow planarity private access.
     */
    friend class planarity;

    /**
     * @internal
     * Allow operator<< private access.
     */
	GTL_EXTERN friend std::ostream& operator<<(std::ostream&, const pq_tree&);
};


/**
 * @internal
 */
class GTL_EXTERN p_node : public pq_node 
{
private:
    /**
     * @internal
     */
    p_node(node, int);

    /**
     * @internal
     */
    p_node(node, int, symlist<pq_node*>&);
    
    //
    // pq_node interface
    // 

    /**
     * @internal
     */
    void partial(iterator);

    /**
     * @internal
     */
    void full(iterator);

    /**
     * @internal
     * Determines kind of this %node.
     */
    PQ_KIND kind () const 
    {
	return P_NODE;
    }

    /**
     * @internal
     * Print this %node in gml format.
     */
	void write(std::ostream&, int);

    /**
     * @internal
     */
    void clear ();

    // type-casts

    /**
     * @internal
     * Type-cast to P-node.
     */
    p_node* P()
    {
	return this;
    }

    /**
     * @internal
     * Type-cast to Q-node.
     */
    q_node* Q()
    {
	assert(false);
	return 0;
    }

    /**
     * @internal
     * Type-cast to direction indicator.
     */
    direction_indicator* D()
    {
	assert(false);
	return 0;
    }

    /**
     * @internal
     * Type-cast to PQ-leaf.
     */
    pq_leaf* L()
    {
	assert(false);
	return 0;
    }

    //
    // Additional
    //

    /**
     * @internal
     * Whenever a child is known to be full, it is moved from the list of 
     * sons to this list.
     */
    symlist<pq_node*> full_sons;

    /**
     * @internal
     * Whenever a child is known to be partial, it is moved from the list of
     * sons to this list.
     */
    symlist<pq_node*> partial_sons;

    /**
     * @internal
     * Number of children.
     */
    int child_count;

    /**
     * @internal
     * Number of partial children.
     */
    int partial_count;

    /**
     * @internal
     * Number of full children.
     */
    int full_count;

    //
    // Friends 
    //

    /**
     * @internal
     * Allow planarity private access.
     */
    friend class planarity;

    /**
     * @internal
     * Allow pq_tree private access.
     */
    friend class pq_tree;

    /**
     * @internal
     * Allow operator<< private access.
     */
	GTL_EXTERN friend std::ostream& operator<<(std::ostream&, const pq_tree&);
};


/**
 * @internal
 */
class GTL_EXTERN q_node : public pq_node 
{
private:
    /**
     * @internal
     */
    q_node (node, int);

    //
    // pq_node interface
    // 

    /**
     * @internal
     */
    void partial(iterator);

    /**
     * @internal
     */
    void full(iterator);

    /**
     * @internal
     * Determines kind of this %node.
     */
    PQ_KIND kind() const
    {
	return Q_NODE;
    }

    /**
     * @internal
     * Print this %node in gml format.
     */
	void write(std::ostream&, int);

    /**
     * @internal
     */
    void clear();

    // type-casts

    /**
     * @internal
     * Type-cast to P-node.
     */
    p_node* P()
    {
	assert (false);
	return 0;
    }

    /**
     * @internal
     * Type-cast to Q-node.
     */
    q_node* Q()
    {
	return this;
    }

    /**
     * @internal
     * Type-cast to direction indicator.
     */
    direction_indicator* D()
    {
	assert (false);
	return 0;
    }

    /**
     * @internal
     * Type-cast to PQ-leaf.
     */
    pq_leaf* L()
    {
	assert (false);
	return 0;
    }

    //
    // Additional
    //

    /**
     * @internal
     * Determines pert_begin and pert_end the first time a full or partial 
     * child is found.
     */
    void pertinent(iterator);
    
    /**
     * @internal
     * In #Q2 and #Q3 matchings the sons of partial children have to be
     * merged into the list of sons of this node at the partial node's
     * position
     */
    q_node* merge (iterator);
    
    /**
     * @internal
     * @em Depreacted.
     */
    void turn (); 

    /**
     * @internal
     * First son full or partial viewed from the beginning of the list of
     * pq_node::sons.
     */
    iterator pert_begin;    

    /**
     * @internal
     * Last son full or partial; usually this is the last son.
     */
    iterator pert_end;

    /**
     * @internal
     * Positions of the partial nodes among the sons. Normally only two
     * partial sons are allowed, but the third one is needed in planarity
     * testing.
     */
    iterator partial_pos[3];

    /**
     * @internal
     * True when all the pertinent children are consecutive; íf false
     * @a pert_begin lies in one block of pertinent children and @a pert_end
     * in another, such that <tt>--pert_end</tt> is empty and between the
     * two blocks.
     */
    bool pert_cons;

    /**
     * @internal
     * Number of partial children.
     */
    int partial_count;

    /**
     * @internal
     * Number of full children.
     */
    int full_count;
    
    //
    // Friends 
    //

    /**
     * @internal
     * Allow planarity private access.
     */
    friend class planarity;

    /**
     * @internal
     * Allow pq_tree private access.
     */
    friend class pq_tree;
};


/**
 * @internal
 */
class GTL_EXTERN pq_leaf : public pq_node 
{
public:
    /**
     * @internal
     */
    pq_leaf (int, int, edge, node);
private:
    /**
     * @internal
     * Determines kind of this %node.
     */
    PQ_KIND kind() const
    {
	return LEAF;
    }

    /**
     * @internal
     * Print this %node in gml format.
     */
	void write(std::ostream&, int);

    // type-casts

    /**
     * @internal
     * Type-cast to P-node.
     */
    p_node* P()
    {
	assert(false);
	return 0;
    }

    /**
     * @internal
     * Type-cast to Q-node.
     */
    q_node* Q()
    {
	assert(false);
	return 0;
    }

    /**
     * @internal
     * Type-cast to direction indicator.
     */
    direction_indicator* D()
    {
	assert(false);
	return 0;
    }

    /**
     * @internal
     * Type-cast to PQ-leaf.
     */
    pq_leaf* L()
    {
	return this;
    }
    
    //
    // Additional
    //

    /**
     * @internal
     */
    int other_id;

    /**
     * @internal
     */
    edge e;

    //
    // Friends 
    //

    /**
     * @internal
     * Allow planarity private access.
     */
    friend class planarity;

    /**
     * @internal
     * Allow pq_tree private access.
     */
    friend class pq_tree;
};


/**
 * @internal
 */
class GTL_EXTERN direction_indicator : public pq_node 
{
private:
    /**
     * @internal
     */
    direction_indicator (node n_, int id_) : pq_node (n_, id_) { };

    //
    // pq_node interface
    // 
    
    /**
     * @internal
     * Determines kind of this %node.
     */
    PQ_KIND kind() const
    {
	return DIR;
    }

    /**
     * @internal
     * Print this %node in gml format.
     */
	void write(std::ostream& os, int);

    // type-casts 

    /**
     * @internal
     * Type-cast to P-node.
     */
    p_node* P()
    {
	assert(false);
	return 0;
    }

    /**
     * @internal
     * Type-cast to Q-node.
     */
    q_node* Q()
    {
	assert(false);
	return 0;
    }

    /**
     * @internal
     * Type-cast to direction indicator.
     */
    direction_indicator* D()
    {
	return this;
    }

    /**
     * @internal
     * Type-cast to PQ-leaf.
     */
    pq_leaf* L()
    {
	assert(false);
	return 0;
    }
    
    //
    // Additional
    //

    /**
     * @internal
     */
    bool direction;	

    //
    // Friends 
    //

    /**
     * @internal
     * Allow planarity private access.
     */
    friend class planarity;

    /**
     * @internal
     * Allow pq_tree private access.
     */
    friend class pq_tree;
};

__GTL_END_NAMESPACE

#endif

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
