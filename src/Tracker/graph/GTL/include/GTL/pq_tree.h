/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   pq_tree.h
//
//==========================================================================
// $Id: pq_tree.h,v 1.20 2008/02/03 18:17:08 chris Exp $

#ifndef PQ_TREE_H
#define PQ_TREE_H

#include <GTL/GTL.h>
#include <GTL/pq_node.h>
#include <GTL/embedding.h>
#include <GTL/debug.h>

#include <list>
#include <iostream>

__GTL_BEGIN_NAMESPACE
    
/**
 * $Date: 2008/02/03 18:17:08 $
 * $Revision: 1.20 $
 * 
 * @brief PQ-Trees. 
 * 
 */ 
class GTL_EXTERN pq_tree 
{
public:
    /**
     * @internal
     */
    typedef symlist<pq_node*> sons_list;

    /**
     * @internal
     */
    typedef symlist<pq_node*>::iterator sons_iterator;

    /**
     * @brief Creates empty pq_tree.
     */
    pq_tree() : root(0), pert_root(0), pseudo(0), fail(0)
    {
    }
    
    /**
     * @brief Creates a PQ-tree consisting of a single P-node whose
     * whose children are the leaves given in list @p le.
     *
     * @param id st-number of @p n 
     * @param n node in the %graph to which the P-node refers
     * @param le list of children
     */
	pq_tree(int id, node n, const std::list<pq_leaf*>& le);

    /**
     * @brief Deletes PQ-tree.
     */
    ~pq_tree();

    /**
     * @brief Applies so called template matchings to the tree until either
     * all leaves labeled with @c id are consecutive in all equivalent
     * trees or until it is recognized that this can't be achieved. 
     * 
     * This operation is guaranteed to perform in O(PPT), where
     * PPT is the size of the so called @em pruned @em pertinent
     * @em subtree, which can be constructed, by cutting away all
     * the parts of the PQ-tree, that do not contain a leaf
     * labeled with @c id.
     *
     * @param leaves list of full leaves 
     *
     * @retval true if tree was successfully reduced
     * @retval false if reduction failed
     */
	bool reduce(std::list<pq_leaf*>& leaves);

    /**
     * @brief Replaces all the pertinent parts of the PQ-tree after a
     * (successful) reduction by a new P-node, whose children are given in
     * @p le.
     *
     * The edges (in the %graph), represented by the leaves are stored in
     * left to right order in @c em[n] They form (up to reversion)
     * the so called upward-embedding. A direction indicator representing
     * the direction in which the leaves were scanned is added to the sons
     * of the root of the pertinent subtree (if neccessary). All direction
     * indicators in the pertinent subtree are stored in @p dirs.
     *
     * @param id st-number of @p n
     * @param n node in the %graph to which the new P-node refers
     * @param le list of children
     * @param em planar embedding 
     * @param dirs direction indicators in pertinent subtree
     */
    void replace_pert(int id,
		      node n,
			  const std::list<pq_leaf*>& le,
		      planar_embedding* em = 0,
			  std::list<direction_indicator>* dirs = 0);
    
    /**
     * @brief Scans whole tree from left to right and stores edges (in the
     * %graph) represented by the leaves in @p em. 
     * 
     * All direction indicators in the tree are stored in @p
     * dirs. This is used in %planarity test to get the upward
     * %embedding of the last node, because no reduction is
     * needed in this case since all leaves are labeled with the
     * same number.
     *
     * @param em planar embedding 
     * @param dirs direction indicators in tree
     */
	void get_frontier(planar_embedding& em, std::list<direction_indicator>& dirs);

    /**
     * @brief After a (successful) reduction @c reset has to be called in
     * order to prepare the tree for the next reduction.
     */
    void reset ();

    /**
     * @brief Returns the (PQ-) node to which none of the
     * template matchings were applicable.
     *
     * @return PQ-node at which the reduction failed
     */
    pq_node* get_fail()
    {
	return fail;
    }

    /**
     * @brief Returns true iff fail is the root of the
     * pertinent subtree.
     *
     * @retval true iff reduction failed at the root of the
     *	       pertinent subtree.
     */
    bool is_fail_root()
    {
	return failed_at_root;
    }

    /**
     * @brief Remove a direction indicator among sons of a Q-node.
     * Needed for computation of the obstruction set.
     *
     * @param q_fail the Q-node on which the reduction failed
     * @param the position of the direction indicator among the sons
     *
     * @retval next valid sons iterator
     */
    sons_iterator remove_dir_ind(q_node* q_fail, sons_iterator s_it);

    /**
     * @brief Checks the structure of the tree. 
     *
     * @note Use this only for debugging since it scans the whole tree,
     * which isn't acceptable in terms of performance in most cases.
     * 
     * @retval true iff tree passes checks
     */
    bool integrity_check () const;

//    p_node* insert_P (pq_node*, sons_list&);

//    q_node* insert_Q (pq_node*, sons_list&);

//    pq_leaf* insert_leaf (pq_node*);

//    void insert (pq_node*, pq_node*);
private:
    /**
     * @internal
     * Tries to give all the nodes in the pertinent subtree the right father 
     * pointer. If either all nodes in the pertinent subtree recieved a
     * valid father pointer or there was excactly one block of inner nodes
     * just below the root of the pertinent subtree, the result is true. If
     * @c bubble_up returns false a reduction isn't possible.
     *
     * @param leaves list of full leaves
     *
     * @retval true iff bubble-up succeeded
     */
	bool bubble_up(std::list<pq_leaf*>& leaves);
    
    /**
     * @internal
     * Scans the subtree rooted at @p p and stores edges (in the %graph)
     * represented by the leaves in @p em. All direction indicators in the
     * subtree are stored in @p dirs.
     *
     * @param p root of subtree  
     * @param em planar embedding
     * @param dirs direction indicators in subtree
     */
    void dfs(pq_node* p,
	     planar_embedding& em,
		 std::list<direction_indicator>& dirs);

    /**
     * @internal
     * Test whether one of the predecessors of @p le has mark @c BLOCKED.
     * Used when bubble-up failed to determine a minimum subtree, whose root
     * has inner pertinent children. Minimum in this regard means that no 
     * descendant of the subtree's root has @c BLOCKED children.
     *
     * @param le (PQ-)node
     *
     * @return @c BLOCKED node or @c 0
     */
    pq_node* leads_to_blocked(pq_node* le);

    
    /**
     * @internal
     * Tests wheter @p le leads to @p other, i.e. if  @p other is a
     * predecessor of @p le. Used to limit the leaves for reduction in case
     * that bubble-up failed to the leaves in the minimum subtree, whose
     * root has inner pertinent children.
     *
     * @param le node to be tested
     * @param other root of subtree
     *
     * @retval true iff @p le is in subtree rooted at @p other
     */
    bool leads_to(pq_node* le, pq_node* other);


    /**
     * @internal
     * In case bubble-up failed a (PQ-)node has to be found which has inner 
     * children pertinent such that no node in its subtree has inner
     * children pertinet. Template matchings then are only performed in this
     * subtree.
     *
     * @param leaves list of full leaves
     *
     * @return root of the minimum subtree
     */
	pq_node* where_bubble_up_failed(std::list<pq_leaf*>& leaves);


    /**
     * @internal
     * Tests whether some descendants of @p n are @c BLOCKED.
     *
     * @param n root for subtree to be checked
     *
     * @return (PQ-) @c BLOCKED node or @c 0
     */
    pq_node* blocked_in_subtree(pq_node* n);


    //
    // Template Matchings
    //

    //---------------------------------------------------------- P-Templates
    
    /**
     * @internal
     * Template P1.
     */
    bool P1 (p_node* x, bool);

    /**
     * @internal
     * Template P2.
     */
    bool P2 (p_node* x);

    /**
     * @internal
     * Template P3.
     */
    bool P3 (p_node* x);

    /**
     * @internal
     * Template P4.
     */
    bool P4 (p_node* x);

    /**
     * @internal
     * Template P5.
     */
    bool P5 (p_node* x);

    /**
     * @internal
     * Template P6.
     */
    bool P6 (p_node* x);

    //---------------------------------------------------------- Q-Templates
    
    /**
     * @internal
     * Template Q1.
     */
    bool Q1 (q_node* x, bool);

    /**
     * @internal
     * Template Q2.
     */
    bool Q2 (q_node* x, bool);

    /**
     * @internal
     * Template Q3.
     */
    bool Q3 (q_node* x);


    //
    // Data
    //

    /**
     * @internal
     * List of (PQ-) nodes to be cleared if the reduction stopped now.
     */
	std::list<pq_node*> clear_me;

    /**
     * @internal
     * Root of tree.
     */
    pq_node* root;

    /**
     * @internal
     * Root of pertinent subtree; defined after succesful reduction.
     */
    pq_node* pert_root;

    /**
     * @internal
     * In some cases the root of the pertinent subtree might not be known, 
     * because it is a Q-node and all its pertinent children are inner. In
     * this case for the time of reduction an pseudo node is created as root
     * of the pertinent subtree, which gets only the pertinent children as
     * sons.
     */
    q_node* pseudo;

    /**
     * @internal
     * (PQ-) node for which the reduction failed.
     */
    pq_node* fail;

    /**
     * @internal
     * @c true iff reduction failed at the root of the pertinent subtree.
     */
    bool failed_at_root;

    /**
     * @internal
     * Number of pertinent leaves for the current reduction; defined after
     * bubble-up.
     */
    int pert_leaves_count;

    //
    // Friends
    //

    /**
     * @internal
     * Allow operator<< private access.
     */
	GTL_EXTERN friend std::ostream& operator<< (std::ostream&, const pq_tree&);
};

__GTL_END_NAMESPACE

#endif

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
