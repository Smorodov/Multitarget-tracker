/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   pq_tree.cpp 
//
//==========================================================================
// $Id: pq_tree.cpp,v 1.22 2008/02/03 18:12:07 chris Exp $

#include <GTL/pq_tree.h>
#include <GTL/debug.h>

#include <stack>
#include <queue>
#include <utility>
#include <cstdio>

__GTL_BEGIN_NAMESPACE

pq_tree::pq_tree (int id, node n, const std::list<pq_leaf*>& li)
{
#ifdef _DEBUG  
    GTL_debug::init_debug();
#endif
    std::list<pq_leaf*>::const_iterator it;
    std::list<pq_leaf*>::const_iterator end = li.end();
    sons_list sons;
    pq_leaf* tmp;
    
    for (it = li.begin(); it != end; ++it) {
	tmp = *it;
	tmp->pos = sons.insert (sons.end(), tmp);
    }
    
    root = new p_node(n, id, sons);
    pert_root = 0;
    fail = 0;
    pseudo = 0;
}

pq_tree::~pq_tree ()
{
#ifdef _DEBUG  
    GTL_debug::close_debug();
#endif
    reset ();

    if (root) {
	delete root;
    }
}


bool pq_tree::bubble_up (std::list<pq_leaf*>& leaves) 
{
	std::queue<pq_node*> qu;
    int block_count = 0;
    int blocked_siblings = 0;
    pert_leaves_count = 0;
    int off_the_top = 0;
    pq_node* tmp;
    
    assert (clear_me.empty());
    
    std::list<pq_leaf*>::const_iterator it = leaves.begin(); 
    std::list<pq_leaf*>::const_iterator lend = leaves.end();
    
    while (it != lend) {
	qu.push (*it);
	(*it)->lpos = clear_me.insert (clear_me.end(), *it);
	pert_leaves_count++;
	++it;
    }
    
    sons_iterator next, prev, end;
    pq_node* father = nullptr;
    int size = pert_leaves_count;
    
    while (size + block_count + off_the_top > 1) {
	if (size == 0) {
	    return false;
	}
	
	tmp = qu.front();
	qu.pop();
	size--;
	tmp->pert_leaves = 0;
	
	if (tmp == root) {
	    off_the_top = 1;
	    tmp->mark = pq_node::UNBLOCKED;
	    continue;
	} 	

	tmp->mark = pq_node::BLOCKED;
	
	if (tmp->is_endmost) {
	    father = tmp->father;
	    tmp->mark = pq_node::UNBLOCKED;
	    end = father->sons.end();
	    
	    if (father->kind() == pq_node::Q_NODE) {
		blocked_siblings = 0;
 		next = tmp->pos;
		prev = tmp->pos;
		++next;
		--prev;
		
		if (next != end) {
		    if ((*next)->mark == pq_node::BLOCKED) {
			++blocked_siblings;
		    }
		} else if (prev != end) {		    
		    if ((*prev)->mark == pq_node::BLOCKED) {
			++blocked_siblings;
		    }
		}
	    }
	    
	} else {
	    next = tmp->pos;
	    prev = tmp->pos;
	    ++next;
	    --prev;
	    blocked_siblings = 0;
	    
	    if ((*prev)->mark == pq_node::UNBLOCKED) {
		tmp->mark = pq_node::UNBLOCKED;
		tmp->father = (*prev)->father;
		father = tmp->father;
		end = father->sons.end();
	    } else if ((*prev)->mark == pq_node::BLOCKED) {
		blocked_siblings++;
	    }
	    
	    if ((*next)->mark == pq_node::UNBLOCKED) {
		tmp->mark = pq_node::UNBLOCKED;
		tmp->father = (*next)->father;
		father = tmp->father;
		end = father->sons.end();
	    } else if ((*next)->mark == pq_node::BLOCKED) {
		blocked_siblings++;
	    } 
	}
	
	if (tmp->mark == pq_node::UNBLOCKED) {
	    ++(father->pert_children);
	    
	    if (father->mark == pq_node::UNMARKED) {
		qu.push (father);
		father->lpos = clear_me.insert (clear_me.end(), father);
		size++;
		father->mark = pq_node::QUEUED;
	    }
	    
	    if (father->kind() == pq_node::Q_NODE) {
    	        pq_node* tmp;

 		while (next != end) {
		    tmp = *next;
		    if (tmp->mark == pq_node::BLOCKED) {
			tmp->father = father;
			tmp->mark = pq_node::UNBLOCKED;

			if (tmp->kind () != pq_node::DIR) {
			    ++(father->pert_children);
			}
		    } else if (tmp->kind () == pq_node::DIR && 
			       tmp->mark == pq_node::UNMARKED) {
 			tmp->lpos = clear_me.insert (clear_me.end(), tmp);
			tmp->father = father;
			tmp->mark = pq_node::UNBLOCKED;
		    } else {
			break;
		    }

		    ++next;
		}

 		while (prev != end) {
 		    tmp = *prev;
		    if (tmp->mark == pq_node::BLOCKED) {
			tmp->father = father;
			tmp->mark = pq_node::UNBLOCKED;

			if (tmp->kind () != pq_node::DIR) {
			    ++(father->pert_children);
			}
		    } else if (tmp->kind () == pq_node::DIR && 
			       tmp->mark == pq_node::UNMARKED) {
 			tmp->lpos = clear_me.insert (clear_me.end(), tmp);
			tmp->father = father;
			tmp->mark = pq_node::UNBLOCKED;
		    } else {
			break;
		    }

		    --prev;
		}
		
		block_count -= blocked_siblings;
	    }
	    
	} else {
	    
	    //
	    // tmp is BLOCKED
	    //

 	    while ((*next)->kind() == pq_node::DIR && 
		   (*next)->mark == pq_node::UNMARKED) {
		(*next)->mark = pq_node::BLOCKED;
		(*next)->lpos = clear_me.insert (clear_me.end(), *next);
		++next;
	    }

	    while ((*prev)->kind() == pq_node::DIR && 
		   (*prev)->mark == pq_node::UNMARKED) {
		(*prev)->mark = pq_node::BLOCKED;
		(*prev)->lpos = clear_me.insert (clear_me.end(), *prev);
		--prev;
	    }

	    block_count += 1 - blocked_siblings;
	}
    }
    
    return true;
}


pq_node* pq_tree::where_bubble_up_failed (std::list<pq_leaf*>& leaves)
{

    //
    // Search the first leaf that leads to an interior block.
    //

    pq_leaf* le;
    pq_node* blocked = 0;
    
    std::list<pq_leaf*>::iterator l_it = leaves.begin();
    std::list<pq_leaf*>::iterator l_end = leaves.end();
    q_node* father = 0;

    while (l_it != l_end ) {
	le = *l_it;
	blocked = leads_to_blocked (le);

	if (blocked != 0) {
	    //
	    // Search the father of this block.
	    //
	    
	    sons_iterator it = blocked->pos;
	    while (!(*it)->is_endmost) {
		++it;
	    }
	    
	    father = (*it)->father->Q();
	    
	    //
	    // give all the children the right father. 
	    //
	    
	    it = father->sons.begin();
	    sons_iterator end = father->sons.end();
	    
	    while (it != end) {
		if ((*it)->mark == pq_node::BLOCKED) {
		    (*it)->father = father;
		    (*it)->mark = pq_node::UNBLOCKED;
		    if ((*it)->kind() != pq_node::DIR) {
			++(father->pert_children);
		    }
		}
		
		++it;
	    }

	    //
	    // We have to assure that there isn't any other interior block in the
	    // subtree of father. 
	    //

	    pq_node* another = blocked_in_subtree (father);
	    
	    if (another == 0) {
		break;
	    }
	}
	
	++l_it;
    }

    assert (father != 0);

    //
    // delete all pertinent leaves that do not lead to father
    //

    l_it = leaves.begin();

    while (l_it != l_end ) {
	le = *l_it;

	if (!leads_to (le, father)) {
	    l_it = leaves.erase (l_it);
	} else {
	    ++l_it;
	}
    }

    return father;
}


pq_node* pq_tree::blocked_in_subtree (pq_node* n) 
{
    if (n->kind() == pq_node::LEAF) {
	return 0;
    } 

    if (n->mark == pq_node::BLOCKED) {
	return n;
    }

    sons_iterator it = n->sons.begin();
    sons_iterator end = n->sons.end();

    while (it != end) {
	pq_node* bl = blocked_in_subtree (*it);
	
	if (bl) {
	    return bl;
	}

	++it;
    }

    return 0;
}


bool pq_tree::leads_to (pq_node* le, pq_node* other) 
{
    if (le == root) {
	return false;
    } else if (le->mark == pq_node::BLOCKED) {
	return false;
    } else if (le->mark == pq_node::UNMARKED) {
	return false;
    } else if (le->father == other) {
	return true;
    } else {
	return leads_to (le->father, other);
    }
}

pq_node* pq_tree::leads_to_blocked (pq_node* le) 
{
    if (le == root) {
	return 0;
    } else if (le->mark == pq_node::BLOCKED) {
	return le;
    } else if (le->mark == pq_node::UNMARKED) {
	return 0;
    } else {
	return leads_to_blocked (le->father);
    }
}


bool pq_tree::reduce (std::list<pq_leaf*>& leaves) 
{   

    GTL_debug::debug_message ("REDUCING %d\n", leaves.front()->id);
    fail = 0;
    
    if (!bubble_up (leaves)) {

	//
	// Find the node that has an interior block. 
	//

	GTL_debug::debug_message ("Bubble-Up failed !!\n");
	fail = where_bubble_up_failed (leaves);
    }
    
	std::queue<pq_node*> qu;
    pq_leaf* le;
    std::list<pq_leaf*>::iterator l_it = leaves.begin();
    std::list<pq_leaf*>::iterator l_end = leaves.end();
    
    while (l_it != l_end ) {
	le = *l_it;
	qu.push (le);
	le->pert_leaves = 1;
	++l_it;
    }
    
    pq_node* tmp;

    while (!qu.empty()) {
	tmp = qu.front();
	qu.pop();
	clear_me.erase (tmp->lpos);

	if (tmp->mark == pq_node::BLOCKED) {
	    pseudo = new q_node (node(), 0);	    
	    sons_iterator past = tmp->pos;
			
	    //
	    // Get maximal connected block of BLOCKED siblings right of tmp
	    //

	    while ((*past)->mark == pq_node::BLOCKED) {
		(*past)->mark = pq_node::UNBLOCKED;
		(*past)->father = pseudo;

		if ((*past)->kind() != pq_node::DIR) { 
		    pseudo->pert_children++;
		}

		++past;
	    }
	    

	    //
	    // Delete surrounding direction indicators
	    //

	    --past;

	    while ((*past)->kind() == pq_node::DIR) {
		(*past)->clear();
		clear_me.erase ((*past)->lpos);
		--past;
	    }

	    pseudo->pert_end = past;

	    //
	    // Get maximal connected block of BLOCKED siblings left of tmp
	    // 

	    sons_iterator first = tmp->pos;
	    --first;

	    while ((*first)->mark == pq_node::BLOCKED) {
		(*first)->mark = pq_node::UNBLOCKED;
    		(*first)->father = pseudo;

		if ((*first)->kind() != pq_node::DIR) {
		    pseudo->pert_children++;
		}

		--first;
	    }
	    

	    //
	    // Delete surrounding direction indicators
	    //

	    ++first;

	    while ((*first)->kind() == pq_node::DIR) {
		(*first)->clear();
		clear_me.erase ((*first)->lpos);
		++first;
	    }

	    pseudo->pert_begin = first;
			
	    GTL_debug::debug_message ("creating pseudo-node as root\n");
	    pseudo->mark = pq_node::UNBLOCKED;
	    ++past;
	    pseudo->sons.attach_sublist (first, past);
	    pseudo->pert_cons = true;
	    pseudo->lpos = clear_me.insert (clear_me.end(), pseudo);
	}
	
	if (tmp->pert_leaves == pert_leaves_count) {
	    
	    //
	    // tmp is the root of the pertinent subtree
	    //
	    
	    if (tmp->kind() == pq_node::LEAF) {
		pert_root = tmp;
		GTL_debug::debug_message ("full leaf is root\n");
	    } else if (tmp->kind() == pq_node::P_NODE) {
		if (P1 (tmp->P(), true)) {
		    GTL_debug::debug_message ("P1 matched for root\n");
		} else if (P2 (tmp->P())) {
		    GTL_debug::debug_message ("P2 matched for root\n");
		} else if (P4 (tmp->P())) {
		    GTL_debug::debug_message ("P4 matched for root\n");
		} else if (P6 (tmp->P())) {
		    GTL_debug::debug_message ("P6 matched for root\n");
		} else {
		    GTL_debug::debug_message ("NO MATCHING FOR P-ROOT\n");
		    fail = tmp;
		    failed_at_root = true;
		    return false;
		}
	    } else {
		if (!tmp->Q()->pert_cons) {
		    GTL_debug::debug_message ("pertinent children not consecutive\n");
		    fail = tmp;
		    failed_at_root = true;
		    return false;
		} else if (Q1 (tmp->Q(), true)) {
		    GTL_debug::debug_message ("Q1 matched for root\n");
		} else if (Q2 (tmp->Q(), true)) {
		    GTL_debug::debug_message ("Q2 matched for root\n");
		} else if (Q3 (tmp->Q())) {
		    GTL_debug::debug_message ("Q3 matched for root\n");
		} else {
		    GTL_debug::debug_message ("NO MATCHING FOR Q-ROOT\n");
		   		    
		    if (tmp == pseudo) {

			//
			// search the real father 
			//

			sons_iterator it = pseudo->sons.begin();
			pseudo->sons.front()->is_endmost = false;
			pseudo->sons.back()->is_endmost = false;
			pseudo->sons.detach_sublist();
			assert (pseudo->sons.empty());

			while (!(*it)->is_endmost) {
			    --it;
			}
			
			tmp = (*it)->father;
			q_node* q_tmp = tmp->Q();
			q_tmp->pert_begin = pseudo->pert_begin;
			q_tmp->pert_end = pseudo->pert_end;
			q_tmp->partial_count = pseudo->partial_count;
			q_tmp->full_count = pseudo->full_count;
			q_tmp->pert_cons = pseudo->pert_cons;

			for (int i = 0; i < q_tmp->partial_count; ++i) {
			    q_tmp->partial_pos[i] = pseudo->partial_pos[i];
			}

			delete pseudo;
			pseudo = 0;
		    } 

		    fail = tmp;
		    failed_at_root = true;
		    return false;
		}
	    }

	} else {
	    
	    //
	    // tmp is not the root of the pertinent subtree.
	    //
	    
	    if (tmp == pseudo || tmp == root) {

		//
		// This should not happen when bubble_up was true.
		//

		assert (false);

	    } else {
		pq_node* father = tmp->father;
		
		if (tmp->kind() == pq_node::LEAF) {
		    father->full (tmp->pos);
		    tmp->clear();
		    GTL_debug::debug_message ("full leaf processed\n");
		    
		} else if (tmp->kind() == pq_node::P_NODE) {
		    if (P1 (tmp->P(), false)) {
			GTL_debug::debug_message ("P1 matched for non-root\n");
		    } else if (P3 (tmp->P())) {
			GTL_debug::debug_message ("P3 matched for non-root\n");
		    } else if (P5 (tmp->P())) {
			GTL_debug::debug_message ("P5 matched for non-root\n");
		    } else {
			GTL_debug::debug_message ("NO MATCHING FOR P-NON-ROOT\n");
			fail = tmp;
			failed_at_root = false;
			return false;
		    }
		    
		} else {
		    if (!tmp->Q()->pert_cons) {
			GTL_debug::debug_message ("pertinent children not consecutive\n"); 
			fail = tmp;
			return false;
		    } else if (Q1 (tmp->Q(), false)) {
			GTL_debug::debug_message ("Q1 matched for non-root\n");
		    } else if (Q2 (tmp->Q(), false)) {
			GTL_debug::debug_message ("Q2 matched for non-root\n");
		    } else {
			GTL_debug::debug_message ("NO MATCHING FOR Q-NON-ROOT\n");
			fail = tmp;
			failed_at_root = false;
			return false;
		    }
		} 
		
		
		//
		// If all the other pertinent siblings of tmp have already been
		// matched father of tmp is queued.
		//
		
		--(father->pert_children);
		
		if (father->pert_children == 0) {	
		    if (father == fail) {
			failed_at_root = false;
			return false;
		    } else {
			qu.push (father);
		    }
		}
	    } 
	}
    }
    
    return true;
}


void pq_tree::reset ()
{
    pq_node* tmp;
    
    while (!clear_me.empty()) {
	tmp = clear_me.front();
	GTL_debug::debug_message ("Clearing %d\n", tmp->id);
	clear_me.pop_front();
	tmp->clear();
	tmp->pert_children = 0;
    }
    
    if (pert_root) {	
	pert_root->clear();
	pert_root = 0;
    }
    
    if (pseudo) {
	pseudo->sons.front()->is_endmost = false;
	pseudo->sons.back()->is_endmost = false;
	pseudo->sons.detach_sublist();
	assert (pseudo->sons.empty());
	delete pseudo;
	pseudo = 0;
    } 

    if (fail) {
	fail->clear();
	fail = 0;
    }    
}


void pq_tree::dfs (pq_node* act, planar_embedding& em,
		   std::list<direction_indicator>& dirs) 
{
    if (act->kind() == pq_node::LEAF) {
	em.push_back (act->n, ((pq_leaf*) act)->e);
	return;
    }

    sons_iterator it = act->sons.begin();
    sons_iterator end = act->sons.end();
    
    while (it != end) {
	if ((*it)->kind() == pq_node::DIR) {
	    direction_indicator* dir = (*it)->D();
	    if (dir->mark != pq_node::UNMARKED) {
		clear_me.erase (dir->lpos);
	    }
	    sons_iterator tmp = it;
	 
	    if (++tmp == ++(dir->pos)) {
		dir->direction = true;
	    } else {
		dir->direction = false;
	    }
	
	    dirs.push_back (*dir);

	} else {
	    dfs (*it, em, dirs);
	}

	++it;
    }
}


void pq_tree::replace_pert (int id, node _n, const std::list<pq_leaf*>& li,
			    planar_embedding* em, std::list<direction_indicator>* dirs) 
{
    assert (pert_root);
    assert (!li.empty());
    pq_leaf* tmp = 0;
    std::list<pq_leaf*>::const_iterator it;
    std::list<pq_leaf*>::const_iterator end = li.end();
    sons_list sons;
    int size = 0;
    
    for (it = li.begin(); it != end; ++it) {
	tmp = *it;
	tmp->pos = sons.insert (sons.end(), tmp);
	++size;
    }
    
    pq_node* ins;
    
    if (size == 1) {
    	sons.erase (tmp->pos);
    	ins = tmp;
    } else {
	ins = new p_node(_n, id, sons);
    }
    
    if (pert_root->kind() == pq_node::Q_NODE) {
	q_node* q_root = pert_root->Q();	
	sons_iterator it = q_root->pert_begin;
	sons_iterator end = q_root->pert_end;
	sons_iterator tmp = it;
	sons_iterator sons_end = q_root->sons.end();
	--tmp;

	while (tmp != sons_end) {
	    if ((*tmp)->kind() != pq_node::DIR) {
		break;
	    }

	    --tmp;
	}   

	it = ++tmp;

	tmp = end;
	++tmp;

	while (tmp != sons_end) {
	    if ((*tmp)->kind() != pq_node::DIR) {
		break;
	    }

	    ++tmp;
	}   

	end = --tmp;

	ins->is_endmost = (*end)->is_endmost;
	++end;
	
	while (it != end) {
	    if (em && dirs) {
		if ((*it)->kind() == pq_node::DIR) {
		    direction_indicator* dir = (*it)->D();
		    clear_me.erase (dir->lpos);
		    sons_iterator tmp = it;
		
		    if (++tmp == ++(dir->pos)) {
			dir->direction = true;
		    } else {
			dir->direction = false;
		    }
		    
		    dirs->push_back (*dir);
		} else {
		    dfs (*it, *em, *dirs);
		}
	    }

	    delete *it;
	    it = pert_root->sons.erase (it);
	}	
	
	if (pert_root->sons.empty() && pert_root != pseudo) {
	    ins->pos = pert_root->pos;
	    ins->father = pert_root->father;
	    ins->is_endmost = pert_root->is_endmost;
	    ins->up = pert_root->up;
	    ins->up_id = pert_root->up_id;
	    
	    if (root == pert_root) {
		root = ins;
	    } else { 
		*(pert_root->pos) = ins;
	    }
	    
	    delete pert_root;
	    pert_root = 0;
	    
	} else {
	    if (em && dirs) {
		direction_indicator* ind = new direction_indicator (_n, id);
		ind->is_endmost = false;
		ind->pos = pert_root->sons.insert (end, ind);
	    }

	    ins->pos = pert_root->sons.insert (end, ins);
	    ins->father = pert_root;
	    ins->up = _n;
	    ins->up_id = id;
	}	    
	
    } else {
	if (em && dirs) {
	    dfs (pert_root, *em, *dirs);	
	}

	ins->is_endmost = pert_root->is_endmost;
	ins->father = pert_root->father;
	ins->pos = pert_root->pos;
	ins->up = pert_root->up;
	ins->up_id = pert_root->up_id;
	
	if (root == pert_root) {
	    root = ins;
	} else {
	    *(pert_root->pos) = ins;
	}
	
	delete pert_root;
	pert_root = 0;
    }    
}	    

void pq_tree::get_frontier (planar_embedding& em, 
			    std::list<direction_indicator>& dirs)
{
    dfs (root, em, dirs);
}

//------------------------------------------------------------------------ P1
// Requirements:
//
// * x is a P-node having only full children
// * wheter x is the root or not is specified by the second parameter
//

bool pq_tree::P1 (p_node* x, bool is_root)
{
    if (x->child_count == x->full_count) {
	if (!is_root) {
	    x->father->full (x->pos);
	} else {
	    pert_root = x;
	}
	
	x->sons.splice (x->sons.end(), x->full_sons.begin(),
	    x->full_sons.end());
	x->clear();
	return true;
    } 
    
    return false;
}


//----------------------------------------------------------------------- P2
// Requirements:
//
// * x is a P-node having both full and empty children
// * x has no partial children 
// * x is the root of the pertinent subtree 
//     ==> more than one pertinent child 
// * P1 didn't match 
//     ==> at least one non-full child
//
bool pq_tree::P2 (p_node* x) 
{
    if (x->partial_count != 0) {
	return false;
    }

    p_node* ins = new p_node(x->n, x->id, x->full_sons);
    ins->father = x;
    ins->is_endmost = true;
    ins->pos = x->sons.insert (x->sons.end(), ins);
    x->child_count -= (x->full_count - 1);
    x->clear();
    pert_root = ins;
    return true;
}


//------------------------------------------------------------------------ P3
// Requirements:
//
// * x is a P-node having both full and empty children.
// * x isn't the root of the pertinent subtree.
// * P1 didn't match.
//   ==> at least one non-full child.
// * x has no partial children
//

bool pq_tree::P3 (p_node* x) 
{
    if (x->partial_count != 0) {
	return false;
    }
    
    q_node* new_q = new q_node (x->n, x->id);
    pq_node* father = x->father;
    pq_node* ins;
    
    *(x->pos) = new_q; 
    new_q->pos = x->pos;
    new_q->up = x->up;
    new_q->up_id = x->up_id;
    new_q->is_endmost = x->is_endmost;
    new_q->father = father;
    new_q->pert_leaves = x->pert_leaves;
    
    if (x->full_count > 1) {
	ins = new p_node (x->n, x->id, x->full_sons);
    } else {
    	ins = x->full_sons.front();
    	x->full_sons.erase (x->full_sons.begin());
    	assert (x->full_sons.empty());
    }
    
    ins->up = x->n;
    ins->up_id = x->id;
    ins->is_endmost = true;
    ins->father = new_q;
    ins->pos = new_q->sons.insert (new_q->sons.end(), ins);
    new_q->pert_cons = true;
    new_q->pert_begin = ins->pos;
    new_q->pert_end = ins->pos;
    
    if (x->child_count - x->full_count > 1) {
	ins = x;
	ins->up = x->n;
	ins->up_id = x->id;
	x->child_count -= x->full_count;
	x->clear();
    } else {
    	ins = x->sons.front();
	ins->up = x->n;
	ins->up_id = x->id;
    	x->sons.erase (x->sons.begin());
    	assert (x->sons.empty());
    	delete x;
    }
    
    ins->is_endmost = true;
    ins->father = new_q;
    ins->pos = new_q->sons.insert (new_q->pert_begin, ins);
    father->partial (new_q->pos);
    
    return true;
}

//------------------------------------------------------------------------ P4
// Requirements:
//
// * x is a P-node and the root of the pertinent subtree.
//   ==> more than one non-empty child, i.e. at least one full child.
// * x has excactly one partial child
// * P1 and P2 didn't match 
//   ==> at least one partial child
//
bool pq_tree::P4 (p_node* x) 
{
    if (x->partial_count > 1) {
	return false;
    }
    
    q_node* part = x->partial_sons.front()->Q();
    part->n = x->n;
    part->id = x->id;
    pq_node* ins;
    
    if (x->full_count > 1) {
	ins = new p_node (x->n, x->id, x->full_sons);
    } else {
    	ins = x->full_sons.front();
    	x->full_sons.erase (x->full_sons.begin());
    	assert (x->full_sons.empty());
    }
    
    part->sons.back()->is_endmost = false;
    ins->is_endmost = true;
    
    ins->up = x->n;
    ins->up_id = x->id;
    ins->father = part;
    ins->pos = part->sons.insert (part->sons.end(), ins);
    part->pert_end = ins->pos;
    x->child_count -= x->full_count;
    
    if (x->child_count == 1) {
	if (root == x) {
	    root = part;
	} else {
	    *(x->pos) = part;
	}

	part->pos = x->pos;
	part->is_endmost = x->is_endmost;
	part->father = x->father;
	part->up = x->up;
	part->up_id = x->up_id;
	x->partial_sons.erase (x->partial_sons.begin());
	
	delete x; 
    } else {
	x->sons.splice (x->sons.end(), part->pos);
	x->clear();
    }
    
    pert_root = part;
    return true;
}


//------------------------------------------------------------------------ P5
// Requirements:
//
// * x is a P-node and not the root of the pertinent subtree
// * x has exactly one partial child.
// * P1 and P3 didn't match
//  ==> at least one partial child
//
bool pq_tree::P5 (p_node* x)
{
    if (x->partial_count > 1) {
	return false;
    }
    
    pq_node* father = x->father;
    q_node* part = x->partial_sons.front()->Q();	 
    part->n = x->n;
    part->id = x->id;
    part->up = x->up;
    part->up_id = x->up_id;

    x->partial_sons.erase (x->partial_sons.begin());
    part->is_endmost = x->is_endmost;
    part->father = father;
    *(x->pos) = part;
    part->pos = x->pos;
    part->pert_leaves = x->pert_leaves;
    pq_node* ins;
    
    if (x->full_count > 1) {
	ins = new p_node (x->n, x->id, x->full_sons);
    } else if (x->full_count == 1) {
	ins = x->full_sons.front();
	x->full_sons.erase (x->full_sons.begin());
	assert (x->full_sons.empty());
    } else {
	ins = 0;
    }
    
    if (ins) {
	ins->up = x->n;
	ins->up_id = x->id;
	part->sons.back()->is_endmost = false;
	ins->is_endmost = true;
	ins->father = part;
	ins->pos = part->sons.insert (part->sons.end(), ins);
	part->pert_end = ins->pos;
    }
    
    x->child_count -= (x->full_count + 1);
    
    if (x->child_count > 1) {
	ins = x;
	ins->up = x->n;
	ins->up_id = x->id;
	x->clear();
    } else if (x->child_count == 1) {
	ins = x->sons.front();
	ins->up = x->n;
	ins->up_id = x->id;
	x->sons.erase (x->sons.begin());
	delete x;
    } else {
	ins = 0;
	delete x;
    }
    
    if (ins) {
	part->sons.front()->is_endmost = false;
	ins->is_endmost = true;
	ins->father = part;
	ins->pos = part->sons.insert (part->sons.begin(), ins);
    }
    
    father->partial (part->pos);
    return true;
}


//------------------------------------------------------------------------ P6
// Requirements:
//
// * x is the root of the pertinent subtree and has two partial children.
// * P1, P2 and P4 didn't match
//   ==> at least two partial children.
//
bool pq_tree::P6 (p_node* x)
{
    if (x->partial_count > 2) {
	return false;
    }
    
    
    q_node* part2 = x->partial_sons.front()->Q();
    x->partial_sons.erase (x->partial_sons.begin());
    q_node* part1 = x->partial_sons.front()->Q();
    part1->n = x->n;
    part1->id = x->id;
    pq_node* ins;
    
    if (x->full_count > 1) {
	ins = new p_node (x->n, x->id, x->full_sons);
    } else if (x->full_count == 1) {
	ins = x->full_sons.front();
	x->full_sons.erase (x->full_sons.begin());
	assert (x->full_sons.empty());
    } else {
	ins = 0;
    }	 
    
    part1->sons.back()->is_endmost = false;
    
    if (ins) {
	ins->up = x->n;
	ins->up_id = x->id;
	ins->is_endmost = false;
	ins->pos = part1->sons.insert (part1->sons.end(), ins);
    }
    
    part2->turn ();
    part2->sons.front()->is_endmost = false;
    part2->sons.back()->father = part1;
    part1->sons.splice (part1->sons.end(), part2->sons.begin(),
	part2->sons.end()); 
    part1->pert_end = part2->pert_begin;
    part1->pert_end.reverse();
    x->child_count -= (x->full_count + 1);
    delete part2;
    
    if (x->child_count == 1) {
	if (root == x) {
	    root = part1;
	} else {
	    *(x->pos) = part1;
	}
	part1->pos = x->pos;
	part1->is_endmost = x->is_endmost;
	part1->father = x->father;
	part1->up = x->up;
	part1->up_id = x->up_id;
	x->partial_sons.erase (x->partial_sons.begin());
	
	delete x;
    } else { 
	x->sons.splice (x->sons.end(), x->partial_sons.begin());
	x->clear();
    }
    
    pert_root = part1;
    return true;
}

//------------------------------------------------------------------------ Q1
// Requirements:
//
// * x is a Q-node having only full children
// * wheter x is the root or not is specified by the second parameter
//
bool pq_tree::Q1 (q_node* x, bool is_root)
{
    if (x->partial_count > 0) return false;
    
    if (*(x->pert_begin) == x->sons.front() 
	&& *(x->pert_end) == x->sons.back()) {
	
	if (!is_root) {
	    x->father->full (x->pos);
	} else {
	    pert_root = x;
	}
	
	return true;
    } 
    
    return false;
}


//------------------------------------------------------------------------ Q2
// Requirements:
//
// * Q1 didn't match ==> x has at least one non-full child.
// * wheter x is the root or not is specified by the second parameter
// * x has at most one partial child
// * If x has empty children, the partial child must be at pert_begin;
//   if x hasn't any empty children the partial child is allowed to be at 
//   pert_end, since this can be transformed into the former case.
//
bool pq_tree::Q2 (q_node* x, bool is_root) 
{
    if (x->partial_count > 1) {
	return false;
    }
    
    if (x->partial_count == 1) {
	if (x->partial_pos[0] == x->pert_end && 
	    x->pert_begin == x->sons.begin() && 
	    x->pert_begin != x->pert_end)
	{
	    if (!is_root) {
		q_node* part = (*(x->pert_end))->Q();
		x->turn();
		sons_iterator tmp = x->pert_begin;
		x->pert_begin = x->pert_end;
		x->pert_end = tmp;
		x->pert_begin.reverse();
		x->pert_end.reverse();
		x->merge (x->pert_begin);
		x->pert_begin = part->pert_begin;		 
		delete part;
	    } else {
		q_node* part = (*(x->pert_end))->Q();
		part->turn();
		x->merge (x->pert_end);
		x->pert_end = x->pert_begin;
		x->pert_begin = part->pert_begin;
		x->pert_end.reverse();
		// x->pert_begin.reverse();
		delete part;
	    }
 
	} else if (x->partial_pos[0] != x->pert_begin) {
	    return false;
	} else {
	    //
	    // Partial child is at pert_begin and x has at least one 
	    // empty child (i.e. pert_begin != sons.begin())
	    // 
	    
	    q_node* part = x->merge (x->pert_begin);
	    
	    if (x->pert_begin == x->pert_end) {
		x->pert_end = part->pert_end;
	    }
	    
	    x->pert_begin = part->pert_begin;
	    delete part;
	}
    } 
    
    if (!is_root) {
	x->father->partial (x->pos);
    } else {
	pert_root = x;
    }
    
    return true;
}


//------------------------------------------------------------------------ Q3
// Requirements:
//
// * x is the root of the pertinent subtree.
// * Q1 and Q2 didn't match
//   ==> at least one partial child
// * if there is only one partial child it must be at pert_end, and x must
//   have at least one empty and one full child.
//   if there are two partial children they must be at pert_begin and
//   pert_end. 
// 
bool pq_tree::Q3 (q_node* x) 
{
    if (x->partial_count > 2 || x->partial_count < 1) return false;
    
    if (x->partial_count == 1) {
	if (x->partial_pos[0] != x->pert_end) return false;
	
	//
	// One partial child at pert_end.
	//
	
    } else {
	if (x->partial_pos[0] != x->pert_end) {
	    if (x->partial_pos[1] != x->pert_end || 
		x->partial_pos[0] != x->pert_begin) return false;
	} else {
	    if (x->partial_pos[1] != x->pert_begin) return false;
	}
	
	//
	// One partial child at pert_begin and one at pert_end
	// 
    }
    
    q_node* part = (*(x->pert_end))->Q();
    part->turn();
    x->merge (x->pert_end);
    x->pert_end = part->pert_begin;
    x->pert_end.reverse();
    delete part;
    
    if (x->partial_count == 2) {
	part = x->merge (x->pert_begin);
	x->pert_begin = part->pert_begin;
	delete part;
    }
    
    pert_root = x;
    return true;
}




GTL_EXTERN std::ostream& operator<< (std::ostream& os, const pq_tree& tree)
{
    if (!tree.root) return os;
    
    int id = 0;
	std::pair<pq_node*, int> tmp;
	std::queue<std::pair<pq_node*, int> > qu;
    pq_node* act;
    
	os << "graph [\n" << "directed 1" << std::endl;
    tree.root->write (os, id);
    tmp.first = tree.root;
    tmp.second = id;
    ++id;
    qu.push (tmp);
    
    while (!qu.empty()) {
	tmp = qu.front();
	qu.pop();
	
	if (tmp.first->kind() == pq_node::Q_NODE || tmp.first->kind() == pq_node::P_NODE) {
            pq_tree::sons_iterator it = tmp.first->sons.begin();
            pq_tree::sons_iterator end = tmp.first->sons.end();
	    
	    for (; it != end; ++it) {
		act = *it;
		act->write (os, id);
		
		os << "edge [\n" << "source " << tmp.second << std::endl;
		os << "target " << id << "\n]" << std::endl;
		
		qu.push(std::pair<pq_node*, int>(act, id));
		++id;
	    }
        }

        if (tmp.first->kind() == pq_node::P_NODE) {
            p_node* P = tmp.first->P();
            pq_tree::sons_iterator it = P->full_sons.begin();
            pq_tree::sons_iterator end = P->full_sons.end();

            for (; it != end; ++it) {
                act = *it;
                act->write (os, id);

				os << "edge [\n" << "source " << tmp.second << std::endl;
				os << "target " << id << "\n]" << std::endl;

				qu.push(std::pair<pq_node*, int>(act, id));
                ++id;
            }
            
            it = P->partial_sons.begin();
            end = P->partial_sons.end();

            for (; it != end; ++it) {
                act = *it;
                act->write (os, id);

				os << "edge [\n" << "source " << tmp.second << std::endl;
				os << "target " << id << "\n]" << std::endl;

				qu.push(std::pair<pq_node*, int>(act, id));
                ++id;
            }            
        }
    }
    
	os << "]" << std::endl;
    
    return os;
}

pq_tree::sons_iterator
pq_tree::remove_dir_ind(q_node* q_fail, sons_iterator s_it)
{
    direction_indicator* dir = (*s_it)->D();
    sons_iterator res = q_fail->sons.erase(s_it);
    clear_me.erase(dir->lpos);
    delete dir;
    return res;
}


//--------------------------------------------------------------------------
//   DEBUGGING 
//--------------------------------------------------------------------------

bool pq_tree::integrity_check () const
{    
    if (!root) return true;
    
	std::queue<pq_node*> qu;
    qu.push (root);
    pq_node* tmp;
    
    while (!qu.empty()) {
	tmp = qu.front();
	qu.pop();
	
	if (tmp->kind() == pq_node::LEAF) continue;
	if (tmp->kind() == pq_node::DIR) continue;
	
	sons_iterator it = tmp->sons.begin();
	sons_iterator end = tmp->sons.end();
	int count = 0;
	int endmost_count = 0;
	
	for (; it != end; ++it) {
	    ++count;
	    if ((*it)->is_endmost) {
		++endmost_count;
		
		if ((*it)->father != tmp) {
		    GTL_debug::debug_message ("Wrong father !!!\n");
 		    GTL_debug::close_debug();
		    return false;
		}
	    }
	    
	    if ((*it)->pos != it) {
		GTL_debug::debug_message ("Wrong position !!\n");
 		GTL_debug::close_debug();
		return false;
	    }
	    
	    qu.push (*it);
	}
	
	if (tmp->kind() == pq_node::P_NODE 
	    && count != (tmp->P()->child_count)) {
	    GTL_debug::debug_message ("Wrong number of children !!!\n");
	    GTL_debug::close_debug();
	    return false;
	}
	
	if (tmp->kind() == pq_node::Q_NODE && count < 2) {
	    GTL_debug::debug_message ("Q-Node with too few children !!\n"); 
	    GTL_debug::close_debug();
	    return false;
	} 
	
	if (tmp->kind() == pq_node::P_NODE && count < 2) {
	    GTL_debug::debug_message ("P-Node with too few children !!\n");
	    GTL_debug::close_debug();
	    return false;
	}
	
	if (tmp->kind() == pq_node::Q_NODE) {
	    if (endmost_count == 2) {
		if (!(tmp->sons.front()->is_endmost && 
		    tmp->sons.back()->is_endmost)) {
		    GTL_debug::debug_message ("Q-node with inner children labeled endmost\n");
		    GTL_debug::close_debug();
		    return false;
		} 
	    } else {
		GTL_debug::debug_message ("Q-node with too many or too few endmost children\n");
		GTL_debug::close_debug();
		return false;
	    }
	}
    }
    
    return true;
}

/*
void pq_tree::insert (pq_node* father, pq_node* ins) {
    ins->father = father;
    ins->is_endmost = true;
    
    if (father->kind() == pq_node::Q_NODE) {
	father->sons.back()->is_endmost = false;
    } else {
	((p_node*)father)->child_count++;
    }
    
    ins->pos = father->sons.insert (father->sons.end(), ins); 
}    


p_node* pq_tree::insert_P (pq_node* father, sons_list& sons) 
{
    p_node* p = new p_node();
    insert (father, p);
    pq_node* tmp;
    
    sons_iterator it = sons.begin();
    sons_iterator end = sons.end(); 
    
    for (; it != end; ++it) {
	p->child_count++;
	tmp = *it;
	tmp->father = p;
	tmp->is_endmost = true;
	tmp->pos  = p->sons.insert (p->sons.end(), tmp);
	
	if (tmp->kind() == pq_node::LEAF) {
	    leaves.push_back ((pq_leaf*)tmp);
	}
    }
    
    return p;
}


q_node* pq_tree::insert_Q (pq_node* father, sons_list& sons) 
{
    q_node* q = new q_node();
    insert (father, q);
    pq_node* tmp;
    sons_iterator it = sons.begin();
    sons_iterator end = sons.end(); 
    
    for (; it != end; ++it) {
	tmp = *it;
	tmp->is_endmost = false;
	tmp->pos = q->sons.insert (q->sons.end(), tmp);
	
	if (tmp->kind() == pq_node::LEAF) {
	    leaves.push_back (tmp->L());
	}
    }
    
    q->sons.front()->father = q;
    q->sons.front()->is_endmost = true;
    q->sons.back()->father = q;
    q->sons.back()->is_endmost = true;
    
    return q;
}

*/

__GTL_END_NAMESPACE

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
