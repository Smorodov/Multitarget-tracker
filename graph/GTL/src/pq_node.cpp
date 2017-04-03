/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   pq_node.cpp 
//
//==========================================================================
// $Id: pq_node.cpp,v 1.12 2002/10/04 08:07:36 chris Exp $

#include <GTL/pq_node.h>

__GTL_BEGIN_NAMESPACE

pq_node::~pq_node () 
{
    while (!sons.empty()) {
	pq_node* tmp = sons.front();
	sons.erase (sons.begin());
	delete tmp;
    }
}
	

//--------------------------------------------------------------------------
//   P-NODE
//--------------------------------------------------------------------------

p_node::p_node (node n_, int id_) : pq_node (n_, id_), partial_count (0), full_count (0)
{
}

p_node::p_node (node n_, int id_, symlist<pq_node*>& s) : 
	pq_node (n_, id_), child_count (0), partial_count (0), full_count (0)
{
    sons.splice (sons.end(), s.begin(), s.end());

    iterator it = sons.begin();
    iterator end = sons.end();

    for (; it != end; ++it) {
	++child_count;
	(*it)->is_endmost = true;
	(*it)->father = this;
    }
} 

void p_node::clear () 
{
    pq_node::clear(); 
    partial_count = full_count = 0;
    if (!full_sons.empty())
	sons.splice (sons.end(), full_sons.begin(), full_sons.end());

    if (!partial_sons.empty())
	sons.splice (sons.end(), partial_sons.begin(), partial_sons.end());
}

inline void p_node::partial (iterator it) 
{
    ++partial_count;
    pert_leaves += (*it)->pert_leaves;
    partial_sons.splice (partial_sons.end(), it);
}

inline void p_node::full (iterator it) 
{
    ++full_count;
    pert_leaves += (*it)->pert_leaves;
    full_sons.splice (full_sons.end(), it);
}


inline void p_node::write(std::ostream& os, int _id)
{
	os << "node [\n" << "id " << _id << std::endl;
    os << "label \"" << id << "\nP" << "\"\n";
    os << "graphics [\n" << "x 100\n" << "y 100\n"; 
    if (mark == UNBLOCKED) {
	os << "outline \"#0000ff\"\n";
    } else if (mark == BLOCKED) {
	os << "outline \"#ff0000\"\n";
    }
	os << "type \"oval\"\n" << "]" << std::endl;
    os << "LabelGraphics [\n";
	os << "type \"text\"\n]\n]" << std::endl;
} 

//--------------------------------------------------------------------------
//   Q-NODE
//--------------------------------------------------------------------------

q_node::q_node (node n_, int id_) : pq_node (n_, id_), partial_count (0), full_count (0)
{ 
}

inline void q_node::partial (iterator it) 
{
    if (partial_count < 3) {
	partial_pos[partial_count] = it;
    }
    
    pert_leaves += (*it)->pert_leaves;
    ++partial_count;
    
    if (pert_begin == iterator()) {
	pertinent (it);
    }
}


inline void q_node::full (iterator it)
{
    ++full_count;
    pert_leaves += (*it)->pert_leaves;


    if (pert_begin == iterator()) {
	pertinent (it);
    }
}


void q_node::pertinent (iterator it) 
{
    iterator end = sons.end();
    iterator tmp = it;
    pq_node* first;
    pq_node* last;
    pert_end = it;
    ++tmp;
    int pert_block_count = 1;

    while (tmp != end) {
	if ((*tmp)->mark != UNBLOCKED) {
	    break;
	}

	if ((*tmp)->kind () != DIR) {
	    ++pert_block_count;
	    pert_end = tmp;
	} 

	++tmp;
    }

    last = *pert_end;
    
    pert_begin = tmp = it;
    --tmp;

    while (tmp != end) {
	if ((*tmp)->mark != UNBLOCKED) {
	    break;
	}
	
	if ((*tmp)->kind () != DIR) {
	    ++pert_block_count;
	    pert_begin = tmp;
	} 

	--tmp;
    }
    
    first = *pert_begin;
    pert_cons = (pert_block_count == pert_children);

    //
    // it must be true, that either first or last is in 
    // {sons.front(), sons.back()} (or both). Thus we are able to achive
    // the following normalization: pert_end is *always* sons.last() and
    // pert_begin is some other child, such that ++pert_begin leads towards 
    // pert_end.
    //
    
    if (pert_cons) {
	if (last == sons.front()) {
	    turn();
	} else if (last != sons.back()) {
	    tmp = pert_begin;
	    pert_begin = pert_end;
	    pert_end = tmp;
	    pert_end.reverse();
	    pert_begin.reverse();

	    if (first == sons.front()) {
		turn();
	    } else if (first != sons.back()) {		

		//
		// This should not happen. In this case the pertinent children are 
		// BLOCKED and thus this method would´t be called.
		//
		// 17.3. Now this can happen. 
		// 

		// pert_cons = false;

		// assert (false);
	    }
	}

    } else {

	//
	// In case that there are more than one block of pertinent children although 
	// bubble up didn´t fail (e.g. pp...pe...ep...pp or ee...ep...pe...ep...pp) 
	// we need some element of the second block in order to find K5 or K33. So we 
	// leave pert_begin as it is, but assign pert_end to something in the second 
	// block
	//

	tmp = pert_begin;
	--tmp;

	while (tmp != sons.end()) {
	    if ((*tmp)->mark == UNBLOCKED && (*tmp)->kind () != DIR) {
		break;
	    }

	    --tmp;
	}


	//
	// We need an empty child. So we always keep the invariant that --pert_end 
	// leads to an empty child. Please note that --pert_end might be a DI.
	//

	tmp.reverse();

	if (tmp == sons.end()) {
	    tmp = pert_end;
	    ++tmp;

	    while (tmp != sons.end()) {
		if ((*tmp)->mark == UNBLOCKED && (*tmp)->kind () != DIR) {
		    break;
		}
		
		++tmp;
	    }

	    assert (tmp != sons.end());
	}

	pert_end = tmp;
    }    
    
    //
    // In the case that there is in fact only one pertinent child we so far 
    // only assured that it is the last child, but it is still possible
    // that pert_begin (and pert_end, too) is headed the wrong way.
    // 

    if (pert_begin == pert_end && pert_cons && pert_end == --(sons.end())) {
	pert_begin = pert_end = --(sons.end());
    }
}

inline void q_node::clear () 
{
    pq_node::clear(); 
    partial_count = full_count = 0;
    pert_begin = symlist<pq_node*>::iterator();
    pert_end = symlist<pq_node*>::iterator();
}

inline void q_node::write(std::ostream& os, int _id)
{
	os << "node [\n" << "id " << _id << std::endl;
    os << "label \"" << id << "\n" << "Q" << "\"\n";
    os << "graphics [\n" << "x 100\n" << "y 100 \n"; 
    if (mark == UNBLOCKED) {
	os << "outline \"#0000ff\"\n";
    } else if (mark == BLOCKED) {
	os << "outline \"#ff0000\"\n";
    }
    os << "]\n";
    os << "LabelGraphics [\n";
	os << "type \"text\"\n]\n]" << std::endl;
} 

q_node* q_node::merge (iterator it) 
{
    assert ((*it)->kind() == pq_node::Q_NODE);
    q_node* part = (q_node*) *it;
    
    if (part == sons.front()) {
	part->sons.front()->father = this;
	part->sons.back()->is_endmost = false;
    } else if (part == sons.back()){
	part->sons.back()->father = this;	    
	part->sons.front()->is_endmost = false;
    } else {
	part->sons.front()->is_endmost = false;
	part->sons.back()->is_endmost = false;
    }

    sons.splice (it, part->sons.begin(), part->sons.end());
    sons.erase (it);

    return part;
}


void q_node::turn () 
{
    sons.reverse();
}
 
   
//--------------------------------------------------------------------------
//   LEAF
//--------------------------------------------------------------------------


pq_leaf::pq_leaf (int id_, int other_, edge e_, node n_) : pq_node (n_, id_) 
{
    up_id = other_;
    up = n_.opposite (e_);
    other_id = other_;
    e = e_;
}

inline void pq_leaf::write(std::ostream& os, int _id)
{
	os << "node [\n" << "id " << _id << std::endl;
    os << "label \"" << other_id << "\n" << id << "\"\n";
    os << "graphics [\n" << "x 100\n" << "y 100 \n"; 
    if (mark == UNBLOCKED) {
	os << "outline \"#0000ff\"\n";
    } else if (mark == BLOCKED) {
	os << "outline \"#ff0000\"\n";
    }
    os << "]\n";
    os << "LabelGraphics [\n";
	os << "type \"text\"\n]\n]" << std::endl;
} 


void direction_indicator::write(std::ostream& os, int _id)
{
	os << "node [\n" << "id " << _id << std::endl;
    os << "label \"DIR\n" << id << "\"\n";
    os << "graphics [\n" << "x 100\n" << "y 100 \n"; 
    if (mark == UNBLOCKED) {
	os << "outline \"#0000ff\"\n";
    } else if (mark == BLOCKED) {
	os << "outline \"#ff0000\"\n";
    }
    os << "]\n";
    os << "LabelGraphics [\n";
	os << "type \"text\"\n]\n]" << std::endl;
}

__GTL_END_NAMESPACE

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
