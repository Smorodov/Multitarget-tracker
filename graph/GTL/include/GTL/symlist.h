/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   symlist.h
//
//==========================================================================
// $Id: symlist.h,v 1.17 2002/12/20 08:26:08 chris Exp $

#ifndef SYMLIST_H
#define SYMLIST_H

#include <GTL/GTL.h> 

#include <cassert>

__GTL_BEGIN_NAMESPACE

/**
 * @internal
 */
template <class T>
struct symnode 
{
    /**
     * @internal
     */
    symnode()
    {
    }

    /**
     * @internal
     */
    symnode(const T& n) : data(n)
    {
    }

    /**
     * @internal
     */
    symnode* adj[2];

    /**
     * @internal
     */
    T data;
};

/**
 * @internal
 */
template <class T, class Ref>
struct symlist_iterator 
{
    /**
     * @internal
     */
    typedef symlist_iterator<T, Ref> self;

    /**
     * @internal
     */
    typedef symnode<T>* linktype;

    /**
     * @internal
     */
    symlist_iterator() : act (0)
    {
    }

    /**
     * @internal
     */
    symlist_iterator(const self& it) : act(it.act), dir(it.dir)
    {
    }

    /**
     * @internal
     */
    symlist_iterator(linktype _act, int _dir) : act(_act), dir(_dir)
    {
    }
    
    /**
     * @internal
     */
    symlist_iterator(linktype _act, linktype _prev) :
	act(_act),
 	dir (where_not (_act, _prev))
    {
    }

    /**
     * @internal
     */
    self& operator=(const self& it)
    {
	act = it.act;
	dir = it.dir;
	return *this;
    }

    /**
     * @internal
     */
    bool operator==(const self& it) const
    {
	return act == it.act;
    }
    
    /**
     * @internal
     */
    bool operator!=(const self& it) const
    {
	return act != it.act;
    }
    
    /**
     * @internal
     */
    Ref operator*() 
    {
	return act->data;
    }

    /**
     * @internal
     */
    self& operator++();

    /**
     * @internal
     */
    self& operator--();
    
    /**
     * @internal
     */
    static int where(linktype _act, linktype _prev) 
    {
	return _prev == _act->adj[0] ? 0 : 1;
    }

    /**
     * @internal
     */
    static int where_not(linktype _act, linktype _prev)
    {
	return _prev == _act->adj[1] ? 0 : 1;
    }

    /**
     * @internal
     */
    void reverse()
    {
	dir = 1 - dir;
    }
	
    /**
     * @internal
     */
    linktype& next()
    {
	return act->adj[dir];
    }

    /**
     * @internal
     */
    linktype& prev()
    {
	return act->adj[1 - dir];
    }

    /**
     * @internal
     */
    linktype act;

    /**
     * @internal
     */
    int dir;
};

/**
 * @brief List which can be reversed in @f$\mathcal{O}(1)@f$. 
 *
 * The problem with the STL class list - as with most doubly linked lists --
 * is that isn't possible to turn it in constant time, because each entry in
 * the list contains next and prev pointer and turning the list means to
 * switch these two in @em each element in the list. Another point is the
 * splice operation in STL lists, which is constant time, but for the same
 * reason as mentioned above it is not possible to splice a list in reverse
 * order into another in constant time.
 * <p>
 * The problems arise from the fact that each element "knows" what its next
 * and previous elements are. An element in a symlist only knows what its
 * neighbors are, what is next and what previous depends on the direction of
 * iteration. This of course imposes some overhead in iteration (one
 * if-statement) but allows reversion and a splice in reversed order in
 * constant time.
 */
template <class T>
class symlist 
{
public:
    /**
     * @internal
     */
    typedef symlist_iterator<T, T&> iterator;

    /**
     * @internal
     */
    typedef symlist_iterator<T, const T&> const_iterator;

    /**
     * @brief Creates empty symlist.
     */
    symlist()
    {
	link = new symnode<T>;
	link->adj[0] = link->adj[1] = link;
    }

    /**
     * @brief Makes the created list a copy of @c li.
     *
     * @param li symlist.
     */
    symlist(const symlist<T>& li);

    /**
     * @brief Assignes @c li to this list.
     *
     * @note All elements in this list will be deleted.
     *
     * @param li 
     *
     * @return this list
     */
    symlist<T>& operator=(const symlist<T>& li);

    /**
     * @brief Destructor 
     */
    ~symlist();

    /**
     * @brief Checks whether list is empty.
     *
     * Takes constant time.
     *
     * @retval true iff list is empty
     */
    bool empty() const
    {
	return link->adj[0] == link && link->adj[1] == link;
    }

    /**
     * @brief First element in list.
     *
     * Assumes that list ins't empty.
     *
     * @return first element
     */
    T& front()
    {
	return link->adj[0]->data;
    }

    /**
     * @brief Last element in list.
     *
     * Assumes that list ins't empty.
     *
     * @return last element
     */
    T& back()
    {
	return link->adj[1]->data;
    }

    /**
     * @brief Start iteration through elements of list.
     *
     * @return start iterator
     */
    iterator begin()
    {
	return ++end();
    }

    /**
     * @brief End of iteration through elements of list.
     *
     * @return end iterator
     */
    iterator end()
    {
	return iterator(link, 0);
    }

    /**
     * @brief Start iteration through elements of list.
     *
     * @return start iterator
     */
    const_iterator begin() const
    {
	return ++end();
    }

    /**
     * @brief End of iteration through elements of list.
     *
     * @return end iterator
     */
    const_iterator end () const
    {
	return const_iterator (link, 0);
    }

    /**
     * @brief Start iteration through element of list in reverse order.
     *
     * @return start iterator 
     */
    iterator rbegin()
    {
	return ++rend();
    }

    /**
     * @brief End of iteration through elements of list in reverse order.
     *
     * @return end iterator
     */
    iterator rend()
    {
	return iterator (link, 1);
    }

    /**
     * @brief Start iteration through element of list in reverse order.
     *
     * @return start iterator 
     */
    const_iterator rbegin() const
    {
	return ++rend();
    }

    /**
     * @brief End of iteration through elements of list in reverse order.
     *
     * @return end iterator
     */
    const_iterator rend() const
    {
	return const_iterator(link, 1);
    }

    /**
     * @brief Inserts @p data before @p pos in list.
     *
     * @param pos position
     * @param data element to be inserted
     *
     * @return position of insertion
     */
    iterator insert (iterator pos, const T& data);

    /**
     * @brief Inserts the element @p it points to before @p pos into this
     *	      list.
     *
     * It is assumed that the element @p it refers lies in a different list.
     * All iterators to elements in either of the two lists stay valid.
     * Takes constant time.
     *
     * @param pos position 
     * @param it position of element to be inserted
     */
    void splice (iterator pos, iterator it);

    /**
     * @brief Inserts the elements <tt>[it,end)</tt> refers to before @p pos
     *	      into this list.
     *
     * It is assumed that <tt>[it,end)</tt> lies in a different
     * list. All iterators to elements in either of the two lists stay
     * valid. Takes constant time.
     *
     * @param pos position 
     * @param it position of first element to be inserted
     * @param end position of one-past the last element to be inserted
     */
    void splice (iterator pos, iterator it, iterator end);

    /**
     * @brief Deletes element at position @p pos from list.
     *
     * @param pos position to be deleted
     *
     * @return position of next element
     */
    iterator erase (iterator pos);

    /**
     * @brief Deletes the elements <tt>[it, end)</tt> from list.
     *
     * @param it first position to be deleted
     * @param end one-past the last position to be deleted
     *
     * @return position of next element.
     */
    iterator erase (iterator it, iterator end);

    /**
     * @internal
     */
    void attach_sublist (iterator, iterator);

    /**
     * @internal
     */
    void detach_sublist ();

    /**
     * @brief Change the direction of list.
     *
     * Takes constant time.
     */
    void reverse ();
private:
    /**
     * @internal
     */
    symnode<T>* link;

    /**
     * @internal
     *
     * @note Needed only when used as sublist.
     */
    iterator _prev;

    /**
     * @internal
     *
     * @note Needed only when used as sublist.
     */
    iterator _next;
};


// Implementation Begin

template <class T, class Ref>
symlist_iterator<T, Ref>& symlist_iterator<T, Ref>::operator++()
{
    symnode<T>* prev = act; 
    act = act->adj[dir];
    dir = where_not(act, prev);
    return *this;
}


template <class T, class Ref>
symlist_iterator<T, Ref>& symlist_iterator<T, Ref>::operator--()
{
    symnode<T>* prev = act;
    act = act->adj[1 - dir];
    dir = where(act, prev); 
    return *this;
}


template <class T>
symlist<T>::symlist (const symlist<T>& l)
{
    link = new symnode<T>;
    link->adj[0] = link->adj[1] = link;    
    
    const_iterator it = l.begin();
    const_iterator e = l.end();

    while (it != e)
    {
	insert(end(), *it);
	++it;
    }
}


template <class T>
symlist<T>::~symlist()
{
    if (_next == iterator())
    {
	erase (begin(), end());
    }
    else
    {
	detach_sublist();
    }
    
    delete link;
}


template <class T>
symlist<T>& symlist<T>::operator=(const symlist<T>& l) 
{
    erase(begin(), end());

    const_iterator it = l.begin();
    const_iterator e = l.end();

    while (it != e)
    {
	insert(end(), *it);
	++it;
    }
    
    return *this;
}


template <class T>
symlist_iterator<T, T&> symlist<T>::insert(
    symlist_iterator<T,T&> pos,
    const T& ins)
{
    iterator prev = pos;
    --prev;
    symnode<T>* n = new symnode<T>(ins);
    n->adj[0] = pos.act;
    n->adj[1] = prev.act;

    if (pos == prev)
    {
	pos = prev;
    }

    pos.prev() = n;
    prev.next() = n;

    return iterator(n, 0);
}


template <class T>
void symlist<T>::splice(symlist_iterator<T, T&> pos,
			symlist_iterator<T, T&> beg,
			symlist_iterator<T, T&> end) 
{
    if (beg != end)
    {
	iterator prev = beg;
	--prev;
	iterator last = end;
	--last;

	//
	// The following seems to be rather senseless, but it is required
	// since two iterator are equal, iff the point to the same element.
	// This implies that they might have different directions. Suppose
	// that prev == end is true and they have different directions,
	// than prev.next() and end.prev() return the same element !! Thus
	// the assignment prev = end corrects this, since the direction
	// gets copied, too.
	//
	if (prev == end)
	{
	    prev = end;
	}
	    
	prev.next() = end.act;
	end.prev() = prev.act;

	prev = pos;
	--prev;

	if (pos == prev)
	{
	    pos = prev;
	}

	if (last == beg)
	{
	    last = beg;
	}
	
	prev.next() = beg.act;
	beg.prev() = prev.act;
	pos.prev() = last.act;
	last.next() = pos.act;
    }
}


template <class T>
void symlist<T>::splice(symlist_iterator<T,T&> pos, 
			symlist_iterator<T,T&> beg)
{
    iterator tmp = beg;
    ++tmp;
    splice(pos, beg, tmp);
}


template <class T>
symlist_iterator<T,T&> symlist<T>::erase(symlist_iterator<T,T&> pos) 
{
    assert (pos.act != link);
    iterator prev = pos;
    --prev;
    iterator next = pos;
    ++next;
    
    if (next == prev)
    {
	next = prev;
    }

    next.prev() = prev.act;
    prev.next() = next.act;

    delete (pos.act);

    return next;
}

template <class T>
symlist_iterator<T,T&> symlist<T>::erase(symlist_iterator<T,T&> beg, 
					 symlist_iterator<T,T&> end)
{
    iterator prev = beg;
    --prev;
    iterator it = beg;
    symnode<T>* act;

    while (it != end)
    {
	assert (it.act != link);
	act = it.act;
	++it;
	delete (act);
    }

    if (prev == end)
    {
	prev = end;
    }	
    
    end.prev() = prev.act;
    prev.next() = end.act;
   
    return end;
}    


template <class T>
void symlist<T>::attach_sublist(symlist_iterator<T,T&> it, 
				symlist_iterator<T,T&> end) 
{ 
    assert (empty());
    iterator last = end;
    --last;
    _prev = it;
    --_prev;
    _next = end;

    if (it == last)
    {
	it = last;
    }

    link->adj[0] = it.act;
    it.prev() = link;
    link->adj[1] = last.act;
    last.next() = link;
}   

    
template <class T>
void symlist<T>::detach_sublist() 
{
    if (_next != iterator())
    {
	iterator it(begin());
	iterator e(end());
	
	--e;
	
	if (e == it)
	{
	    e = it;
	}

	_prev.next() = it.act;
	it.prev() = _prev.act;
	_next.prev() = e.act;
	e.next() = _next.act;
	link->adj[0] = link->adj[1] = link;

	_next = iterator();
	_prev = iterator();
    }
}


template <class T>
inline void symlist<T>::reverse()
{
    symnode<T>* tmp = link->adj[0];
    link->adj[0] = link->adj[1];
    link->adj[1] = tmp;
}

// Implementation End

__GTL_END_NAMESPACE

#endif // SYMLIST_H

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
