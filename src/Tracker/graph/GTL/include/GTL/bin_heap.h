/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   bin_heap.h
//
//==========================================================================
// $Id: bin_heap.h,v 1.10 2003/01/07 07:01:05 chris Exp $

#ifndef GTL_BIN_HEAP_H
#define GTL_BIN_HEAP_H

#include <GTL/GTL.h>

#include <cassert>
#include <vector>
#include <map>

__GTL_BEGIN_NAMESPACE

/**
 * @internal
 * Node type of container.
 */
template <class T>
class heap_node
{
public:
    /**
     * @internal
     * Default constructor.
     */
    heap_node()
    {
    }

    /**
     * @internal
     */
    heap_node(const T& n) : data(n)
    {
    }

    /**
     * @internal
     * Data member.
     */
    T data;

    /**
     * @internal
     * Position in container.
     */
    int pos;
};

    
/**
 * @brief Binary heap.
 *
 * @author Christian Bachmaier chris@infosun.fmi.uni-passau.de
 */
template <class T, class Pred>
class bin_heap
{
public:
    /**
     * @brief Creates empty binary heap.
     *
     * @param prd binary predicate to compare two <code>T</code>s
     */
    bin_heap(const Pred& prd);

    /**
     * @brief Creates empty binary heap.
     *
     * @param prd binary predicate to compare two @c Ts
     * @param est_size estimated maximal size of heap
     */
    bin_heap(const Pred& prd, const int est_size);

    /**
     * @brief Copy constructor.
     *
     * @param bh binary heap to copy
     */
    bin_heap(const bin_heap<T, Pred>& bh);

    /**
     * @brief Assigns @c bh to this binary heap.
     *
     * All elements in this heap will be deleted. The predicate of this heap
     * must be physically the same as the one of @p bh.
     * 
     * @param bh binary heap
     *
     * @return this heap
     */
    bin_heap<T, Pred>& operator=(const bin_heap<T, Pred>& bh);
    
    /**
     * @brief Destructor.
     */
    ~bin_heap();

    /**
     * @brief Inserts @p ins in heap.
     *
     * @param ins data element to be inserted
     */
    void push(const T& ins);

    /**
     * @brief Removes the element on top of the heap.
     */
    void pop();

    /**
     * @brief Returns a reference to the element at the top of the heap.
     *
     * @return top element of the heap
     */
    const T& top() const;
    
    /**
     * @brief Reconstructs heap condition after changing key value of @p
     * cha externally.
     *
     * @param cha element with changed key value
     *
     * @note @c changeKey doesn't operate if @p cha is a primitive data
     * structure, because it represents its key value itself, or if one
     * object is stored more than once in the data structure.
     *
     * @sa dijkstra
     */
    void changeKey(const T& cha);

    /**
     * @brief Checks if heap is empty.
     *
     * @return @c true iff empty
     */
    bool is_empty() const;

    /**
     * @internal
     * Makes heap empty.
     */
    void clear();
private:
    /**
     * @internal
     * Binary predicate to compare two <code>T</code>'s.
     */
    const Pred& prd;

    /**
     * @internal
     * Next free position in @a container.
     */
    int size;

    /**
     * @internal
     * Estimated maximum size of @a container. Initially set to estimated
     * size of user in constructor #bin_heap.
     */
    int capacity;

    /**
     * @internal
     * Data container.
     */
	std::vector<heap_node<T>* > container;

    /**
     * @internal
     * Mapping between data member T and its heap_node.
     */
	std::map<T, heap_node<T>* > heap_node_map;

    /**
     * @internal
     * Reconstructs heap condition with bubbling up heap_node @p n.
     */
    void bubble_up(heap_node<T>* const n);

    /**
     * @internal
     * Reconstructs heap condition with bubbling down heap_node @p n.
     */
    void bubble_down(heap_node<T>* const n);
#ifdef _DEBUG
public:
    /**
     * @internal
     * Prints @a container for debug purposes.
     */
    void print_data_container();
#endif	// _DEBUG
};

// Implementation Begin

template <class T, class Pred>
bin_heap<T, Pred>::bin_heap(const Pred& prd) :
    prd(prd), size(0), capacity(50)
{
    container.resize(capacity);
}


template <class T, class Pred>
bin_heap<T, Pred>::bin_heap(const Pred& prd, const int est_size) :
    prd(prd), size(0), capacity(50)
{
    if (est_size > 50)
    {
	capacity = est_size;
    }
    container.resize(capacity);
}


template <class T, class Pred>
bin_heap<T, Pred>::bin_heap(const bin_heap<T, Pred>& bh) :
    prd(bh.prd), size(bh.size), capacity(bh.capacity)
{
    container.resize(capacity);
    for (int i = 0; i < size; ++i)
    {
	container[i] = new heap_node<T>(bh.container[i]->data);
    }
}


template <class T, class Pred>
bin_heap<T, Pred>& bin_heap<T, Pred>::operator=(const bin_heap<T, Pred>& bh)
{
    if (this != &bh)	// no self assignment
    {
	assert(&prd == &(bh.prd));
	clear();
	size = bh.size;
	capacity = bh.capacity;
	container.resize(capacity);
	for (int i = 0; i < size; ++i)
	{
	    container[i] = new heap_node<T>(bh.container[i]->data);
	}
    }
    return *this;
}


template <class T, class Pred>
bin_heap<T, Pred>::~bin_heap()
{
    clear();
}


template <class T, class Pred>
void bin_heap<T, Pred>::push(const T& ins)
{
    if (size == capacity)
    {
	 // dynamic memory allocation
	capacity *= 2;
	container.resize(capacity);
    }
    heap_node<T>* n = new heap_node<T>(ins);
    n->pos = size;
    container[size] = n;
    heap_node_map[ins] = n;
    ++size;
    bubble_up(n);
}


template <class T, class Pred>
void bin_heap<T, Pred>::pop() 
{
    assert(size > 0);
    // save smallest element for return (ensured by heap condition)
    heap_node_map.erase(container[0]->data);
    delete container[0];
    // replace by last element in array and decrease heap "size"
    if (size > 1)
    {
	container[0] = container[--size];
	container[0]->pos = 0;
	// reorder heap to ensure heap conditions
	bubble_down(container[0]);
    }
    else
    {
	size = 0;
    }
}


template <class T, class Pred>
const T& bin_heap<T, Pred>::top() const
{
    return container[0]->data;
}


template <class T, class Pred>
void bin_heap<T, Pred>::changeKey(const T& cha)
{
    int pos = heap_node_map[cha]->pos;
    heap_node<T>* n = container[pos];
    if (pos != 0)
    {
	heap_node<T>* father = container[(pos - 1) / 2];
	if (prd(n->data, father->data))
	{
	    bubble_up(n);
	    return;
	}
    }
    bubble_down(n);
}


template <class T, class Pred>
bool bin_heap<T, Pred>::is_empty() const
{
    // empty if if first free index is 0
    return size == 0;
}
  

template <class T, class Pred>
void bin_heap<T, Pred>::clear()
{
    for (int i = 0; i < size; ++i)
    {
	delete container[i];
    }
    size = 0;
    heap_node_map.clear();
}

  
template <class T, class Pred>
void bin_heap<T, Pred>::bubble_up(heap_node<T>* const n)
{
    int pos = n->pos;
    // if we are not already at top AND the parent in heap is more
    while ((pos != 0) &&
	   (prd(n->data, container[(pos - 1) / 2]->data)))
    {
	// move father down
	container[pos] = container[(pos - 1) / 2];
	container[pos]->pos = pos;
	// increment k to parent index
	pos = (pos - 1) / 2;
    }
    // place value in its highest position in heap
    container[pos] = n;
    container[pos]->pos = pos;
}


template <class T, class Pred>
void bin_heap<T, Pred>::bubble_down(heap_node<T>* const n)
{
    int pos = n->pos;
    int j = 0;
    while (pos < size / 2)
    {
	j = 2 * pos + 1;
	// if right child is smaller than left child get right child
	if ((j < size - 1) &&
	    (prd(container[j + 1]->data, container[j]->data)))
	{
	    ++j;
	}
	// if element is less or equal than its child leave it here
	if (!prd(container[j]->data, n->data))
	{
	    break;
	}
	// else move its child up
	container[pos] = container[j];
	container[pos]->pos = pos;
	// repeat for new position
	pos = j;
    }
    // place element into position, where heap condition is fulfilled
    container[pos] = n;
    container[pos]->pos = pos;
}
  
#ifdef _DEBUG
template <class T, class Pred>
void bin_heap<T, Pred>::print_data_container()
{
    if (size == 0)
    {
	cout << "empty";
    }
    else
    {
	for (int pos = 0; pos < size; ++pos)
	{
	    cout << container[pos]->data << " ";
	}
    }
    cout << endl;
}
#endif	// _DEBUG

// Implementation End

__GTL_END_NAMESPACE

#endif	// GTL_BIN_HEAP_H

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
