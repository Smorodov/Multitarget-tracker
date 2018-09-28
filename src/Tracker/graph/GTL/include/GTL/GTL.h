/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   GTL.h - Internal header: DO NO USE IT DIRECTLY !!!
//
//==========================================================================
// $Id: GTL.h,v 1.29 2008/02/03 18:17:08 chris Exp $

#ifndef GTL_GTL_H
#define GTL_GTL_H

#include <GTL/version.h>

//--------------------------------------------------------------------------
//   Generic iteration over container elements
//--------------------------------------------------------------------------
//
// elem: loop variable
// cont: container to iterate over
// iter_t: iterator type
// iter: prefix for begin() and end()
//
// contains a hack for Microsoft Visual C++ 5.0, because code like
//
//   for(int i=0; i<10; ++i) { ... do something ... }
//   for(int i=0; i<10; ++i) { ... do something again ... }
//
// is illegal with Microsoft Extensions enabled, but without Microsoft
// Extensions, the Microsoft STL does not work :-(.
// So we code the line number (__LINE__) into our loop variables.

#define GTL_CONCAT(x, y) x##y
#define GTL_FORALL_VAR(y) GTL_CONCAT(GTL_FORALL_VAR, y)

#define GTL_FORALL(elem, cont, iter_t, iter)			\
if ((cont).iter##begin() != (cont).iter##end())			\
    (elem) = *((cont).iter##begin());				\
for (iter_t GTL_FORALL_VAR(__LINE__) = (cont).iter##begin();    \
    GTL_FORALL_VAR(__LINE__) != (cont).iter##end();             \
    (elem) = (++GTL_FORALL_VAR(__LINE__)) ==                    \
        (cont).iter##end() ? (elem) : *GTL_FORALL_VAR(__LINE__))

//--------------------------------------------------------------------------
//   Configuration for GCC >= 2.8.0
//--------------------------------------------------------------------------

//
// Using namespaces is the default; may be unset by one of the 
// following configurations.
//
 
#define __GTL_USE_NAMESPACES

#ifdef __GNUC__

#  define __GTL_GCC

#  if __GNUC__ == 2 && __GNUC_MINOR__ >= 8

#    undef __GTL_USE_NAMESPACES

#  elif __GNUC__ < 3

#    error "Need at least version 2.8.0 of GCC to compile GTL."

#  endif

// 
// 2/3/2008 chris:
//
// Enable comparison of iterators in debug mode
//

#  if __GNUC__ >= 4
#    undef _GLIBCXX_DEBUG
#  endif
#endif

//--------------------------------------------------------------------------
//    Configuration for Microsoft Visual C++ 5.0
//--------------------------------------------------------------------------

#ifdef _MSC_VER
/*
#  if _MSC_VER >= 1400 // Visual Studio 2005

#    define _HAS_ITERATOR_DEBUGGING 0
#    define _CRT_SECURE_NO_DEPRECATE 1
#    define _SECURE_SCL 0

#  endif
*/
#  if _MSC_VER >= 1100
    
#    define __GTL_USE_NAMESPACES
#    define __GTL_MSVCC

#    pragma warning( disable : 4786 )
#    pragma warning( disable : 4251 )

#    define GTL_EXTERN
#  else

#    error "Need at least version 5.0 of MS Visual C++ to compile GTL."

#  endif
#else

#   define GTL_EXTERN

#endif

//--------------------------------------------------------------------------
//   Namespaces
//--------------------------------------------------------------------------

#ifdef __GTL_USE_NAMESPACES

#  define __GTL_BEGIN_NAMESPACE namespace GTL {
#  define __GTL_END_NAMESPACE }

#else

#  define __GTL_BEGIN_NAMESPACE
#  define __GTL_END_NAMESPACE

#endif

//--------------------------------------------------------------------------
//   Temporary hack until Graphlet (i.e. gcc) supports Namespaces
//--------------------------------------------------------------------------

#ifdef __GTL_USE_NAMESPACES

namespace GTL {}
using namespace GTL;

#endif // __GTL_USE_NAMESPACES

#endif // GTL_GTL_H
	
//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
