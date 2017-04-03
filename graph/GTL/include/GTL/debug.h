/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   debug.h - Functions, which are useful for debugging 
//
//==========================================================================
// $Id: debug.h,v 1.8 2001/10/10 08:30:00 chris Exp $

#ifndef GTL_DEBUG_H
#define GTL_DEBUG_H

#include <GTL/GTL.h>

#include <iostream>

__GTL_BEGIN_NAMESPACE

//
// If _DEBUG is defined the funtions defined here will produce output.
// You can either define _DEBUG here (or undef it) or you can set it as 
// option of your compiler.
//
//#define _DEBUG 1
//#undef _DEBUG
//

/**
 * @internal
 */
class GTL_EXTERN GTL_debug {
public:
    static void debug_message (const char*, ...); 
    static void init_debug();
    static void close_debug();
	static std::ostream& os()
	{ return *GTLerr; }

private:
	static std::ostream* GTLerr;
};

__GTL_END_NAMESPACE

#endif // GTL_DEBUG_H

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
