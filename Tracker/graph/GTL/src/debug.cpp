/* This software is distributed under the GNU Lesser General Public License */
//==========================================================================
//
//   debug.cpp
//
//==========================================================================
// $Id: debug.cpp,v 1.10 2001/11/07 13:58:09 pick Exp $

#include <GTL/debug.h>

#include <fstream>
#include <cstdarg>
#include <cstdio>

__GTL_BEGIN_NAMESPACE

std::ostream* GTL_debug::GTLerr = 0;

void GTL_debug::debug_message (const char* message, ...) 
{
#ifdef _DEBUG
    va_list arg_list;
    va_start(arg_list, message);

    char buf[1024];
    vsprintf(buf, message, arg_list);
    if (GTLerr) {
	os() << buf;
    }
#endif
}

void GTL_debug::init_debug () 
{
    if (!GTLerr) {
#ifdef __GTL_MSVCC
		GTLerr = new std::ofstream("ERRLOG.txt", std::ios::out | std::ios::app);
#else
	GTLerr = &std::cerr;
#endif
    }
}

void GTL_debug::close_debug () 
{
    if (GTLerr) {
#ifdef __GTL_MSVCC 
		((std::ofstream*) GTLerr)->close();
	delete GTLerr;
	GTLerr = 0;
#endif
    }
}

__GTL_END_NAMESPACE

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
