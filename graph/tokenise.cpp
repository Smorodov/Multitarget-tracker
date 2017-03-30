/*
 *  tokenise.cpp
 *  taxonnames
 *
 *  Created by Roderic Page on Fri Apr 04 2003.
 *  Copyright (c) 2001 __MyCompanyName__. All rights reserved.
 *
 */

#include "tokenise.h"

void Tokenise (std::string s, std::string delimiters, std::vector<std::string> &tokens)
{
	tokens.erase (tokens.begin(), tokens.end());
	int start, stop;
	int n = s.length();
	start = s.find_first_not_of (delimiters);
	while ((start >= 0) && (start < n))
	{
		stop = s.find_first_of (delimiters, start);
		if ((stop < 0) || (stop > n)) stop = n;
		tokens.push_back (s.substr(start, stop - start));
		start = stop + delimiters.length();
	}
}


