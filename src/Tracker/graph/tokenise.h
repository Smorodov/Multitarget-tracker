/*
 *  tokenise.h
 *  taxonnames
 *
 *  Created by Roderic Page on Fri Apr 04 2003.
 *  Copyright (c) 2001 __MyCompanyName__. All rights reserved.
 *
 */
 
#ifndef TOKENISE_H
#define TOKENISE_H

#include <string>
#include <vector>
#include <algorithm>

void Tokenise (std::string s, std::string delimiters, std::vector<std::string> &tokens);

#endif
