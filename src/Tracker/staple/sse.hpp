/*******************************************************************************
* Piotr's Computer Vision Matlab Toolbox      Version 3.23
* Copyright 2014 Piotr Dollar.  [pdollar-at-gmail.com]
* Licensed under the Simplified BSD License [see external/bsd.txt]
*******************************************************************************/
#ifndef _SSE_HPP_
#define _SSE_HPP_
#include <emmintrin.h> // SSE2:<e*.h>, SSE3:<p*.h>, SSE4:<s*.h>

namespace sse{

#define RETf inline __m128
#define RETi inline __m128i

// set, load and store values
RETf SET( const float &x ) { return _mm_set1_ps(x); }
RETf SET( float x, float y, float z, float w ) { return _mm_set_ps(x,y,z,w); }
RETi SET( const int &x ) { return _mm_set1_epi32(x); }
RETf LD( const float &x ) { return _mm_load_ps(&x); }
RETf LDu( const float &x ) { return _mm_loadu_ps(&x); }
RETf STR( float &x, const __m128 y ) { _mm_store_ps(&x,y); return y; }
RETf STR1( float &x, const __m128 y ) { _mm_store_ss(&x,y); return y; }
RETf STRu( float &x, const __m128 y ) { _mm_storeu_ps(&x,y); return y; }
RETf STR( float &x, const float y ) { return STR(x,SET(y)); }

// arithmetic operators
RETi ADD( const __m128i x, const __m128i y ) { return _mm_add_epi32(x,y); }
RETf ADD( const __m128 x, const __m128 y ) { return _mm_add_ps(x,y); }
RETf ADD( const __m128 x, const __m128 y, const __m128 z ) {
    return ADD(ADD(x,y),z); }
RETf ADD( const __m128 a, const __m128 b, const __m128 c, const __m128 &d ) {
    return ADD(ADD(ADD(a,b),c),d); }
RETf SUB( const __m128 x, const __m128 y ) { return _mm_sub_ps(x,y); }
RETf MUL( const __m128 x, const __m128 y ) { return _mm_mul_ps(x,y); }
RETf MUL( const __m128 x, const float y ) { return MUL(x,SET(y)); }
RETf MUL( const float x, const __m128 y ) { return MUL(SET(x),y); }
RETf INC( __m128 &x, const __m128 y ) { return x = ADD(x,y); }
RETf INC( float &x, const __m128 y ) { __m128 t=ADD(LD(x),y); return STR(x,t); }
RETf DEC( __m128 &x, const __m128 y ) { return x = SUB(x,y); }
RETf DEC( float &x, const __m128 y ) { __m128 t=SUB(LD(x),y); return STR(x,t); }
RETf MIN( const __m128 x, const __m128 y ) { return _mm_min_ps(x,y); }
RETf RCP( const __m128 x ) { return _mm_rcp_ps(x); }
RETf RCPSQRT( const __m128 x ) { return _mm_rsqrt_ps(x); }

// logical operators
RETf AND( const __m128 x, const __m128 y ) { return _mm_and_ps(x,y); }
RETi AND( const __m128i x, const __m128i y ) { return _mm_and_si128(x,y); }
RETf ANDNOT( const __m128 x, const __m128 y ) { return _mm_andnot_ps(x,y); }
RETf OR( const __m128 x, const __m128 y ) { return _mm_or_ps(x,y); }
RETf XOR( const __m128 x, const __m128 y ) { return _mm_xor_ps(x,y); }

// comparison operators
RETf CMPGT( const __m128 x, const __m128 y ) { return _mm_cmpgt_ps(x,y); }
RETf CMPLT( const __m128 x, const __m128 y ) { return _mm_cmplt_ps(x,y); }
RETi CMPGT( const __m128i x, const __m128i y ) { return _mm_cmpgt_epi32(x,y); }
RETi CMPLT( const __m128i x, const __m128i y ) { return _mm_cmplt_epi32(x,y); }

// conversion operators
RETf CVT( const __m128i x ) { return _mm_cvtepi32_ps(x); }
RETi CVT( const __m128 x ) { return _mm_cvttps_epi32(x); }

#undef RETf
#undef RETi

}
#endif
